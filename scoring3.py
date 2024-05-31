import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from transformers import pipeline
from functools import lru_cache
import urllib.parse
import time
import openai

# Initialize the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load the classification model
classifier = pipeline("zero-shot-classification", model="sileod/deberta-v3-base-tasksource-nli")

# Define categories for classification
CATEGORIES = [
    "Software/Web Development", "Data Science/Engineering/Analytics", "Cloud Computing/DevOps",
    "Sales and Marketing", "HR and Administration", "Cybersecurity", "Finance and Accounting", "Graphic Design", "Customer Service/Support"
]

def clean_location(location):
    """Remove numbers, punctuation, and common terms, convert to lowercase tokens."""
    stopwords = {'street', 'road', 'avenue', 'blvd', 'lane', 'suite', 'st', 'rd', 'ave', 'ln', 'boulevard'}
    location = re.sub(r'[^\w\s]', '', location).lower()
    tokens = [token for token in location.split() if token not in stopwords and not token.isdigit()]
    return tokens

def categorize_location(candidate_tokens, job_location_tokens):
    """Determine if any candidate location token matches the job location tokens."""
    return 'In Your Location' if any(token in job_location_tokens for token in candidate_tokens) else 'Not In Your Location'

def categorize_scores(scores):
    categories = {
        'Top Matched Profiles': 0,
        'Highly Qualified Candidates': 0,
        'Well-Matched Candidates': 0,
        'Moderately Matched Candidates': 0,
        'Partially Matched Candidates': 0
    }
    for score in scores:
        if score >= 80:
            categories['Top Matched Profiles'] += 1
        elif score >= 70:
            categories['Highly Qualified Candidates'] += 1
        elif score >= 60:
            categories['Well-Matched Candidates'] += 1
        elif score >= 50:
            categories['Moderately Matched Candidates'] += 1
        else:
            categories['Partially Matched Candidates'] += 1
    return categories

def format_resume(resume_data):
    """Extract and format the resume data into a single string."""
    experiences = ' '.join([f"{exp['job_title']} {' '.join(exp['tasks'])}" for exp in resume_data['EXPERIENCE']])
    education = ' '.join([f"{edu['school_name']} {edu['graduation_title']}" for edu in resume_data['EDUCATION']])
    skills = ' '.join(resume_data['OTHER']['skills'])
    certificates = ' '.join(resume_data['OTHER']['certificates'])
    return f"{experiences} {education} {skills} {certificates}"

def format_job_offer(job_offer_data):
    """Extract and format the job offer data into a single string."""
    job_title = job_offer_data['Job Post Title']
    tasks = job_offer_data['Tasks']
    education_required = job_offer_data['Education Required']
    skills_needed = job_offer_data['Skills Needed']
    return f"{job_title} {tasks} {education_required} {skills_needed}"

@lru_cache(maxsize=128)
def calculate_similarity(resume_text, job_offer_text):
    """Calculate the cosine similarity between the resume and job offer embeddings."""
    resume_embedding = model.encode(resume_text)
    job_offer_embedding = model.encode(job_offer_text)
    return util.pytorch_cos_sim(resume_embedding, job_offer_embedding).item()

def plot_certificates_pie_chart(certificates_counts):
    """Use modern pie chart designs."""
    labels = ['With Certificates', 'Without Certificates']
    values = [certificates_counts['with'], certificates_counts['without']]
    custom_colors = ['#006769', '#D9EDBF']
    fig = px.pie(values=values, names=labels, title="Certificates Distribution", color_discrete_sequence=custom_colors)
    fig.update_traces(textposition='outside', textinfo='percent+label')
    fig.update_layout(font=dict(family="Arial, bold", size=12, color="black"))
    return fig

def plot_score_bar_chart(categories):
    """Use a gradient color bar chart."""
    labels = list(categories.keys())
    values = list(categories.values())
    colors = ['#003C43', '#135D66', '#77B0AA', '#CDE8E5', '#CDE8E5']
    fig = go.Figure(go.Bar(x=values, y=labels, orientation='h', marker=dict(color=colors)))
    fig.update_layout(title="Resume Scoring Distribution", 
                      yaxis=dict(categoryorder='total ascending'), 
                      template='plotly_dark', 
                      plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family="Arial, bold", size=12, color="white"))
    return fig

def plot_location_donut_chart(location_counts):
    """Generate a donut chart for candidate location distribution with a specified green color."""
    labels = list(location_counts.keys())
    values = list(location_counts.values())
    custom_color = ['#D9EDBF', '#006769']
    fig = px.pie(names=labels, values=values, title="Location Match Distribution", hole=0.6, color_discrete_sequence=custom_color)
    fig.update_traces(textposition='outside', textinfo='percent+label')
    fig.update_layout(font=dict(family="Arial, bold", size=12, color="black"))
    return fig

# Add the function to classify resumes and generate a DataFrame
def classify_resumes(resumes):
    classifications = []
    for resume in resumes:
        formatted_resume = format_resume(resume)
        result = classifier(formatted_resume, CATEGORIES)
        top_category = result['labels'][0]
        classifications.append(top_category)
    return classifications

# Function to call the OpenAI ChatGPT API with dynamic prompts
def get_chatgpt_response(prompt):
    api_key = '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'  # Add your API key here
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )
    return response.choices[0].message['content']

# Function to generate questions and answers based on the comparison
def generate_questions_and_answers(resume, job_offer):
    questions_and_answers = []

    # Check for skills
    resume_skills = resume.get("OTHER", {}).get("skills", [])
    job_skills = job_offer.get("Skills Needed", "").split(", ")
    missing_skills = [skill for skill in job_skills if skill and skill not in resume_skills]
    if missing_skills:
        questions_and_answers.append({
            "question": "Does the candidate have the skills needed?",
            "answer": f"The candidate is missing the following skills: {', '.join(missing_skills)}."
        })
    else:
        questions_and_answers.append({
            "question": "Does the candidate have the skills needed?",
            "answer": "The candidate has all the required skills."
        })

    # Check for education
    required_education = job_offer.get("Education Required", "")
    resume_education = [f"{edu.get('school_name', '')} ({edu.get('graduation_title', '')})" for edu in resume.get("EDUCATION", [])]
    if required_education in resume_education:
        questions_and_answers.append({
            "question": "Does the candidate meet the educational requirements?",
            "answer": "The candidate meets the educational requirements."
        })
    else:
        questions_and_answers.append({
            "question": "Does the candidate meet the educational requirements?",
            "answer": f"The candidate does not meet the educational requirements. Required: {required_education}."
        })

    # Check for years of experience
    experience_years = 0
    for exp in resume.get("EXPERIENCE", []):
        start_year = int(exp["Start date"].split()[-1])
        end_year = 2024 if exp.get("Is_current", False) else int(exp["End date"].split()[-1])
        experience_years += end_year - start_year
    
    required_experience = job_offer.get("Years of Experience", "0 years")
    match = re.search(r'\d+', required_experience)
    required_experience_years = int(match.group()) if match else 0
    if experience_years >= required_experience_years:
        questions_and_answers.append({
            "question": "Does the candidate have the required years of experience?",
            "answer": f"The candidate has {experience_years} years of experience, which meets the requirement."
        })
    else:
        questions_and_answers.append({
            "question": "Does the candidate have the required years of experience?",
            "answer": f"The candidate has {experience_years} years of experience, which does not meet the {required_experience_years} years required."
        })

    return questions_and_answers

# Function to get the resume by candidate name
def get_resume_by_name(resumes, candidate_name):
    for resume in resumes:
        if resume.get("CONTACT DETAILS", {}).get("FullName", "").lower() == candidate_name.lower():
            return resume
    return None

# Main Streamlit application
def main():
    st.set_page_config(layout="wide")  # Optional: set the layout to 'wide'

    st.markdown(
        """
        <style>
        .main {
            background-color: #ffffff;
        }
        .stApp {
            padding: 10px;
        }
        .section {
            padding: 20px;
            background-color: #ffffff;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .section-title {
            text-align: center;
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 20px;
            color: #333333;
        }
        .upload-section {
            max-width: 800px;
            margin: auto;
        }
        .graph-frame {
            border-radius: 15px;
        }
        .stButton>button {
            width: 100px;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title of the dashboard
    st.markdown('<div class="section-title">Resume Scoring System</div>', unsafe_allow_html=True)

    # Upload section for job offer and resumes
    st.markdown('<div class="section-title">Upload Job Offer and Resumes</div>', unsafe_allow_html=True)
    with st.expander("Upload Section", expanded=True):
        with st.container():
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            job_offer_json = st.file_uploader("Upload Job Offer JSON", type=['json'], key="job_offer")
            if job_offer_json:
                job_offer_data = json.load(job_offer_json)
                job_offer_text = format_job_offer(job_offer_data)
                job_location_tokens = clean_location(job_offer_data['Location'])

                # Extract the company name and job title from the job offer
                company_name = job_offer_data.get('Company Name', '[Company Name]')
                job_title = job_offer_data.get('Job Post Title', '[Job Title]')

                resume_files = st.file_uploader("Upload Resume JSONs", type=['json'], key="resumes", accept_multiple_files=True)
                if resume_files:
                    resumes = []
                    for resume_file in resume_files:
                        resume_data = json.load(resume_file)
                        resumes.append(resume_data)
                    
                    classifications = classify_resumes(resumes)
                    df_classification = pd.DataFrame({"Resume": range(len(resumes)), "Category": classifications})

                    resume_details = []
                    location_counts = {'In Your Location': 0, 'Not In Your Location': 0}
                    certificates_counts = {'with': 0, 'without': 0}
                    total_candidates = len(resume_files)

                    for i, resume_data in enumerate(resumes):
                        candidate_location_tokens = clean_location(resume_data['CONTACT DETAILS']['Address'])
                        location_category = categorize_location(candidate_location_tokens, job_location_tokens)
                        location_counts[location_category] += 1

                        resume_text = format_resume(resume_data)
                        has_certificates = 'with' if resume_data['OTHER']['certificates'].strip() else 'without'
                        certificates_counts[has_certificates] += 1

                        score = calculate_similarity(resume_text, job_offer_text)
                        resume_details.append({
                            "Name": resume_data['CONTACT DETAILS']['FullName'],
                            "Email": resume_data['CONTACT DETAILS']['Email'],
                            "Score": round(score * 100, 1),
                            "Experience": ' | '.join([f"{exp['job_title']}: {'; '.join(exp['tasks'])}" for exp in resume_data.get('EXPERIENCE', [])]),
                            "Education": ' | '.join([f"{edu['school_name']} - {edu['graduation_title']}" for edu in resume_data.get('EDUCATION', [])]),
                            "Skills": ', '.join(resume_data['OTHER'].get('skills', [])),
                            "Certificates": ', '.join(resume_data['OTHER'].get('certificates', [])),
                            "Location Category": location_category,
                            "Has Certificates": has_certificates,
                            "Classification": classifications[i]
                        })

                    df_resumes = pd.DataFrame(resume_details)

                    # Metric card for total candidates
                    st.metric("Total Number of Candidates Applied", total_candidates)

                    # Extract scores for categorization
                    scores = [res['Score'] for res in resume_details]
                    categories = categorize_scores(scores)

                    # Layout the dashboard
                    st.markdown('<div class="section-title">Dashboard</div>', unsafe_allow_html=True)
                    with st.container():
                        col1, col2 = st.columns(2)

                        # Display the scoring bar chart at the top
                        with col1:
                            st.subheader("Scoring Distribution")
                            score_fig = plot_score_bar_chart(categories)
                            st.plotly_chart(score_fig, use_container_width=True, classes="graph-frame")

                        # Display the pie chart for certificates
                        with col2:
                            st.subheader("Certificates Distribution")
                            certificates_fig = plot_certificates_pie_chart(certificates_counts)
                            st.plotly_chart(certificates_fig, use_container_width=True, classes="graph-frame")
                        
                        # New layout column for location and classification charts
                        col3, col4 = st.columns(2)

                        # Display the donut chart for location
                        with col3:
                            st.subheader("Location Distribution")
                            location_fig = plot_location_donut_chart(location_counts)
                            st.plotly_chart(location_fig, use_container_width=True, classes="graph-frame")

                        # Display the classification graph with proper bar sizes based on category counts
                        with col4:
                            st.subheader("Resume Classification Distribution")
                            df_classification_counts = df_classification['Category'].value_counts().reset_index()
                            df_classification_counts.columns = ['Category', 'count']
                            fig_classification = px.bar(df_classification_counts, x='Category', y='count', title="Resume Classification Distribution", color_discrete_sequence=['#003C43'])
                            st.plotly_chart(fig_classification, use_container_width=True, classes="graph-frame")

                    st.markdown('<div class="section-title">Filter and Table</div>', unsafe_allow_html=True)
                    with st.container():
                        # Add filters for classification and score
                        classifications_filter = st.multiselect("Filter by Classification", options=CATEGORIES)
                        score_min, score_max = st.slider("Filter by Score", min_value=0, max_value=100, value=(0, 100))

                        filtered_df_resumes = df_resumes[
                            (df_resumes['Classification'].isin(classifications_filter) if classifications_filter else True) &
                            (df_resumes['Score'] >= score_min) & (df_resumes['Score'] <= score_max)
                        ]

                        # Display the filtered dataframe with email buttons
                        st.write('<style>div.stButton > button {width: 100px;}</style>', unsafe_allow_html=True)
                        for i, row in filtered_df_resumes.iterrows():
                            col1, col2 = st.columns([8, 1])
                            with col1:
                                st.write(row.to_frame().T)
                            with col2:
                                mailto_url = f"https://mail.google.com/mail/?view=cm&fs=1&to={urllib.parse.quote(row['Email'])}&su=Invitation to Interview&body=Dear {row['Name']},%0D%0A%0D%0AWe are pleased to inform you that you have been shortlisted for an interview for the {job_title} position at {company_name}. We would like to invite you to an interview at our office on [Date] at [Time].%0D%0A%0D%0APlease let us know if the proposed time works for you or if you need to reschedule. We look forward to meeting you.%0D%0A%0D%0ABest regards,%0D%0A[Your Name]%0D%0A[Your Position]%0D%0A{company_name}"
                                st.markdown(f'<a href="{mailto_url}" target="_blank"><button>Send Email</button></a>', unsafe_allow_html=True)

    # Left sidebar for candidate name input and chatbot
    with st.sidebar:
        st.markdown('<div class="section-title">Chatbot</div>', unsafe_allow_html=True)
        candidate_name = st.text_input("Enter Candidate Full Name")
        if candidate_name and resume_files:
            resume = get_resume_by_name(resumes, candidate_name)
            if resume:
                st.success(f"Resume for {candidate_name} found!")
                questions_and_answers = generate_questions_and_answers(resume, job_offer_data)
            
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                chat_placeholder = st.empty()

                with chat_placeholder.container():
                    for i, chat in enumerate(st.session_state.chat_history):
                        if chat['role'] == 'user':
                            st.markdown(f"**You:** {chat['content']}")
                        else:
                            st.markdown(f"**Chatbot:** {chat['content']}")

                selected_question = st.selectbox("Select a question to ask the chatbot:", [qa["question"] for qa in questions_and_answers])
                if st.button("Ask"):
                    answer = next(qa["answer"] for qa in questions_and_answers if qa["question"] == selected_question)
                    st.session_state.chat_history.append({"role": "user", "content": selected_question})
                    st.session_state.chat_history.append({"role": "assistant", "content": "..."})

                    chat_placeholder.empty()
                    with chat_placeholder.container():
                        for i, chat in enumerate(st.session_state.chat_history):
                            if chat['role'] == 'user':
                                st.markdown(f"**You:** {chat['content']}")
                            else:
                                st.markdown(f"**Chatbot:** {chat['content']}")

                    time.sleep(1)  # Simulate thinking time
                    st.session_state.chat_history[-1] = {"role": "assistant", "content": answer}
                    chat_placeholder.empty()
                    with chat_placeholder.container():
                        for i, chat in enumerate(st.session_state.chat_history):
                            if chat['role'] == 'user':
                                st.markdown(f"**You:** {chat['content']}")
                            else:
                                st.markdown(f"**Chatbot:** {chat['content']}")
            else:
                st.error(f"Resume for {candidate_name} not found.")

if __name__ == "__main__":
    main()
