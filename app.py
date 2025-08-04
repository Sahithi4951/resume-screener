# SAHITHI BASHETTY

from flask import Flask, render_template, request
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# Here we are creating a flask app
app = Flask(__name__)

# here we are removing the punctuation,numbers


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# This is a function of extracting text from pdf file
def extract_text_from_pdf(pdf_file):
    text = ''
    try:
        reader = PyPDF2.PdfReader(pdf_file.stream)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        print("Error reading PDF:", e)
    return text

# we are creating a route and methods post -> we are sending the information to the backend server


@app.route('/')
def greet():
    return render_template('greet.html')


@app.route('/screen', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    suggestions = []
    score = None
    filter_suggestions = []
    tech_keywords = set(['python', 'java', 'c', 'c++', 'c#', 'javascript', 'typescript',
                      'go', 'kotlin', 'ruby', 'swift', 'r', 'php', 'rust', 'scala',
                      'react', 'angular', 'vue', 'django', 'flask', 'spring', 'springboot',
                      'express', 'nextjs', 'nestjs', 'tailwind', 'bootstrap', 'junit',
                      'pytest', 'redux', 'jquery', 'tensorflow', 'keras', 'pytorch',
                      'scikit-learn', 'matplotlib', 'seaborn', 'pandas', 'numpy', 'machine learning', 'deep learning', 'ai', 'ml', 'nlp',
                      'computer vision', 'regression', 'classification', 'clustering',
                      'lstm', 'cnn', 'rnn', 'recommendation system', 'data visualization', 'mysql', 'postgresql', 'mongodb', 'oracle', 'firebase',
                      'sqlite', 'redis', 'cassandra', 'elasticsearch', 'sql', 'nosql', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
                      'terraform', 'ansible', 'ci/cd', 'linux', 'bash', 'nginx', 'unittest', 'junit', 'pytest', 'mocha', 'chai', 'selenium',
                      'cypress', 'jest', 'testing library', 'integration testing',
                      'unit testing', 'test automation', 'rest', 'restful', 'graphql', 'api', 'microservices',
                      'oop', 'mvc', 'authentication', 'authorization',
                      'websockets', 'multithreading', 'concurrency', 'data structures',
                      'algorithms', 'design patterns', 'html', 'css', 'javascript', 'typescript', 'sass', 'less',
                      'responsive design', 'accessibility', 'ui/ux', 'figma', 'adobe xd', 'git', 'github', 'gitlab', 'bitbucket', 'jira', 'postman',
                      'vs code', 'intellij', 'eclipse', 'pycharm', 'android studio'])

    # we are using a request method post to send the info
    if request.method == 'POST':
        uploaded_file = request.files['file']
        job_description = request.form['des']

        # creating a function to extract the text from pdf and calculate the score
        if uploaded_file.filename.endswith('.pdf'):
            resume_text = clean_text(extract_text_from_pdf(uploaded_file))
            job_description = clean_text(job_description)

            # we are here vectorizing the 2 docs
            docs = [resume_text, job_description]
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(docs)

            # we are calculating the similarities and match score for 2 docs
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            score = int(similarity[0][0] * 100)

            # we are coverting them into sets cuz its easier to compare
            resume_words = set(resume_text.split())
            jd_words = set(job_description.split())
            suggestions = [word for word in (
                jd_words - resume_words) if word not in ENGLISH_STOP_WORDS]
            filter_suggestions = [
                word for word in suggestions if word in tech_keywords]

        else:
            score = 0
            filter_suggestions = []

    return render_template('result.html', score=score, filter_suggestions=filter_suggestions)


if __name__ == "__main__":
    app.run(debug=True)
