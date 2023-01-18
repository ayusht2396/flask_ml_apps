#Import Flask library and required callbacks
from flask import Flask,render_template,request
import pickle
import spacy
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader


#Launch App and Load the Model
app = Flask(__name__)
result = ""
jd = ""
cv_path = "./static/cv.pdf"

#Index or Landing Page
@app.route('/')
def home():

    return render_template('index.html',**locals())

def read_cv(path):
    reader = PdfReader(path) # .pdf
    number_of_pages = len(reader.pages)
    page = reader.pages[0]
    text = page.extract_text().replace("\n"," ")
    return text

def process(text):
    nlp = spacy.load("en_core_web_lg")

    doc = nlp(text.lower())
    words = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            words.append(token.lemma_)
    return(" ".join(words))

def prepare_corpus(jd,cv_path):
    cv = read_cv(cv_path)
    cv_txt = process(cv)
    
    jd_txt = process(jd)
    return [cv_txt,jd_txt]

#Function called on form submission
@app.route('/score',methods=['POST','GET'])
def predict():

    jd = request.form['job_description']
   
    pdf_file = request.files['resume']

    # Check if the file is present
    if pdf_file.filename == '':
        result = 'No file selected'

    # Check if the file is a PDF
    if pdf_file and pdf_file.filename.endswith('.pdf'):
        # Save the file to the server
        pdf_file.save(cv_path)
        result =  'File uploaded successfully'
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        cvz = TfidfVectorizer()
        corpus = prepare_corpus(jd,cv_path)
            
        cvz.fit(corpus)
        vc = cvz.transform(corpus).toarray()
        v1=vc[0]
        v2=vc[1]

        from sklearn.metrics.pairwise import cosine_similarity
        result = f"Your Score is: {round(100*cosine_similarity(v1.reshape(1,-1),v2.reshape(1,-1))[0][0],2)}%"
    else:
        result =  'Invalid file type'

    
    
    return render_template('predict.html',**locals())

if __name__=="__main__":
    app.run(debug=True)
