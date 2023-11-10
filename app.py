import os
import re

import nltk
import requests
from flask import Flask, redirect, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy
from googletrans import Translator
from nltk.translate.bleu_score import sentence_bleu
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from finetuned_marian_120k import translate_with_finetuned
#from llama2_13b_translate import llama2_translation
from marefa_trans import translate_with_marefa
from trans_test import \
    translate_text  # Ensure this module is in the same directory or adjust the import
from turjuman_trans import translate_text_turjuman

# Ensure the NLTK tokenizer model is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:slimanemultiverse@localhost/postgres'


translator = Translator()
db = SQLAlchemy(app)

class Score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    bleu_score = db.Column(db.Float, nullable=True)
    cosine_similarity = db.Column(db.Float, nullable=True)
    lsa_score = db.Column(db.Float, nullable=True)
    
    
    


@app.route('/addscore', methods=['POST'])
def add_score():
    if request.method == 'POST':
        bleu_score = request.form.get('bleu_score')
        cosine_similarity = request.form.get('cosine_similarity')
        lsa_score = request.form.get('lsa_score')

        new_score = Score(bleu_score=bleu_score, cosine_similarity=cosine_similarity, lsa_score=lsa_score)
        db.session.add(new_score)
        db.session.commit()

        return "Score added successfully!"
    return "Failed to add score."







def compute_bleu(reference, candidate):
    # Tokenizing words in the reference and candidate sentences
    reference = [re.findall(r'\w+', reference)]
    candidate = re.findall(r'\w+', candidate)

    # Defining weights for BLEU (you can adjust this as per requirement)
    weights = (0.33, 0.33, 0.33, 0)  

    try:  # Wrapping the score computation to handle potential exceptions
        score = sentence_bleu(reference, candidate, weights=weights)
    except Exception as e:
        print(f"Error: {str(e)}")
        score = 0  # Returning a score of 0 if BLEU computation fails

    return score


def compute_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]




def compare_texts_using_lsa(wiki_arabic, translated_text, variance_threshold=0.95):
    # If both texts are exactly the same, return similarity of 1
    if wiki_arabic == translated_text:
        return 1.0
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([wiki_arabic, translated_text])

    # Determine the number of topics to retain a certain variance
    total_variance = tfidf_matrix.shape[1]
    n_components = total_variance

    while n_components > 0:
        lsa_temp = TruncatedSVD(n_components=n_components)
        lsa_temp.fit(tfidf_matrix)
        if sum(lsa_temp.explained_variance_ratio_) >= variance_threshold:
            break
        n_components -= 1

    # Apply LSA (dimensionality reduction)
    lsa_model = TruncatedSVD(n_components=n_components)
    lsa_topic_matrix = lsa_model.fit_transform(tfidf_matrix)

    # Compute cosine similarity on LSA-transformed vectors
    lsa_cosine_sim = cosine_similarity(lsa_topic_matrix)
    return lsa_cosine_sim[0][1]


@app.route("/")
def landing():
    return render_template("landing.html")


""""
@app.route("/translate", methods=["GET", "POST"])
def translate():
    translation = ""
    if request.method == "POST":
        english_text = request.form.get("english_text", "")
        # Assume your translate_text function takes an English text and returns its Arabic translation
        translation = translate_text(english_text)
    return render_template("template.html", translation=translation)
"""


@app.route("/translate", methods=["GET", "POST"])
def translate():
    translation = ""
    tool_name = "MarianMT"  # Define the tool name here for clarity and consistency
    if request.method == "POST":
        english_text = request.form.get("english_text", "")
        translation = translate_text(english_text)
        # Instead of rendering the template, redirect to the prepare_compare route
        return redirect(url_for('prepare_compare', translation=translation, tool_name=tool_name))
    return render_template("template.html", translation=translation)

# New route to prepare for comparison
@app.route("/prepare_compare")
def prepare_compare():
    # Retrieve the translation and tool name from the query parameters
    translation = request.args.get('translation', '')
    tool_name = request.args.get('tool_name', '')
    # Render the compare.html template with the translated text and tool name
    return render_template("compare.html", translated_text=translation, tool_name=tool_name)




@app.route("/compare", methods=["GET", "POST"])
def compare():
    score = None
    reference_text = ""
    translated_text = ""
    
    if request.method == "POST":
        reference_text = request.form.get("main_text", "")
        translated_text = request.form.get("translated_text", "")
        comparison_type = request.form.get("comparison_type", "")
        
        if comparison_type == "bleu" and reference_text and translated_text:
            score = compute_bleu(reference_text, translated_text)
        elif comparison_type == "cosine" and reference_text and translated_text:
            score = compute_cosine_similarity(reference_text, translated_text)
            
        elif comparison_type == 'lsa':
            score = compare_texts_using_lsa(reference_text, translated_text)
    
    return render_template("compare.html", score=score, reference_text=reference_text, translated_text=translated_text)


""""

@app.route('/translate_gpt4', methods=['GET', 'POST'])
def gpt4_translation_interface():
    
    if request.method == 'POST':
        english_text = request.form.get('english_text')
        translation = translator.translate(english_text, dest='ar')
        return render_template('translate_gpt4.html', gpt4_translated_text=translation.text)
    return render_template('translate_gpt4.html', gpt4_translated_text="")
"""

@app.route('/translate_gpt4', methods=['GET', 'POST'])
def gpt4_translation_interface():
    tool_name = "Google Translate"  # Define the tool name as Google Translate
    if request.method == 'POST':
        english_text = request.form.get('english_text')
        translation = translator.translate(english_text, dest='ar').text
        # Redirect to the prepare_compare route with translation and tool_name
        return redirect(url_for('prepare_compare', translation=translation, tool_name=tool_name))
    return render_template('translate_gpt4.html', gpt4_translated_text="")

""""
@app.route("/translate_turjuman", methods=["GET", "POST"])
def translate_turjuman():
    translation = ""
    if request.method == "POST":
        english_text = request.form.get("english_text", "")
        translation = translate_text_turjuman(english_text)
    return render_template("translate_turjuman.html", translation=translation)
"""

@app.route("/translate_turjuman", methods=["GET", "POST"])
def translate_turjuman():
    translation = ""
    tool_name = "Turjuman"  # Define the tool name for Turjuman
    if request.method == "POST":
        english_text = request.form.get("english_text", "")
        translation = translate_text_turjuman(english_text)
        # Redirect to the prepare_compare route with translation and tool_name
        return redirect(url_for('prepare_compare', translation=translation, tool_name=tool_name))
    return render_template("translate_turjuman.html", translation=translation)



@app.route('/translate_marefa', methods=['GET', 'POST'])
def translate_marefa():
    translation = ""
    tool_name = "Marefa"  # Define the tool name here
    if request.method == 'POST':
        english_text = request.form['english_text']
        translation = translate_with_marefa(english_text)
        # Redirect to the prepare_compare route with translation and tool_name
        return redirect(url_for('prepare_compare', translation=translation, tool_name=tool_name))
    return render_template('marefa.html', translation=translation)

"""
@app.route('/translate_marefa', methods=['GET', 'POST'])
def translate_marefa():
    translation = ""
    if request.method == 'POST':
        english_text = request.form['english_text']
        translation = translate_with_marefa(english_text)
    return render_template('marefa.html', translation=translation)


@app.route("/translate_finetuned", methods=["GET", "POST"])
def translate_finetuned():
    translation = ""
    if request.method == "POST":
        english_text = request.form.get("english_text", "")
        translation = translate_with_finetuned(english_text)
    return render_template("fine_tuned_translation.html", translation=translation)
"""

@app.route("/translate_finetuned", methods=["GET", "POST"])
def translate_finetuned():
    translation = ""
    tool_name = "Fine-tuned MarianMT"  # Define the tool name here
    if request.method == "POST":
        english_text = request.form.get("english_text", "")
        translation = translate_with_finetuned(english_text)
        # Redirect to the prepare_compare route with translation and tool_name
        return redirect(url_for('prepare_compare', translation=translation, tool_name=tool_name))
    return render_template("fine_tuned_translation.html", translation=translation)


#@app.route('/translate_llama2', methods=['GET', 'POST'])
#def translate_llama2():
#    translation = ''
#    if request.method == 'POST':
#        english_text = request.form['english_text']
#        translation = llama2_translation(english_text)
#    return render_template('llama2_template.html', translation=translation)




if __name__ == "__main__":
    app.run(debug=True)
