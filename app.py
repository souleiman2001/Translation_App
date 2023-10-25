import os
import re

import nltk
import requests
from flask import Flask, render_template, request
from googletrans import Translator
from nltk.translate.bleu_score import sentence_bleu

from finetuned_marian_120k import translate_with_finetuned
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
translator = Translator()

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




@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/translate", methods=["GET", "POST"])
def translate():
    translation = ""
    if request.method == "POST":
        english_text = request.form.get("english_text", "")
        # Assume your translate_text function takes an English text and returns its Arabic translation
        translation = translate_text(english_text)
    return render_template("template.html", translation=translation)

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
    
    return render_template("compare.html", score=score, reference_text=reference_text, translated_text=translated_text)




@app.route('/translate_gpt4', methods=['GET', 'POST'])
def gpt4_translation_interface():
    
    if request.method == 'POST':
        english_text = request.form.get('english_text')
        translation = translator.translate(english_text, dest='ar')
        return render_template('translate_gpt4.html', gpt4_translated_text=translation.text)
    return render_template('translate_gpt4.html', gpt4_translated_text="")

@app.route("/translate_turjuman", methods=["GET", "POST"])
def translate_turjuman():
    translation = ""
    if request.method == "POST":
        english_text = request.form.get("english_text", "")
        translation = translate_text_turjuman(english_text)
    return render_template("translate_turjuman.html", translation=translation)


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




if __name__ == "__main__":
    app.run(debug=True)
