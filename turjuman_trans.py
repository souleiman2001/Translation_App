
from transformers import pipeline


def translate_text_turjuman(english_text):
    translator = pipeline("text2text-generation", model="UBC-NLP/turjuman")
    translated_output = translator(english_text, max_length=2000)
    arabic_translation = translated_output[0]['generated_text']
    return arabic_translation
