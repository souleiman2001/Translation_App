# finetuned_marian_120k.py

from transformers import pipeline


def translate_with_finetuned(english_text: str) -> str:
    translator = pipeline("translation", model="Shularp/model-translate-en-to-ar-from-120k-dataset-ar-en-th230111447")
    
    # Translate the input
    translated = translator(english_text)
    
    # Extract the translated text and return
    arabic_translation = translated[0]['translation_text']
    return arabic_translation
