from transformers import pipeline


def translate_with_marefa(english_text: str) -> str:
    translator = pipeline("translation_en_to_ar", model="marefa-nlp/marefa-mt-en-ar")
    
    # Translate the input
    translated = translator(english_text)
    
    # Extract the translated text and return
    arabic_translation = translated[0]['translation_text']
    return arabic_translation
