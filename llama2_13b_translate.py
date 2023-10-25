from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
model = AutoModelForSeq2SeqLM.from_pretrained("SnypzZz/Llama2-13b-Language-translate")

def translate_with_llama2_13b(english_text: str) -> str:
    # Prepare the model inputs
    inputs = tokenizer(english_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    # Set the forced_bos_token_id to the Arabic language code to ensure translation to Arabic
    arabic_lang_id = tokenizer.lang_code_to_id["ar_AR"]
    
    # Generate the translation
    translated_tokens = model.generate(**inputs, forced_bos_token_id=arabic_lang_id)
    
    # Decode the tokens to get the translated text
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
    return translation

text = input("\nEnter English text for translation: ")
translation = translate_with_llama2_13b(text)
print("\nTranslated text:", translation)
