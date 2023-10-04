from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-ar"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(english_text):
   
    src_text = [">>ara<< " + english_text]
    
    translated = model.generate(**tokenizer(src_text, return_tensors="pt"))
    
    return tokenizer.decode(translated[0], skip_special_tokens=True)
