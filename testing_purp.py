import googletrans
from googletrans import Translator

translator = Translator()

input_text = input('Input Your English Text for Translation: ')
translation = translator.translate(input_text, dest='ar')
print(translation.text)
