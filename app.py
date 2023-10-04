from flask import Flask, render_template, request

from trans_test import translate_text

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    translation = ""
    if request.method == "POST":
        english_text = request.form["english_text"]
        translation = translate_text(english_text)  
    return render_template("template.html", translation=translation)

if __name__ == "__main__":
    app.run(debug=True)
