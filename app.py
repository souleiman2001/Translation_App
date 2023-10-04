from flask import Flask, redirect, render_template, request, url_for

from trans_test import \
    translate_text  # Ensure this module is in the same directory or adjust the import

app = Flask(__name__)

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/translate", methods=["GET", "POST"])
def translate():
    translation = ""
    if request.method == "POST":
        english_text = request.form["english_text"]
        translation = translate_text(english_text)
    return render_template("template.html", translation=translation)

@app.route("/compare", methods=["GET", "POST"])
def compare():
    # Your comparison logic will be placed here if the comparison is done server-side.
    # If the comparison is done client-side with JavaScript, you might not need POST method here.
    return render_template("compare.html")

if __name__ == "__main__":
    app.run(debug=True)
