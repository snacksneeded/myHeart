from flask import Flask, render_template, request
from main import load_data, classify, predict_heart_disease

app = Flask(__name__)

# Load data and train model when app starts
features_data, target_data = load_data()
model = classify(features_data, target_data)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_data = [
            int(request.form.get("age")),
            int(request.form.get("gender")),
            int(request.form.get("chestpain", 0)),
            int(request.form.get("restingblood")),
            int(request.form.get("chol")),
            int(request.form.get("fastingblood")),
            int(request.form.get("restingECG")),
            int(request.form.get("maximumheartrate")),
            int(request.form.get("angina")),
            float(request.form.get("STdepression")),
            int(request.form.get("slope")),
            int(request.form.get("fluoroscopy")),
            int(request.form.get("thallium")),
        ]

        probability = predict_heart_disease(model, user_data)

        print(user_data)

        return render_template("index.html", probability=probability * 100)

    return render_template("index.html", probability=None)


if __name__ == "__main__":
    app.run(debug=True)

