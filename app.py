import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(_name_)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def home():
    return render_template("home.html")

@flask_app.route("/predict", methods = ["GET","POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    if prediction==1:
        return render_template("home.html", prediction_text = "there is chance of cancer")
    else:
        return render_template("home.html", prediction_text = "there is no chance of cancer")



if _name_ == "_main_":
    flask_app.run(debug=True)

    