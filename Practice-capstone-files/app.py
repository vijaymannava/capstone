import flask


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)

#model = pickle.load(open("models/newscl.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")


if __name__ == "__main__":
    flask_app.run(debug=True)