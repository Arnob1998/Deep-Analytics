
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
# app.config["SECRECT_KEY"] = 'lol'
app.secret_key = "lol"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['UPLOAD_FOLDER'] = 'static/file'
db = SQLAlchemy(app)

from application import routes

import os
os.environ['DIALOGFLOW_PROJECT_ID'] = 'deep-analytics-va-wvey'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "credentials\deep-analytics-va-wvey-4a46b26aa0f0.json"