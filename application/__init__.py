
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
# app.config["SECRECT_KEY"] = 'lol'
app.secret_key = "lol"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['UPLOAD_FOLDER'] = 'static/file'
db = SQLAlchemy(app)

from application import routes