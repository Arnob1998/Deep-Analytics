from asyncio.windows_utils import pipe
from flask import render_template,jsonify,request,url_for,session,redirect,send_from_directory
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import json
from application.Database import Aespect, Data, Review, SimilarityCluster, AespectIndividual,SBAEModelFlow,SBAEModelSelection
from application import app
import pandas as pd
from application import db

from application.Pipeline import PipeLine, RoutePipeLine, TextTransformationModel

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

# TODO if no database found create

@app.before_first_request
def before_first_request():
    db.create_all()
    dflow = SBAEModelFlow(name="Default",flow={0:"BART-Summerizer",1:"Pegasus-Parapharser",2:"No Change"})
    db.session.add(dflow)
    db.session.commit()

@app.route("/file-upload", methods = ['GET',"POST"]) # cwd C:\Users\Home\Desktop\app
def file_upload():
    files = os.listdir(os.getcwd() + "\\application\\" + str(app.config['UPLOAD_FOLDER']).replace("/","\\"))
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        return "File has been uploaded."
    return render_template("file_upload.html", form = form, file = files)

@app.route("/file-view")
def file_view():
    files = os.listdir(os.getcwd() + "\\application\\" + str(app.config['UPLOAD_FOLDER']).replace("/","\\"))
    pipeline = PipeLine()
    if files:
        sorted_data = []
        for file in files:
            review_data = pipeline.load_site_data(os.getcwd() + "\\application\\" + str(app.config['UPLOAD_FOLDER'] + "\\" + file))
            review_data = review_data.head(3).append(review_data.tail(3))
            columns = list(review_data.columns.values)
            temp = {}
            for col in columns:
                temp[col] = []
            for column in columns:
                temp[column].append(review_data[column].to_list())
            sorted_data.append(temp)
        return render_template("file_view.html", tables = sorted_data, filename = files) # asuming all table has same header
    return "No Files avaiable."

@app.route("/", methods = ['GET',"POST"])
def file_menu():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        redirect(request.url)

    sorted_data = []
    files = os.listdir(os.getcwd() + "\\application\\" + str(app.config['UPLOAD_FOLDER']).replace("/","\\"))
    pipeline = PipeLine()
    if files:   
        for file in files:
            review_data = pipeline.load_site_data(os.getcwd() + "\\application\\" + str(app.config['UPLOAD_FOLDER'] + "\\" + file))
            review_data = review_data.head(3).append(review_data.tail(3))
            columns = list(review_data.columns.values)
            temp = {}
            for col in columns:
                temp[col] = []
            for column in columns:
                temp[column].append(review_data[column].to_list())
            sorted_data.append(temp)
    return render_template("file_menu.html", form = form, tables = sorted_data, filename = files)

@app.route("/file-selected/<file_name>", methods=['GET', 'POST'])
def file_selected(file_name):
    dir_data = os.getcwd() + "\\application\\" + str(app.config['UPLOAD_FOLDER'] + "\\" + file_name)
    
    if not Data.query.filter_by(name=file_name).first():
        print(f"{file_name} not Found in Data")
        print("Creating a record")
        db_data = Data(name=file_name)
        db.session.add(db_data)
        db.session.commit()

    data_id = Data.query.filter_by(name=file_name).first().id
    exists = Review.query.filter_by(data_id=data_id).first() is not None # check is last non and compare with index if last
    if not exists:
        pipe = PipeLine()
        site_processed = pipe.load_site_data(dir_data)
        title = site_processed["Title"]
        content = site_processed["Content"]
        rating = site_processed["Ratings"]
        for i in range(len(site_processed)): # CLEAN HERE

            if content[i][0] == '{':
                content[i] = content[i].split("modal window.")[1].lstrip().rstrip()
                
            rev = Review(rating=rating[i], title=title[i], review=content[i], data_id=data_id)
            db.session.add(rev)
            db.session.commit()

    sbae_exists = SBAEModelSelection.query.filter_by(data_id=data_id).first() is not None
    model_flow = None
    if not sbae_exists:
        model_flow = SBAEModelFlow.query.filter_by(name="Default").first().flow
    else:
        flow_name = SBAEModelSelection.query.filter_by(data_id=data_id).first().current_flow
        model_flow = SBAEModelFlow.query.filter_by(name=flow_name).first().flow

    return render_template("file-selected.html", file = file_name, sbae_exists = model_flow)

@app.route("/all-aespect/<file_name>", methods=['GET'])
def all_aespect(file_name):
    ctrl_all_aespect = RoutePipeLine(file_name)
    aespect_out = ctrl_all_aespect.route_all_aespect()

    header = ["#","Review","Aespects","Score"]
    return render_template("aespect-all.html", file = file_name,col_head=header,data=aespect_out)

@app.route("/individual-aespect/<file_name>", methods=['GET'])
def individual_aespect(file_name):
    ctrl_individual_aespect = RoutePipeLine(file_name)
    indi_aespect_out = ctrl_individual_aespect.route_individual_aespect()

    header = ["#","Aespects","Postive-Negative Ratio","Influence Score"]
    return render_template("aespect-individual.html",col_head=header,data=indi_aespect_out,file=file_name) # use data from DB

@app.route("/models/<file_name>", methods=['GET'])
def models(file_name):

    info = {"SBAE": {"header":"Sentiment-Based Aspect Extractor",
                     "description":"Discover core aspects of each review related to the emotional tone that underlies it, along with a scoring system that represents that tone.",
                     "l1" : 0,
                     "l2" : {"percent": None},
                     "l3" : None,
                     "lt" : {"k" : None, "v" : None}
                    },
            "A2S": {"header":"Aspect to Score",
                    "description":"Get a thorough analysis of the most important aspects",
                    "note":"Process Sentiment-Based Aspect Extractor first",
                    "ratio": [None,None,None],
                    "u_aespect" : None,
                    "stat" : None
                   },
            "SMC": {"header":"Similarity Cluster",
                    "description":"Find out how many divided group of opinions does people have about the product and get the most accurate reflection of those opinions.",
                     "l1" : 0,
                     "l2" : None,
                     "l3" : None,
                     "l4" : None
                   },
            "TPM":{"header":"Similarity Cluster",
                   "description":"Extract topics discussed in the reviews",
                   "l1": "3",
                   "l2": "Great",
                   "l3":24
                   }
            }

    ctrl_models = RoutePipeLine(file_name)
    info = ctrl_models.route_models(info)

    return render_template("models.html" , file = file_name, info=info)

@app.route("/into-set-SBAE/<file_name>", methods=['GET'])
def into_set_SBAE(file_name):
    from application.Pipeline import TextTransformationModel
    tt_obj = TextTransformationModel()
    models = tt_obj.model_list()

    model_info = {}
    for i in range(len(models)):
        model_info[i] = {"name":models[i].model_name, "base-name": models[i].base_name, "transformation":models[i].transformation, "description": models[i].description}

    infoTB_head = ["#","Name","Base-Name","Description"];

    data_id = Data.query.filter_by(name=file_name).first().id
    sbae_exists = SBAEModelSelection.query.filter_by(data_id=data_id).first() is not None
    model_flow = None
    if not sbae_exists:
        model_flow = SBAEModelFlow.query.filter_by(name="Default").first().flow
    else:
        flow_name = SBAEModelSelection.query.filter_by(data_id=data_id).first().current_flow
        model_flow = SBAEModelFlow.query.filter_by(name=flow_name).first().flow    
    
    return render_template("info-settings-SBAE.html", file = file_name, models = models , infoTB_head = infoTB_head, model_info= model_info, sbae_exists = model_flow)

@app.route("/analytics/<file_name>", methods=['GET'])
def analytics(file_name):
    return render_template("analytics.html" , file = file_name)

@app.route("/similar-cluster/<file_name>", methods=['GET'])
def similarity_cluster(file_name):
    ctrl_similarity_cluster = RoutePipeLine(file_name)
    cluster = ctrl_similarity_cluster.route_similarity_cluster()

    header = ["Review","Similarity Found"]
    return render_template("similar_cluster.html",file = file_name,col_head=header,data=cluster)

@app.route("/topicvec/<file_name>", methods=['GET'])
def topic_model(file_name):
    data_id = Data.query.filter_by(name=file_name).first().id
    pipe = PipeLine()
    t_words,t_dict,t_cloud = pipe.topic_modeling(data_id)

    return render_template("topic-model.html",file = file_name,topics=t_words,topic_dict = t_dict, word_cloud = t_cloud)


# @app.route("/modal-index")
# def modal_index():
#     return render_template("_modal-test.html")


# @app.route("/modal-modal")
# def modal_content():
#     return render_template("_modal-test-main.html")