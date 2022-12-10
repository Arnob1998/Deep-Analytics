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

@app.route("/", methods = ['GET',"POST"])
def file_menu():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        redirect(request.url)

    files = []
    data_que = Data.query.all()
    for db_cls in data_que:
        files.append(db_cls.name)

    sorted_data = []
    pipeline = PipeLine()
    if files:   
        for file in files:
            file_id = Data.query.filter_by(name=file).first().id
            review_data = pd.read_sql_query(sql = Review.query.filter_by(data_id=file_id).statement, con=db.session.bind)
            review_data.rename(columns={'rating': 'Ratings', 'title': 'Title', 'review': 'Content'}, inplace=True)
            print(review_data)
            review_data = review_data.head(3).append(review_data.tail(3))
            columns = list(review_data.columns.values)
            temp = {}
            for col in columns:
                temp[col] = []
            for column in columns:
                temp[column].append(review_data[column].to_list())
            sorted_data.append(temp)
    return render_template("file_menu.html", form = form, tables = sorted_data, filename = files)

@app.route("/file-selected/<file_name>", methods=['GET', 'POST']) # review gets uploaded to db here
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
        datetime = site_processed["Date"]
        for i in range(len(site_processed)): # CLEAN HERE

            if content[i][0] == '{':
                content[i] = content[i].split("modal window.")[1].lstrip().rstrip()
                
            rev = Review(review_id = i,rating=rating[i], title=title[i], review=content[i], datetime=datetime[i], data_id=data_id)
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
    # with open('output.json', 'w') as outfile:
    #     json.dump(aespect_out, outfile)
    # return render_template("aespect-all.html", file = file_name,col_head=["#","Review","Aespects","Score"],data=aespect_out)
    def process_overall_score(aespect_out):
        import numpy as np
        scores = {'POSITIVE':[], 'NEUTRAL': [], 'NEGATIVE': []}
        scores_avg = {'POSITIVE':0, 'NEUTRAL': 0, 'NEGATIVE': 0}
        label_freq = {'POSITIVE':0, 'NEUTRAL': 0, 'NEGATIVE': 0}
        cat_freq = {}
        min_n_max_index = {'POSITIVE': {"min" : 0, "max": 0}, 'NEUTRAL': {"min" : 0, "max": 0}, 'NEGATIVE': {"min" : 0, "max": 0}}

        for a in aespect_out:
            for cat in a["category"]:
                if cat in cat_freq:
                    cat_freq[cat] += 1
                else:
                    cat_freq[cat] = 0

            for label in a['score']:
                scores[label].append(a['score'][label])

        for label in scores:
            min_n_max_index[label]["min"] = scores[label].index(min(i for i in scores[label] if i != []))
            min_n_max_index[label]["max"] = scores[label].index(max(i for i in scores[label] if i != []))
            clean_scores = list(filter(None, scores[label]))
            label_freq[label] = len(clean_scores)
            scores_avg[label] = np.mean(clean_scores)

        overall_stat = {"category_freq" : cat_freq, "ratio" : label_freq, "scores_avg": scores_avg, "min_n_max_index" : min_n_max_index}

        return overall_stat

    overall_status = process_overall_score(aespect_out)
    return render_template("aespect-all_v2.html" , file = file_name,data=aespect_out, overall_stat=overall_status)

@app.route("/individual-aespect/<file_name>", methods=['GET'])
def individual_aespect(file_name):
    ctrl_individual_aespect = RoutePipeLine(file_name)
    indi_aespect_out = ctrl_individual_aespect.route_individual_aespect()
    # header = ["#","Aespects","Postive-Negative Ratio","Influence Score"]
    # return render_template("aespect-individual.html",col_head=header,data=indi_aespect_out,file=file_name) # use data from DB
    return render_template("aespect-individual_v2.html",data=indi_aespect_out,file=file_name)

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
                    "stat" : None,
                    "freq_class" : None
                   },
            "SMC": {"header":"Similarity Cluster",
                    "description":"Find out how many divided group of opinions does people have about the product and get the most accurate reflection of those opinions.",
                     "l1" : 0,
                     "l2" : None,
                     "l3" : None,
                     "l4" : None
                   },
            "TPM":{"header":"Topic Extraction",
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
    # data_id = Data.query.filter_by(name=file_name).first().id
    # pipe = PipeLine()
    # t_words,t_dict,t_cloud = pipe.topic_modeling(data_id)
    ctrl_topic = RoutePipeLine(file_name)
    t_words,t_dict,t_cloud = ctrl_topic.route_topic_model()
    
    return render_template("topic-model.html",file = file_name,topics=t_words,topic_dict = t_dict, word_cloud = t_cloud)


@app.route("/v-assistant-ui/<file_name>")
def v_assistant_ui(file_name):

    ctrl_v_assistant = RoutePipeLine(file_name)

    aspect = ctrl_v_assistant.route_v_assistant_ui()

    return render_template("virtual_assistant.html",file = file_name,indi_aes = aspect)

 
@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json(silent=True)
    return "dummy"

def detect_intent_texts(project_id, session_id, text, language_code):
    from google.cloud import dialogflow_v2 as dialogflow
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(project_id, session_id)
    if text:
        text_input = dialogflow.types.TextInput(text=text, language_code=language_code)
        query_input = dialogflow.types.QueryInput(text=text_input)
        response = session_client.detect_intent(session=session, query_input=query_input)
        return response.query_result.fulfillment_text

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.form['message']
    print("Traceback : method : send_message, route: /send_message")
    print("User : " + message)
    project_id = os.getenv('DIALOGFLOW_PROJECT_ID')
    fulfillment_text = detect_intent_texts(project_id, "unique", message, 'en')
    response_text = { "message":  fulfillment_text }
    print("Traceback : method : send_message, route: /send_message")
    print("Bot : " + str(fulfillment_text))
    return jsonify(response_text)              

# @app.route("/modal-index")
# def modal_index():
#     return render_template("_modal-test.html")


# @app.route("/modal-modal")
# def modal_content():
#     return render_template("_modal-test-main.html")
