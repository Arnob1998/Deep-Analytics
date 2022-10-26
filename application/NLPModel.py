import pickle # TODO REMOVE MODEL USELESS METHOD
import copy
from scipy.special import softmax
import numpy as np
np.set_printoptions(suppress=True)
import copy

class SentimentAnalysis:

    def __init__(self):
        print("Initilizing models for SentimentAnalysis")
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.vader_model = SentimentIntensityAnalyzer()

    def distilBERT_score(self,text):
        inputs_embeddings = self.tokenizer(text, return_tensors='pt', padding=True,  truncation=True)
        outputs_embeddings = self.model(**inputs_embeddings)
        unfilterd_out = outputs_embeddings[0][0].detach().numpy()
        scores = softmax(unfilterd_out)
        return np.array([scores[0],max(scores)-min(scores),scores[1]])

    def vader_score(self,text):
        sentiment_dict = self.vader_model.polarity_scores(text)
        negative = sentiment_dict['neg']
        neutral = sentiment_dict['neu']
        positive = sentiment_dict['pos']
        return np.array([negative,neutral,positive])

    def default_score(self,text):
        models = [self.distilBERT_score, self.vader_score]
        
        all_score = []
        for model in models:
            score = model(text)
            all_score.append(score)

        return np.mean(all_score,axis=0)


class Paraphraser:

    model_name = "Pegasus-Parapharser"
    base_name = 'tuner007/pegasus_paraphrase'
    transformation = "Paraphraser"
    description = "Sentences are changed by adding new words and phrases without altering their meaning. (Token Length may vary)"

    def __init__(self):
        print("Initilizing models for Paraphraser")
        from transformers import PegasusForConditionalGeneration, PegasusTokenizer
        self.tokenizer = PegasusTokenizer.from_pretrained(self.base_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.base_name)

    def paraphraser(self,text):
        inputs_embeddings = self.tokenizer.encode(text,return_tensors="pt", truncation=True, padding = 'longest')
        outputs_embeddings = self.model.generate(inputs_embeddings,temperature=1.5)
        output = self.tokenizer.decode(outputs_embeddings[0],skip_special_tokens=True)
        return output

    def tokenize(self,text):
        inputs_embeddings = self.tokenizer.encode(text,return_tensors="pt", truncation=True, padding = 'longest')
        return inputs_embeddings

    def rephrase(self,text,max_length=None):
        inputs_embeddings = self.tokenize(text)
        if max_length:
            if len(inputs_embeddings[0]) > max_length:
                outputs_embeddings = self.model.generate(inputs_embeddings,temperature=1.5)
                output = self.tokenizer.decode(outputs_embeddings[0],skip_special_tokens=True)
                return output,True
            else:
                return text,False
        else:
            outputs_embeddings = self.model.generate(inputs_embeddings,temperature=1.5)
            output = self.tokenizer.decode(outputs_embeddings[0],skip_special_tokens=True)
            return output,True

class Summmerizer:

    model_name = "BART-Summerizer"
    base_name = "facebook/bart-large-cnn"
    transformation = "Summmerizer"
    description = "Condenses the paragraph by trimming unnecessary sentences and occasionally adding new sentences."

    def __init__(self):
        print("Initilizing models for Summmerizer")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.base_name)

    def summerize(self,text): # useless
        inputs_embeddings = self.tokenizer.encode(text,return_tensors="pt", truncation=True, padding = True)

        if len(inputs_embeddings[0]) > 142:
            outputs_embeddings = self.model.generate(inputs_embeddings)
            output = self.tokenizer.decode(outputs_embeddings[0],skip_special_tokens=True)
            return output,True
        else:
            return text,False

    def tokenize(self,text):
        inputs_embeddings = self.tokenizer.encode(text,return_tensors="pt", truncation=True, padding = True)
        return inputs_embeddings

    def rephrase(self,text, max_length=None):
        inputs_embeddings = self.tokenize(text)
        if max_length:
            if len(inputs_embeddings[0]) > max_length:
                outputs_embeddings = self.model.generate(inputs_embeddings)
                output = self.tokenizer.decode(outputs_embeddings[0],skip_special_tokens=True)
                return output,True
            else:
                return text,False
        else:   
            outputs_embeddings = self.model.generate(inputs_embeddings)
            output = self.tokenizer.decode(outputs_embeddings[0],skip_special_tokens=True)
            return output,True

    def batch_summerize(self,reviews): # useless
        output = []
        for i in range(len(reviews)):
            output.append(self.summerize(reviews[i]))

        return output

class XSummerizer:

    model_name = "Pegasus-Xtrem_Summerizer"
    base_name = "google/pegasus-xsum"
    transformation = "XSummmerizer"
    description = "Condenses the paragraph to a single sentence. This an extremely destructive techniques which may result in loss of important information."

    def __init__(self):
        print("Initilizing models for XSummerizer")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.base_name)

    def tokenize(self,text):
        inputs_embeddings = self.tokenizer.encode(text,return_tensors="pt", truncation=True, padding = True)
        return inputs_embeddings

    def rephrase(self,text, max_length=None):
        inputs_embeddings = self.tokenize(text)
        if max_length: # not available in this model
            if len(inputs_embeddings[0]) > max_length:
                outputs_embeddings = self.model.generate(inputs_embeddings)
                output = self.tokenizer.decode(outputs_embeddings[0],skip_special_tokens=True)
                return output,True
            else:
                return text,False
        else:   
            outputs_embeddings = self.model.generate(inputs_embeddings)
            output = self.tokenizer.decode(outputs_embeddings[0],skip_special_tokens=True)
            return output,True


    def summerize_xtream(self,text):
        inputs_embeddings = self.tokenizer.encode(text,return_tensors="pt", truncation=True, padding = True)

        print("Attempting XSummrization")
        outputs_embeddings = self.model.generate(inputs_embeddings)
        output = self.tokenizer.decode(outputs_embeddings[0],skip_special_tokens=True)
        return output

        # if len(inputs_embeddings[0]) > 100:
        #     print("Attempting XSummrization")
        #     outputs_embeddings = self.model.generate(inputs_embeddings)
        #     output = self.tokenizer.decode(outputs_embeddings[0],skip_special_tokens=True)
        #     return output
        # else:
        #     print("Skipping XSummrization")
        #     return text

    def batch_summerize_xtream(self,reviews):
        output = []
        print("Batch Xtreme Summerization")
        for i in range(len(reviews)):
            print("Index :" + str(i))
            output.append(self.summerize_xtream(reviews[i]))

        return output    

class SentenceSimiliarity:
    from sklearn.metrics.pairwise import cosine_similarity

    def __init__(self):
        print("Initilizing models for SentenceSimiliarity")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def similarity(self,text1,text2):
        text1_embeddings = self.model.decode(text1)
        text2_embeddings = self.model.decode(text2)
        score = self.cosine_similarity(text1_embeddings.reshape(1, -1),text2_embeddings.reshape(1,-1))[0][0]
        return score

    def batch_embeddings(self,reviews):
        return self.model.encode(reviews)

    def model(self):
        return self.model
        



class AespectMiner:
    
    def __init__(self):
        print("Initilizing models for AespectMiner")
        # self.model = pickle.load(open(r'models\aespect_miner.pkl', 'rb'))
        # self.saved_result = None
        from pyabsa import ABSADatasetList, available_checkpoints
        from pyabsa import ATEPCCheckpointManager
        self.model = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english',auto_device=True)
        
    def predict(self,text):    
        inference_source = ["""{}""".format(text)]
        atepc_result = self.model.extract_aspect(inference_source=inference_source,  #
                          save_result=False,
                          print_result=False,  # print the result
                          pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                          )

        return atepc_result

    def process_result(self,result):
        res_aespect = result[0]["aspect"]
        res_sentiment = result[0]["sentiment"]
        res_confidence = result[0]["confidence"]

        standard_output = {"aespect":{"POSITIVE": [], "NEUTRAL": [], "NEGATIVE": []},
                           "score":{"POSITIVE": [], "NEUTRAL": [], "NEGATIVE": []},
                           "doc":[]}

        if len(res_aespect) != 0: # if got ANY output
            temp = copy.deepcopy(standard_output)
            temp["doc"].append(result[0]["sentence"])
            for j in range(len(res_aespect)):
                if res_sentiment[j] == "Positive":
                    temp["aespect"]["POSITIVE"].append(res_aespect[j])
                    temp["score"]["POSITIVE"].append(res_confidence[j])
                if res_sentiment[j] == "Neutral":
                    temp["aespect"]["NEUTRAL"].append(res_aespect[j])
                    temp["score"]["NEUTRAL"].append(res_confidence[j])
                if res_sentiment[j] == "Negative":
                    temp["aespect"]["NEGATIVE"].append(res_aespect[j])
                    temp["score"]["NEGATIVE"].append(res_confidence[j])
            return temp
        else:
            return None

    def batch_predict(self,reviews): # not ready yet
        standard_output = {"aespect":{"POSITIVE": [], "NEUTRAL": [], "NEGATIVE": []},
                           "score":{"POSITIVE": [], "NEUTRAL": [], "NEGATIVE": []},
                           "doc":[]}
        output = []

        rejected_index = {}

        for i in range(len(reviews)):
            result = self.predict(reviews[i])

            res_aespect = result[0]["aspect"]
            res_sentiment = result[0]["sentiment"]
            res_confidence = result[0]["confidence"]

            if len(res_aespect) != 0: # if got ANY output
                temp = copy.deepcopy(standard_output)
                temp["doc"].append(reviews[i])
                for j in range(len(res_aespect)):
                    if res_sentiment[j] == "Positive":
                        temp["aespect"]["POSITIVE"].append(res_aespect[j])
                        temp["score"]["POSITIVE"].append(res_confidence[j])
                    if res_sentiment[j] == "Neutral":
                        temp["aespect"]["NEUTRAL"].append(res_aespect[j])
                        temp["score"]["NEUTRAL"].append(res_confidence[j])
                    if res_sentiment[j] == "Negative":
                        temp["aespect"]["NEGATIVE"].append(res_aespect[j])
                        temp["score"]["NEGATIVE"].append(res_confidence[j])

                output.append(temp)
            else:
                rejected_index[i] = reviews[i]

        return output

    def get_results(self):
        return self.saved_result

class IndividualAespectMiner:

    def __init__(self):
        from nltk.stem import WordNetLemmatizer
        import itertools
        self.itertools = itertools
        self.lemmatizer = WordNetLemmatizer()

    def process(self,results):
        common_aespects = self.unique_intersection(results)
        aespect_score = self.filter_aespects_score(common_aespects,results)
        return self.ratio_caluclator(aespect_score)
    
    def unique_intersection(self,result):
        from collections import Counter
        all_aespects = []
        for i in range(len(result)):    
            all_aespects.append(self.itertools.chain.from_iterable(list(result[i]["aespect"].values())))

        all_aespects = list(self.itertools.chain.from_iterable(all_aespects)) # reshape to -1
        lemma_all_aespects = [self.lemmatizer.lemmatize(items.lower()) for items in all_aespects]
        common = Counter(lemma_all_aespects)
        common_aespects = [k for k, v in common.items() if v > 1]
        return common_aespects

    def filter_aespects_score(self,unique, results):
    # standard = {"PLACEHOLDER": {"POSITIVE":[],"NEUTRAL":[],"NEGATIVE":[]}}
        output = {}

        for record in results:
            labels = ["POSITIVE","NEUTRAL","NEGATIVE"]
            for label in labels:
                record_aespect = record["aespect"][label]
                if record_aespect:
                    lemma_record_aespect = [self.lemmatizer.lemmatize(items.lower()) for items in record_aespect]
                    intersection_aespect = list(set(lemma_record_aespect).intersection(unique))
                    if intersection_aespect:
                        for aespect in intersection_aespect:
                            if aespect in output.keys():
                                output[aespect][label].append(record["score"][label][0])
                            else:
                                output[aespect] = {"POSITIVE":[],"NEUTRAL":[],"NEGATIVE":[]}
                                output[aespect][label].append(record["score"][label][0])
        return output
        
    def ratio_caluclator(self,individual_results):
        standard = {"aespect":None,"ratio":[],"score":None}
        output = []
        for aespect in individual_results:
            temp = copy.deepcopy(standard)
            temp["aespect"] = aespect

            pos_score = individual_results[aespect]["POSITIVE"]
            neu_score = individual_results[aespect]["NEUTRAL"] 
            neg_score = individual_results[aespect]["NEGATIVE"]
            # ratio
            total_len = len(pos_score) + len(neu_score) + len(neg_score)

            pos_ratio = (100*len(pos_score))/total_len
            neu_ratio = (100*len(neu_score))/total_len
            neg_ratio = (100*len(neg_score))/total_len

            temp["ratio"] = [round(pos_ratio),round(neu_ratio),round(neg_ratio)]
            # score    
            sum_pos = sum(pos_score)/len(pos_score) if pos_score else 0
            sum_neu = sum(neu_score)/len(neu_score) if neu_score else 0
            sum_neg = sum(neg_score)/len(neg_score) if neg_score else 0

            temp["score"] = (sum_pos + (sum_neu/2)) - (sum_neg + (sum_neu/2))

            output.append(temp)

        return output

class TopicModel:

    def __init__(self,docs):
        print("Initilizing models for TopicModel")
        n = 1
        while(True):
            try:
                from top2vec import Top2Vec
                self.model = Top2Vec(docs*n,embedding_model='universal-sentence-encoder') #try bert
                break
            except Exception as e: 
                print(e)
                print('Failed to find topics due to low samples')
                print('Attempting fix')
                n = n * 5
                print('n - {}'.format(n))
        # from top2vec import Top2Vec
        # self.model = Top2Vec(docs) 

    def topical_words(self): 
        return self.model.get_topics()[0]

    def get_topics_allkeywords(self,topics): # index starts from 1
        topic_dict = {}
        for i in range(len(topics)):
            for j in range(len(topics[i])):
                key = "{}-{}".format(i+1,topics[i][j])
                value = self.model.search_documents_by_keywords(keywords=[topics[i][j]],num_docs=1)[0][0] # i+1 cz jinja loop counter starts with 1
                topic_dict[key] = value
            
        return topic_dict # CONVENTION : {button_id:centriod_doc}

    def filter_words(self,topics): # combined with stop words
        import nltk
        from nltk.corpus import stopwords
        reject_pos = ['DT','IN','MD','PDT','PRP','PRP$','RP','TO','WDT','WP','WRB','RB',"CC"] # risky - VBZ, VBP, VBD
        stopwords = stopwords.words('english')
        marker = "<DEL>"

        for i in range(len(topics)):
            for j in range(len(topics[i])):
                pos = nltk.pos_tag([topics[i][j]])[0][1]
                if pos in reject_pos or topics[i][j] in stopwords:
                    topics[i][j] = marker

        clean_topics = []
        for topic in topics:
            topic = list(filter((marker).__ne__, topic))
            clean_topics.append(topic)

        return clean_topics

    def top2vec_wordcloud(self,model,background_color = "white"):
        from scipy.special import softmax
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import io
        import base64

        img = io.BytesIO()
        b64out = []
        for i in range(len(model.get_topic_sizes()[1])):
            topic_num = i
            reduced = False
            background_color = background_color
    
            model._validate_topic_num(topic_num, reduced)
            word_score_dict = dict(zip(model.topic_words[topic_num],softmax(model.topic_word_scores[topic_num])))
    
            plt.figure(figsize=(16, 4),dpi=200)
            plt.axis("off")
            plt.imshow(WordCloud(width=1600,height=400,background_color=background_color).generate_from_frequencies(word_score_dict))
            # plt.title("Topic " + str(topic_num), loc='left', fontsize=25, pad=20)
            plt.savefig(img,format = "png")
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            b64out.append(plot_url)
        
        return b64out

# class TextTransformation:
#     model_insta = []
    
#     def load_models(self,model):
#         if type(model) == list:
#             self.model_insta = self.model_insta + model
#         else:
#             self.model_insta = self.model_insta + [model]
            
#     def models(self):
#         if self.model_insta:
#             return self.model_insta
#         else:
#             print("No Models Found.")