import numpy as np
import sys
from application.Database import Aespect, Data, Review, SimilarityCluster, AespectIndividual, SBAEModelFlow, SBAEModelSelection
from application import db
import pandas as pd

class PipeLine:
    saved_result = None

    def load_site_data(self,loc,site="amazon"):
        from application.DataPrep import AmazonDataPrep,DataLoader
        data_loader = DataLoader()
        data = data_loader.load_file(loc)
        if site == "amazon":
            amazon_prep = AmazonDataPrep()
            processed_data = amazon_prep.process(data)
            return processed_data

    def normalize_rating(self,rating):
        return rating/5.0

    def sent_analysis(self,model,result,rating):
        doc = result["doc"][0]
        sent_score = model.default_score(doc)
        if result["score"]["POSITIVE"]:
            result["score"]["POSITIVE"] = [np.array([self.normalize_rating(rating),sent_score[2]]).mean()]
        if result["score"]["NEUTRAL"]:
            result["score"]["NEUTRAL"] = [np.array([self.normalize_rating(rating),sent_score[1]]).mean()]
        if result["score"]["NEGATIVE"]:
            result["score"]["NEGATIVE"] = [np.array([self.normalize_rating(rating),sent_score[0]]).mean()]

        return result

    def aespect_mining_singlular(self,data,model_flow,data_id):
        from application.DataPrep import Sanitizer
        from application.NLPModel import Summmerizer,AespectMiner,Paraphraser,SentimentAnalysis
        cleaner = Sanitizer()
        miner = AespectMiner()
        sent_analyzer = SentimentAnalysis()
        # text_aug = Paraphraser()
        # summerizer = Summmerizer()

        tt_pipe = TextTransformationModel()
        # set flow
        model_dict = tt_pipe.model_dict()
        flow_serial = [int(index) for index in model_flow.keys()]
        flow_serial.sort()

        data = pd.read_sql_query(sql = Review.query.filter_by(data_id=data_id).statement, con=db.session.bind)   

        title = data["title"]
        content = data["review"]
        rating = data["rating"]

        output = []
        rejected_output = {}

        for i in range(len(data)):
            print("INDEX : {}".format(i))
            cleaned_title = cleaner.add_endofline(cleaner.remove_leading_whitespace(cleaner.remove_escapes(title[i])))
            cleaned_content = cleaner.add_endofline(cleaner.remove_leading_whitespace(cleaner.remove_escapes(content[i])))

            ready_data = cleaned_title + cleaned_content

            for model_index in range(len(flow_serial)):
                model = model_dict[model_flow[str(model_index)]]() # class
                out = model.rephrase(ready_data)             
                if out[1] == False: # if repharse fail skip to next
                    print("Rephrase Failed. Skipping to next Model")
                    continue

                aespect_result = miner.predict(out[0])
                result = miner.process_result(aespect_result)
                if result:
                    result = self.sent_analysis(sent_analyzer,result,rating[i])
                    aes = Aespect(content=result,transformation=model.transformation, model_name=model.model_name ,data_id=data_id, review_id=i)
                    db.session.add(aes)
                    db.session.commit()
                    output.append(result)
                    print("{} model was successful".format(model.model_name))
                    break # next review
                elif model_index == len(flow_serial)-1 and result == None:
                    print("All Models failed. Rejecting Review")
                    rejected_output[i] = ready_data

            # TEST
            if i == 20:
                break;
            # review_data = summerizer.summerize(ready_data)

            # aespect_result = miner.predict(review_data[0])
            # result = miner.process_result(aespect_result)
            # if result:
            #     result = self.sent_analysis(sent_analyzer,result,rating[i])
            #     output.append(result)
                
            #     if review_data[1]:
            #         aes = Aespect(content=result,transformation="Summerize", model_name=r"facebook/bart-large-cnn" ,data_id=data_id, review_id=i)
            #         db.session.add(aes)
            #         db.session.commit()
            #     else:
            #         aes = Aespect(content=result,transformation="Default",data_id=data_id, review_id=i)
            #         db.session.add(aes)
            #         db.session.commit()
                    
            # else: # if not aespect found
            #     print("No Aespects found")
            #     print("ATTEMPTING PARAPHRASE")
            #     text_augmented = text_aug.paraphraser(ready_data)
            #     aespect_result = miner.predict(text_augmented)
            #     result = miner.process_result(aespect_result)
            #     if result:
            #         result = self.sent_analysis(sent_analyzer,result,rating[i])
            #         output.append(result) #NOTE:  upload to DB 
            #         print("PARAPHRASE SUCCESS")
                    
            #         aes = Aespect(content=result, transformation="Paraphrase", model_name=r"tuner007/pegasus_paraphrase", data_id=data_id, review_id=i)
            #         db.session.add(aes)
            #         db.session.commit()
            #     else:
            #         rejected_output[i] = text_augmented
            #         print("PARAPHRASE FAILED")
            #         aes = Aespect(data_id=data_id, review_id=i)
            #         db.session.add(aes)
            #         db.session.commit()
        
        # self.saved_result = output # save for individual score # NOTE:  upload to DB
        print("REJECTED LOG:")
        print(rejected_output)
        return output # note check for null

    def aespect_mining_individual(self):
        from application.NLPModel import IndividualAespectMiner
        individual_miner = IndividualAespectMiner()
        if self.saved_result:
            return individual_miner.process(self.saved_result)
        else:
            print("Failed to process Individual Aespect")
            print("No Data found for Individual Aespect")
            sys.exit(1)

    def aespect_mining_individual_test(self,result):
        from application.NLPModel import IndividualAespectMiner
        individual_miner = IndividualAespectMiner() 
        if result:
            return individual_miner.process(result)
        else:
            print("Failed to process Individual Aespect")
            print("No Data found for Individual Aespect")
            sys.exit(1)


    def new_cluster(self,data_id):
        from application.NLPModel import SentenceSimiliarity, XSummerizer
        from sklearn.metrics.pairwise import cosine_similarity
        sent_simi = SentenceSimiliarity()

        content = Review.query.with_entities(Review.title,Review.review).filter(Review.data_id==data_id).all()
        from application.DataPrep import Sanitizer
        cleaner = Sanitizer()
        x_summer = XSummerizer()
        docs = []
        for i in range(len(content)):
            cleaned_title = cleaner.add_endofline(cleaner.remove_leading_whitespace(cleaner.remove_escapes(content[i][0])))
            cleaned_content = cleaner.add_endofline(cleaner.remove_leading_whitespace(cleaner.remove_escapes(content[i][1])))
            ready_data = cleaned_title + cleaned_content 

            print("INDEX {}-----------------------------".format(i))
            xsummed = x_summer.summerize_xtream(ready_data)
            docs.append(xsummed)

        doc_embeddings = sent_simi.batch_embeddings(docs)
        data = dict(zip(docs,doc_embeddings))

        cluster_content = []
        threshold_found = None
    
        cluster_index = -1
        while(True):
            i = len(data)-1 # reset loop
            cluster_index = cluster_index + 1
            cluster_content.append([]) # create for next cluster
            if threshold_found == False: # if no threshold found or if no more in data
                break
            else:
                threshold_found = False # if found reset this param
            
            if threshold_found == None: # first time
                threshold_found = False
            
            while(i >= 0): # hit break on 0
                score = cosine_similarity(data[list(data.keys())[0]].reshape(1, -1),data[list(data.keys())[i]].reshape(1,-1))[0][0]
                if score > .80:
                    cluster_content[cluster_index].append(list(data.keys())[i]) 
                    data.pop(list(data.keys())[i])
                    threshold_found = True
                i = i - 1

        return self.clean_cluster(cluster_content)

    # TODO not in use yet
    def clean_cluster(self,cluster):
        clean = []
        for c in cluster:
#           if len(c) <= 1:
            if len(c) > 1:
                clean.append(c)
        return clean

    def update_oldcuster(cluster,docs):
        from application.NLPModel import SentenceSimiliarity
        sent_simi = SentenceSimiliarity()
        from sklearn.metrics.pairwise import cosine_similarity
    
        for i in range(len(docs)): # ALT - use while
            embedded_curr_doc = sent_simi.model().encode(docs[i])
            for c in cluster: 
                embedded_c = sent_simi.model().encode(c[0])
                score = cosine_similarity(embedded_curr_doc.reshape(1, -1),embedded_c.reshape(1,-1))[0][0]
                if score >= .80:
                    c.append(docs[i])
                    docs.pop(i)
                
        return docs,cluster

    def topic_modeling(self,data_id):
        from application.NLPModel import TopicModel
        from application.DataPrep import Sanitizer

        content = Review.query.with_entities(Review.title,Review.review).filter(Review.data_id==data_id).all()   
        cleaner = Sanitizer()

        docs = []
        for i in range(len(content)):
            cleaned_title = cleaner.add_endofline(cleaner.remove_leading_whitespace(cleaner.remove_escapes(content[i][0])))
            cleaned_content = cleaner.add_endofline(cleaner.remove_leading_whitespace(cleaner.remove_escapes(content[i][1])))
            ready_data = cleaned_title + cleaned_content
            docs.append(ready_data)

        t_model = TopicModel(docs)
        t_words = t_model.topical_words()
        t_cloud = t_model.top2vec_wordcloud(t_model.model)
        t_words = t_model.filter_words(t_words)
        t_dict = t_model.get_topics_allkeywords(t_words)
        
        return t_words,t_dict,t_cloud

class RoutePipeLine:

    def __init__(self, file_name):
        self.data_id = Data.query.filter_by(name=file_name).first().id

    def route_models(self,placeholder_data):
        from collections import Counter
        from itertools import chain
        import functools
        import operator

        def percentage_dict(d):
            percent_calc = lambda n, total : 100 * n/total
            tot = sum(d.values())
            new_dict = {}
            for k,v in d.items():
                if k == None:
                    new_dict["Rejected"] = percent_calc(v,tot)
                else:
                    new_dict[k]= percent_calc(v,tot)
            return new_dict

        # ----------------SBAE------------------
        exists_aespect = Aespect.query.filter_by(data_id=self.data_id).first() is not None
        if exists_aespect:
            aespect_filt = Aespect.query.with_entities(Aespect.transformation).filter(Aespect.data_id==self.data_id).all()
            transformation_name = [t[0] for t in aespect_filt]
            count_aes = dict(Counter(transformation_name))

            dict_percent = percentage_dict(count_aes)

            # assign
            placeholder_data["SBAE"]["l1"] = 100
            placeholder_data["SBAE"]["l2"]["percent"] = 100-dict_percent["Rejected"]
            placeholder_data["SBAE"]["l3"] = dict_percent
            placeholder_data["SBAE"]["lt"]["k"] = list(dict_percent.keys())
            placeholder_data["SBAE"]["lt"]["v"] = list(dict_percent.values())

        # ----------------A2S------------------
        exists_aespectindividual = AespectIndividual.query.filter_by(data_id=self.data_id).first() is not None
        if exists_aespect and exists_aespectindividual:
            aespect_content = Aespect.query.with_entities(Aespect.content).filter(Aespect.data_id==self.data_id,Aespect.content!=None).all()  
            aespect_content = list(chain(*aespect_content))

            pos = []
            neu = []
            neg = []

            for i in range(len(aespect_content)):
                score = aespect_content[i]["score"]
                pos.append(score["POSITIVE"])
                neu.append(score["NEUTRAL"])
                neg.append(score["NEGATIVE"])   

            pos = functools.reduce(operator.iconcat, pos, [])
            neu = functools.reduce(operator.iconcat, neu, [])
            neg = functools.reduce(operator.iconcat, neg, [])

            aes_out = AespectIndividual.query.with_entities(AespectIndividual.content).filter(AespectIndividual.data_id==self.data_id,AespectIndividual.content!=None).first()[0]

            aes_uname = []
            aes_pos_score = []
            aes_neg_score = []

            for i in range(len(aes_out)):
                aes_uname.append(aes_out[i]["aespect"])
                if aes_out[i]["score"] < 0:
                    aes_neg_score.append(aes_out[i]["score"])
                else:
                    aes_pos_score.append(aes_out[i]["score"])

            aes_pos_score = np.mean(aes_pos_score)
            aes_neg_score = np.absolute(np.mean(aes_neg_score))

            stat_msg = "NA"
            if aes_pos_score > aes_neg_score:
                if aes_pos_score >= 0.70:
                    stat_msg = "Highly Positive"
                elif aes_pos_score <= 0.30:
                    stat_msg = "Slightly Positive"
                else:
                    stat_msg = "Positive" 
            elif aes_neg_score > aes_pos_score:
                if aes_neg_score >= 0.70:
                    stat_msg = "Highly Negative"
                elif aes_neg_score <= 0.30:
                    stat_msg = "Slightly Negative"
                else:
                    stat_msg = "Negative" 
            else:
                stat_msg = "Neutral"
        
            # assign
            placeholder_data["A2S"]["ratio"] = [np.mean(pos)*100,np.mean(neu)*100,np.mean(neg)*100]
            placeholder_data["A2S"]["u_aespect"] = len(aes_uname)
            placeholder_data["A2S"]["stat"] = stat_msg

        # ----------------SMC------------------
        exists_similaritycluster = SimilarityCluster.query.filter_by(data_id=self.data_id).first() is not None
        if exists_similaritycluster:
            model_clust = SimilarityCluster.query.with_entities(SimilarityCluster.cluster).filter(SimilarityCluster.data_id==self.data_id).first()[0]
            model_clust_len = [len(item) for item in model_clust]
            # assign
            placeholder_data["SMC"]["l1"] = 100
            placeholder_data["SMC"]["l2"] = len(model_clust)
            placeholder_data["SMC"]["l3"] = max(model_clust_len)
            placeholder_data["SMC"]["l4"] = min(model_clust_len)

        return placeholder_data

    def route_similarity_cluster(self):
        exists = SimilarityCluster.query.filter_by(data_id=self.data_id).first() is not None

        if not exists:
            pipe = PipeLine()
            cluster = pipe.new_cluster(self.data_id)
            clust = SimilarityCluster(cluster=cluster, data_id=self.data_id)
            db.session.add(clust)
            db.session.commit()
        else:
            cluster = SimilarityCluster.query.with_entities(SimilarityCluster.cluster).filter(SimilarityCluster.data_id==self.data_id).first()[0]
        return cluster

    def route_individual_aespect(self):
        exists = Aespect.query.filter_by(data_id=self.data_id).first() is not None
        if not exists:
            print("REDITECT")
        else:
            sub_exists = AespectIndividual.query.filter_by(data_id=self.data_id).first() is not None
            if sub_exists:
                indi_aespect_out = AespectIndividual.query.with_entities(AespectIndividual.content).filter(AespectIndividual.data_id==self.data_id).first()[0]
            else:
                aespect_out = Aespect.query.filter(Aespect.content != None, Aespect.data_id == self.data_id).all()
                aespect_out = [aes.content for aes in aespect_out]
                pipe = PipeLine()
                indi_aespect_out = pipe.aespect_mining_individual_test(aespect_out)  
                print("Writting to AespectIndividual")
                indi_aespect_db = AespectIndividual(content=indi_aespect_out, data_id=self.data_id)
                db.session.add(indi_aespect_db)
                db.session.commit() 
        return indi_aespect_out

    def route_all_aespect(self):
        exists = Aespect.query.filter_by(data_id=self.data_id).first() is not None

        if not exists:
            sbae_exists = SBAEModelSelection.query.filter_by(data_id=self.data_id).first() is not None
            model_flow = None
            if not sbae_exists:
                model_flow = SBAEModelFlow.query.filter_by(name="Default").first().flow
            else:
                flow_name = SBAEModelSelection.query.filter_by(data_id=self.data_id).first().current_flow
                model_flow = SBAEModelFlow.query.filter_by(name=flow_name).first().flow
            pipe = PipeLine()
            data = Review.query.filter_by(data_id=self.data_id).all()
            aespect_out = pipe.aespect_mining_singlular(data,model_flow,self.data_id)    
        else:
            aespect_out = Aespect.query.filter(Aespect.content != None, Aespect.data_id == self.data_id).all()
            print(aespect_out[0])
            aespect_out = [aes.content for aes in aespect_out]  
        
        return aespect_out              

        # exists = Aespect.query.filter_by(data_id=self.data_id).first() is not None
        # if not exists:
        #     pipe = PipeLine()
        #     data = Review.query.filter_by(data_id=self.data_id).all()
        #     aespect_out = pipe.aespect_mining_singlular(data, self.data_id)
        # else:
        #     aespect_out = Aespect.query.filter(Aespect.content != None, Aespect.data_id == self.data_id).all()
        #     print(aespect_out[0])
        #     aespect_out = [aes.content for aes in aespect_out]
        # return aespect_out

class TextTransformationModel:

    class Default:

        model_name = "No Change"
        base_name = "None"
        transformation = "Default"
        description = "The paragraph isn't altered in anyway."

        def rephrase(self,text, max_length=None):
            if max_length:
                return text,True
            else:   
                return text,True

    from application.NLPModel import Summmerizer,Paraphraser,XSummerizer
    models = [Default, Summmerizer, Paraphraser,XSummerizer]

    def model_list(self):
        return self.models

    def model_dict(self):
        md = {}
        for model in self.models:
            md[model.model_name] = model
        return md

    def model_name(self,models):
        name = []
        for model in models:
            name.append(model.model_name)
        return name

