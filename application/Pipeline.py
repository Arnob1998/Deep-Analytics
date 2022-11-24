from application.Database import Aespect, Data, Review, SimilarityCluster, AespectIndividual, SBAEModelFlow, SBAEModelSelection,TopicModel
from application import db

class PipeLine:
    # saved_result = None

    def load_site_data(self,loc,site="amazon"):
        from application.DataPrep import AmazonDataPrep,DataLoader, SytheticDataMaker, Sanitizer
        data_loader = DataLoader()
        syn_maker = SytheticDataMaker()
        san = Sanitizer()
        data = data_loader.load_file(loc)
        if site == "amazon":
            amazon_prep = AmazonDataPrep()
            processed_data = amazon_prep.process(data)
            processed_data = san.datetime_finder(syn_maker.synthetic_date_adder(processed_data))
            return processed_data


class RoutePipeLine:

    def __init__(self, file_name):
        self.data_id = Data.query.filter_by(name=file_name).first().id
        self.cat_threshold = .70

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

            # for ["SBAE"]["l1"]
            len_total = len(Review.query.filter(Review.data_id == self.data_id).all())
            len_aespct = len(Aespect.query.filter(Aespect.data_id == self.data_id).all())

            # assign
            placeholder_data["SBAE"]["l1"] = (len_aespct * 100)/len_total
            if "Rejected" in dict_percent.keys():
                placeholder_data["SBAE"]["l2"]["percent"] = 100-dict_percent["Rejected"]
            else:
                placeholder_data["SBAE"]["l2"]["percent"] = 100
            placeholder_data["SBAE"]["l3"] = dict_percent
            placeholder_data["SBAE"]["lt"]["k"] = list(dict_percent.keys())
            placeholder_data["SBAE"]["lt"]["v"] = list(dict_percent.values())

        # ----------------A2S------------------
        exists_aespectindividual = AespectIndividual.query.filter_by(data_id=self.data_id).first() is not None
        if exists_aespect and exists_aespectindividual:
            
            pos = 0
            neu = 0
            neg = 0

            pos_score = []
            neu_score = []
            neg_score = []

            classy_dict = {}

            aes_out = AespectIndividual.query.with_entities(AespectIndividual.content).filter(AespectIndividual.data_id==self.data_id,AespectIndividual.content!=None).all()
            
            for a in aes_out:     
                pos_score.append(a[0]["Chart_Stat"]["mean"][0])
                neu_score.append(a[0]["Chart_Stat"]["mean"][1])
                neg_score.append(a[0]["Chart_Stat"]["mean"][2])
                pos += a[0]["Chart_SentRatio"]["POSITIVE"]
                neu += a[0]["Chart_SentRatio"]["NEUTRAL"]
                neg += a[0]["Chart_SentRatio"]["NEGATIVE"]

                for class_ in a[0]["Chart_Classy"]:
                    if class_ not in classy_dict:
                        classy_dict[class_] = 0
                    classy_dict[class_] += sum(a[0]["Chart_Classy"][class_])

            total = pos+neu+neg

            stat_msg = "NA"
            pos_score = sum(pos_score)
            neu_score = sum(neu_score)
            neg_score = sum(neg_score)
            total_score = max(pos_score,neu_score,neg_score)

            if pos_score ==  total_score:
                stat_msg = "Positive"
            elif neu_score == total_score:
                stat_msg = "Neutral"
            elif neg_score == total_score:
                stat_msg = "Negative"                
        
            # assign
            placeholder_data["A2S"]["ratio"] = [(pos*100)/total,(neu*100)/total,(neg*100)/total]
            placeholder_data["A2S"]["u_aespect"] = len(aes_out)
            placeholder_data["A2S"]["stat"] = stat_msg
            placeholder_data["A2S"]["freq_class"] = max(classy_dict, key=classy_dict.get)
 
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
        cluster = SimilarityCluster.query.with_entities(SimilarityCluster.cluster).filter(SimilarityCluster.data_id==self.data_id).first()[0]
        return cluster

    def route_individual_aespect(self):
        indi_aespect_out = []
        indi_aespect_out_db = AespectIndividual.query.filter(AespectIndividual.content != None, AespectIndividual.data_id == self.data_id).all()

        for indi_aes in indi_aespect_out_db:
            indi_aespect_out.append(indi_aes.content)

        return indi_aespect_out

    def route_all_aespect(self):
        aespect_out = Aespect.query.filter(Aespect.content != None, Aespect.data_id == self.data_id).all()
        aespect_filtout = []

        for aes in aespect_out:
            aes.content["category"] = []

            for cat in aes.category:
                if aes.category[cat] >= self.cat_threshold:
                    aes.content["category"].append(cat)

            aespect_filtout.append(aes.content)
        
        return aespect_filtout              

    def route_topic_model(self):
        q_res = TopicModel.query.with_entities(TopicModel.topic_words,TopicModel.topic_dict,TopicModel.topic_cloud).filter(TopicModel.data_id==self.data_id).first()
        t_words = q_res.topic_words
        t_dict = q_res.topic_dict
        t_cloud = q_res.topic_cloud
        return t_words,t_dict,t_cloud

class TextTransformationModel:

    class Default:

        model_name = "No Change"
        base_name = "None"
        transformation = "Default"
        description = "The paragraph isn't altered in anyway."

        def __init__(self):
            print("Initilizing models for Default")

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

