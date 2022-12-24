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
        from application.NLPModel import AespectMiner,SentimentAnalysis, ReviewClassifier
        cleaner = Sanitizer()
        miner = AespectMiner()
        sent_analyzer = SentimentAnalysis()
        rev_classifer = ReviewClassifier()

        tt_pipe = TextTransformationModel()
        # set flow
        model_dict = tt_pipe.model_dict()
        flow_serial = [int(index) for index in model_flow.keys()]
        flow_serial.sort()  

        # model_dict = {'No Change': <class 'application.Pipeline.TextTransformationModel.Default'>, 'BART-Summerizer': <class 'application.NLPModel.Summmerizer'>, 'Pegasus-Parapharser': <class 'application.NLPModel.Paraphraser'>, 'Pegasus-Xtrem_Summerizer': <class 'application.NLPModel.XSummerizer'>}
        # model_flow = {'0': 'BART-Summerizer', '1': 'Pegasus-Parapharser', '2': 'No Change'}
        # flow_serial = [0, 1, 2]

        # Pre-load model for optimizaition
        print("Pre-Loading All TextTransformation Models")

        # removing not selected models
        s1 = set(model_flow.values())
        s2 = set(model_dict.keys())
        remove_model = list(s2.difference(s1))
        for m in remove_model:
            del model_dict[m]

        for model_name in model_dict:
            model = model_dict[model_name]() # class    
            model_dict[model_name] = model
        
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
                model = model_dict[model_flow[str(model_index)]]
                out = model.rephrase(ready_data)             
                if out[1] == False: # if repharse fail skip to next
                    print("Rephrase Failed. Skipping to next Model")
                    continue

                aespect_result = miner.predict(out[0])
                result = miner.process_result(aespect_result)
                if result:
                    result = self.sent_analysis(sent_analyzer,result,rating[i])
                    category = rev_classifer.predict(result["doc"])
                    aes = Aespect(content=result,category=category,transformation=model.transformation, model_name=model.model_name ,data_id=data_id, review_id=i)
                    db.session.add(aes)
                    db.session.commit()
                    output.append(result)
                    print("{} model was successful".format(model.model_name))
                    break # next review
                elif model_index == len(flow_serial)-1 and result == None:
                    print("All Models failed. Rejecting Review")                 
                    aes = Aespect(data_id=data_id, review_id=i)
                    db.session.add(aes)
                    db.session.commit()
                    rejected_output[i] = ready_data

        print("REJECTED LOG:")
        print(rejected_output)
        return output # note check for null

    # def aespect_mining_individual(self):
    #     from application.NLPModel import IndividualAespectMiner
    #     individual_miner = IndividualAespectMiner()
    #     if self.saved_result:
    #         return individual_miner.process(self.saved_result)
    #     else:
    #         print("Failed to process Individual Aespect")
    #         print("No Data found for Individual Aespect")
    #         sys.exit(1)

    # def aespect_mining_individual_test(self,result):
    #     from application.NLPModel import IndividualAespectMiner
    #     individual_miner = IndividualAespectMiner() 
    #     if result:
    #         return individual_miner.process(result)
    #     else:
    #         print("Failed to process Individual Aespect")
    #         print("No Data found for Individual Aespect")
    #         sys.exit(1)


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

        # ----------------Model Dependency------------------
        
        dependency = {"A2S" : False, "VCA": False}

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

            if placeholder_data["SBAE"]["l1"] == 100:
                dependency["A2S"] = True

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

            if placeholder_data["A2S"]["ratio"]:
                dependency["VCA"] = True
 
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

        
        # ----------------TPM------------------
        exists_topic = TopicModel.query.filter_by(data_id=self.data_id).first() is not None
        if exists_topic:
            words = TopicModel.query.with_entities(TopicModel.topic_words).filter(TopicModel.data_id==self.data_id).first()[0]

            placeholder_data["TPM"]["l1"] = len(words)

            flat_words = functools.reduce(operator.iconcat, words, [])
            tpm_counts = Counter(flat_words)
            target_word = max(tpm_counts.items(), key=operator.itemgetter(1))[0]
            if tpm_counts[target_word] >= 1:
                pass
            else:
                placeholder_data["TPM"]["l2"] = target_word
            
            placeholder_data["TPM"]["l3"] = len(flat_words)/len(words)


        return placeholder_data, dependency

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
            # add to db
            if not AespectIndividual.query.filter_by(data_id=self.data_id).first() is not None:
                review_db = Review.query.filter_by(data_id=self.data_id).all()
                aespect_db = Aespect.query.filter_by(data_id=self.data_id).all()

                aespect = []
                for i in range(len(aespect_db)):
                    if aespect_db[i].content != None:
                        print(aespect_db[i].review_id, review_db[i].review_id)
                        if aespect_db[i].review_id == review_db[i].review_id:
                            
                            aespect_db[i].content["datetime"] = "{}-{}-{}".format(review_db[i].datetime.year,review_db[i].datetime.month,review_db[i].datetime.day)

                            aespect_db[i].content["category"] = []        
                            for cat in aespect_db[i].category:
                                if aespect_db[i].category[cat] >= self.cat_threshold:
                                    aespect_db[i].content["category"].append(cat)    
 
                            aespect.append(aespect_db[i].content)

                from application.NLPModel import IndividualAespectMiner
                i_miner = IndividualAespectMiner()
                indi_aespect_out = i_miner.a2s_output(aespect)

                for indi_aes in indi_aespect_out:
                    indi_aes_db = AespectIndividual(content = indi_aes, data_id=self.data_id)
                    db.session.add(indi_aes_db)
                    db.session.commit()

                return indi_aespect_out

            else:
                indi_aespect_out = []
                indi_aespect_out_db = AespectIndividual.query.filter(AespectIndividual.content != None, AespectIndividual.data_id == self.data_id).all()

                for indi_aes in indi_aespect_out_db:
                    indi_aespect_out.append(indi_aes.content)

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
            aespect_filtout = []

            import functools,operator

            for aes in aespect_out:
                aes.content["category"] = []

                for cat in aes.category:
                    if aes.category[cat] >= self.cat_threshold:
                        aes.content["category"].append(cat)

                aespect_filtout.append(aes.content)
        
        return aespect_filtout     

    def route_topic_model(self):
        exists = TopicModel.query.filter_by(data_id=self.data_id).first() is not None
        if not exists:
            pipe = PipeLine()
            t_words,t_dict,t_cloud = pipe.topic_modeling(self.data_id)
            tp_db = TopicModel(topic_words = t_words,topic_dict = t_dict, topic_cloud = t_cloud,data_id=self.data_id)
            db.session.add(tp_db)
            db.session.commit()
            return t_words,t_dict,t_cloud
        else:
            q_res = TopicModel.query.with_entities(TopicModel.topic_words,TopicModel.topic_dict,TopicModel.topic_cloud).filter(TopicModel.data_id==self.data_id).first()
            t_words = q_res.topic_words
            t_dict = q_res.topic_dict
            t_cloud = q_res.topic_cloud
            return t_words,t_dict,t_cloud         

    def route_v_assistant_ui(self):
        
        if AespectIndividual.query.filter_by(data_id=self.data_id).first() is not None:
            indi_aespect_out = []
            indi_aespect_out_db = AespectIndividual.query.filter(AespectIndividual.content != None, AespectIndividual.data_id == self.data_id).all()

            for indi_aes in indi_aespect_out_db:
                indi_aespect_out.append(indi_aes.content["name"])
        else:
            indi_aespect_out = None        
        
        return indi_aespect_out

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

