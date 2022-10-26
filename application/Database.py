from application import db

class Data(db.Model):
    id = db.Column(db.Integer,primary_key = True)
    name = db.Column(db.Text, nullable = False)
    relation_review = db.relationship("Review", backref = "ref_review", lazy = True)

    def __repr__(self):
        return f"Data('{self.name}')"   

class Review(db.Model):
    id = db.Column(db.Integer,primary_key = True)
    rating = db.Column(db.Integer, nullable = False)
    title = db.Column(db.Text, nullable = False)
    review = db.Column(db.Text, nullable = False)
    data_id = db.Column(db.Integer, db.ForeignKey("data.id"), nullable = False)   
    # relation_summerize = db.relationship("Summarize", backref = "ref_summarize", lazy = True)

    def __repr__(self):
        return f"Review('{self.rating}','{self.title}','{self.review}')"

# class AespectLog(db.Model): # or paraphrased or just title + review
#     id = db.Column(db.Integer,primary_key = True)
#     transformation = db.Column(db.Text, default=None)  # default is review with no aespects
#     model_name = db.Column(db.Text, default=None) 
#     data_id = db.Column(db.Integer, db.ForeignKey("data.id"), nullable = False)
#     review_id = db.Column(db.Integer, db.ForeignKey("review.id"), nullable = False)

#     def __repr__(self):
#         return f"AespectLog('{self.transformation}','{self.model_name}')" 

class XSummarize(db.Model): # or paraphrased or just title + review
    id = db.Column(db.Integer,primary_key = True)
    content = db.Column(db.Text, default=None)
    data_id = db.Column(db.Integer, db.ForeignKey("data.id"), nullable = False)
    review_id = db.Column(db.Integer, db.ForeignKey("review.id"), nullable = False)

    def __repr__(self):
        return f"Review('{self.content}')" 

class SimilarityCluster(db.Model):
    id = db.Column(db.Integer,primary_key = True)
    cluster = db.Column(db.JSON, nullable = False)
    data_id = db.Column(db.Integer, db.ForeignKey("data.id"), nullable = False)

    def __repr__(self):
        return f"SimilarityCluster('{self.cluster}')" 

class Aespect(db.Model):
    id = db.Column(db.Integer, primary_key = True) # if no aespect just pass content-id
    content = db.Column(db.JSON, default=None)
    transformation = db.Column(db.Text, default=None)  # default is review with no aespects
    model_name = db.Column(db.Text, default=None) 
    data_id = db.Column(db.Integer, db.ForeignKey("data.id"), nullable = False)
    review_id = db.Column(db.Integer, db.ForeignKey("review.id"), nullable = False)
    # summarize_id = db.Column(db.Integer, db.ForeignKey("summarize.id"), nullable = False)

    def __repr__(self):
        return f"Review('{self.content}','{self.transformation}','{self.model_name}')" 

class AespectIndividual(db.Model):
    id = db.Column(db.Integer,primary_key = True)
    content = db.Column(db.JSON, nullable = False)
    data_id = db.Column(db.Integer, db.ForeignKey("data.id"), nullable = False)

    def __repr__(self):
        return f"AespectIndividual('{self.content}')" 

class SBAEModelFlow(db.Model):
    id = db.Column(db.Integer,primary_key = True) 
    name = db.Column(db.Text, nullable = False)
    flow = db.Column(db.JSON, nullable = False)

    def __repr__(self):
        return f"SBAEModelFlow('{self.name}','{self.flow}')"     

class SBAEModelSelection(db.Model):
    id = db.Column(db.Integer,primary_key = True) 
    current_flow = db.Column(db.Text, nullable = False) # name of SBAEModelFlow
    data_id = db.Column(db.Integer, db.ForeignKey("data.id"), nullable = False)

    def __repr__(self):
        return f"SBAEModelSelection('{self.current_flow}')"      

# python 
# from application import db
# from application.Database import Data, Review, Summarize, Aespect

# db.create_all()

# data1 = Data(name="first.txt")
# db.session.add(data1)

# data2 = Data(name="second.txt")
# db.session.add(data2)

# db.session.commit()

# rev1_1 = Review(rating=4, title="1-1_title", review="1-1_review", data_id=data1.id)
# db.session.add(rev1_1)

# rev2_1 = Review(rating=4, title="2-1_title", review="2-1_review", data_id=data2.id)
# db.session.add(rev2_1)

# sum1_1 = Summarize(content="sum1_1", data_id=1 , review_id=1)
# sum1_2 = Summarize(content="sum1_2", data_id=1 , review_id=2)
# sum2_1 = Summarize(content="sum2_1", data_id=2 , review_id=1)
# db.session.add(sum1_1)

# db.session.commit()


# QUERY
# find
# d1 = Data.query.get(1)
# Summarize.query.filter(Summarize.data_id ==1).all()
# Summarize.query.filter(Summarize.data_id ==1,Summarize.review_id==1).all()

# Check exist
# d = db.session.query(Data).filter_by(name='John Smith').exists()
# db.session.query(d).scalar()

# get specific column
# Data.query.filter_by(name="first.txt").first().id


# pd.read_sql_query(sql = Review.query.filter_by(data_id=1).statement, con=db.session.bind)

# Query specific column with condition
# Aespect.query.with_entities(Aespect.transformation,Aespect.model_name).filter(Aespect.content !=None,Aespect.data_id==1).all()


## dekte
# Aespect.query.filter(Aespect.data_id==2).delete()
# db.session.commit()

## detele whole table
## SBAEModelSelection.__table__.drop(db.engine)