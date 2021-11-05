###
import pickle
import pymongo
import json
import pyspark
from pyspark import SparkContext 
from pyspark import conf
from pyspark.sql import SparkSession, udf
from pyspark.sql import SQLContext
import string
import re
import pyspark.ml.feature
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import lower,col , udf,countDistinct
from pyspark.ml.feature import Tokenizer,RegexTokenizer,StopWordsRemover,StringIndexer
from pyspark.ml.feature import NGram,HashingTF,IDF ,CountVectorizer,VectorAssembler
#from pyspark.ml.classification import LogisticRegression,NaiveBayes,RandomForestClassifier,GBTClassifier
from pyspark.ml.evaluation import Evaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType
from pyspark.ml.param import *
import pandas as pd
from flask import Flask,render_template,url_for,request
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():

     df= pd.read_csv("static/news_csv.csv", encoding="latin-1") 
      #Convert to lowercase
     def textclean(text):
      text = lower(text)
     # text = regexp_replace(text,"\+[.*?\]","")
      text = regexp_replace(text,"^rt","")
      # text = regexp_replace(text,"\w*\d\w*","")
      text = regexp_replace(text,"[0-9]","")
      text = regexp_replace(text,"(https?\://)\S+","")
      text = regexp_replace(text,'\[.*?\]', '')
      text = regexp_replace(text,'/', '')
      text = regexp_replace(text,':', '')
      text = regexp_replace(text,'%', '')
      text = regexp_replace(text,'\n', '')
      #text = regexp_replace(text,'-,', '')
      return text
     df = df.select(textclean(col("category")).alias("category"),
     #textclean(col("title")).alias("title")
     textclean(col("summary")).alias("summary"))

     #Value counts
     df_countacat=df.groupBy("category").count()
     #df_countacat.show()


     # Data Cleaning
     # Handling null values-droping null records
     df=df.dropna(subset=("category"))
     #df.show()


     #Feature Extraction

     #Build Features From Text
     #    CountVectorizer
     #    TFIDF
     #    WordEmbedding
     #    HashingTF

     #dir(pyspark.ml.feature)
     #print(dir(pyspark.ml.feature))

     # Stages For the Pipeline
     tokenizer = Tokenizer(inputCol='summary',outputCol='mytokens')
     stopwords_remover = StopWordsRemover(inputCol='mytokens',outputCol='filtered_tokens')
     vectorizer = CountVectorizer(inputCol='filtered_tokens',outputCol='rawFeatures')
     idf = IDF(inputCol='rawFeatures',outputCol='vectorizedFeatures')

     # LabelEncoding/LabelIndexing
     labelEncoder = StringIndexer(inputCol='category',outputCol='label').fit(df)
     

     #labelEncoder.transform(df).show(5)

     # Dict of Labels
     label_dict = {'science':0.0, 'business':1.0,'economics':2.0,'finance':3.0,'tech':4.0,
     'gaming':5.0, 'entertainment':6.0, 'sport':7.0,'beauty':8.0, 'politics':9.0, 
     'world':10.0, 'energy':11.0, 'food':12.0, 'travel':13.0}

     df = labelEncoder.transform(df)
     X=df["summary"]
     y=df["label"]

     ### Split Dataset
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

     ###Building the Pipeline

     pipeline = Pipeline(stages=[tokenizer,stopwords_remover,vectorizer,idf])

     X=pipeline.fit(X_train,y_train)

     ###Estimator - Logistic Regression

     lr = LogisticRegression(featuresCol='vectorizedFeatures',labelCol='label')

     #print(pipeline.stages)


     # Building MOdel
     lr.fit(X_train,y_train)
     lr.transform(X_test,y_test)

     if request.method == 'POST':
         message = request.form['message']
	    
         data = [message]
		
         vect = lr.transform(data).toarray()
		
         my_prediction = lr.predict(vect)
        
         return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)
