#connection,cleaning and preprocessing
#from os import truncate
import pymongo
import json
import pyspark
from pyspark import SparkContext
from pyspark import conf
from pyspark.sql import SparkSession, udf
from pyspark.sql import SQLContext
import string
import re
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import lower,col , udf,countDistinct
from pyspark.ml.feature import Tokenizer,RegexTokenizer,StopWordsRemover,StringIndexer
from pyspark.ml.feature import NGram,HashingTF,IDF ,CountVectorizer,VectorAssembler
from pyspark.ml.classification import LogisticRegression,NaiveBayes,RandomForestClassifier,GBTClassifier
from pyspark.ml.evaluation import Evaluator, MulticlassClassificationEvaluator
from pyspark.sql.types import IntegerType
from pyspark.ml.param import *

import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
#import pandas as pd
#from pymongo import MongoClient

#print(pyspark.__version__)
#connection conf

# load mongo data
input_uri = "mongodb://localhost:27017/capstone.news"
output_uri = "mongodb://localhost:27017/capstone.news"

myspark = SparkSession\
    .builder\
    .appName("mynews")\
    .config("spark.mongodb.input.uri", input_uri)\
    .config("spark.mongodb.output.uri", output_uri)\
    .config('spark.jars.packages','org.mongodb.spark:mongo-spark-connector_2.12:2.4.2')\
    .getOrCreate()

df = myspark.read.format('com.mongodb.spark.sql.DefaultSource').load()

#print(df.printSchema())
#print(df.first())

# Data Cleaning
# Handling null values-droping null records
rowcount=df.count()
 #print(rowcount)
 #nul_drop = df.dropna()
 #rowcount_null = nul_drop.count()
 #print("before",rowcount)
 #print("after",rowcount_null)
df=df.dropna()

# Removing "news" category - 
df.createOrReplaceTempView("capstone_news")
sql_query = "select * from capstone_news where category NOT IN ('news','nation') "
df=myspark.sql(sql_query)

#nonews = new_df.count()
#print("before",rowcount)
#print("after",nonews)
#new_df.show()

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
 textclean(col("title")).alias("title")
,textclean(col("summary")).alias("summary"))
#df.show()
#df_countacat=df.groupBy("category").count()

#df_countacat.show()
#Tokenize data
tokenizer=Tokenizer(inputCol="title",outputCol="token_title")
df_tok = tokenizer.transform(df)
#.select('category','token_title','summary')

tokenizer=Tokenizer(inputCol="summary",outputCol="token_summary")
df = tokenizer.transform(df_tok).select('category','token_title','token_summary')
df_token = tokenizer.transform(df_tok).select('category','token_title','token_summary')
#countTokens = udf(lambda w:len(w),IntegerType())
#df.select("title","token_title").withColumn("tokens",countTokens(col("token_title"))).show()

#df.show(5)

# Stop Words Removal
remover = StopWordsRemover(inputCol="token_title",outputCol="title")
df_stop=remover.transform(df)

remover = StopWordsRemover(inputCol="token_summary",outputCol="summary")
df=remover.transform(df_stop).select('category','title','summary')

#df.show(truncate=False)
#Stemming is not using in this case because it is not very efficiat method
#lemmatization 
lemmatizer = WordNetLemmatizer()
df = df.select('category',lemmatizer.lemmatize('title'),lemmatizer.lemmatize('summary'))
#df.show(3)

# NGram
bigram = NGram(n=2,inputCol="title",outputCol="title_bi")
bigram_df=bigram.transform(df)

bigram = NGram(n=2,inputCol="summary",outputCol="summary_bi")
bigram_df=bigram.transform(bigram_df)

#bigram_df.show(10)

#Term Frequency - Inverse Doc Frequeny(TF-IDF)
hashingTF = HashingTF(inputCol="title",outputCol="title_tf",numFeatures=100)
df_tf = hashingTF.transform(bigram_df)

hashingTF = HashingTF(inputCol="summary",outputCol="summary_tf",numFeatures=100)
df_tf = hashingTF.transform(df_tf)

#df_tf.show(10,truncate=False)
idf = IDF(inputCol="title_tf",outputCol="title_idf")
idf_model=idf.fit(df_tf)

df_rescale = idf_model.transform(df_tf)

idf_summary = IDF(inputCol="summary_tf",outputCol="summary_idf")
idf_model_s=idf_summary.fit(df_rescale)
df_res = idf_model_s.transform(df_rescale)

#df_res.select("summary_idf","title_idf").show(5,truncate=False)

#CountVectorizer - Bag of Words
cv = CountVectorizer(inputCol="title",outputCol="title_cv")
model=cv.fit(df_res)
cv_title= model.transform(df_res)

cv_s = CountVectorizer(inputCol="summary",outputCol="summary_cv")
model=cv_s.fit(cv_title)
cv_title= model.transform(cv_title)

#cv_title.show(1,truncate=False)
#cv_title.printSchema()

#Label encoding of category 
le=StringIndexer(inputCol="category",outputCol="cat_label").fit(df_res)
le.transform(cv_title)
#model_le.select("category","cat_label").show(50)
print(le.labels)

#Final dataframe for model - taking category and summary 
# columns to train and test
#model_df=model_le.select("category","cat_label","summary_cv")
#model_df.show(5,truncate=False)
#train , test = model_le.randomSplit([0.8,0.2])

#Logistic
#log_reg = LogisticRegression(labelCol="cat_label",featuresCol="summary_cv",maxIter=50)
#log_reg.fit(train)
#model_reg=log_reg.fit(train)
#pred_reg=model_reg.transform(test)
#pred_reg.show(2)

#Naivebays
#nbays =NaiveBayes(labelCol="cat_label",featuresCol="summary_cv",smoothing=1.0)
#nbays.fit(train)
#model_nbays=nbays.fit(train)
#pred_nbays=model_nbays.transform(test)
#pred_nbays.show(2)
#RandomForestClassifier
#rfc = RandomForestClassifier(labelCol="cat_label",featuresCol="summary_cv",maxDepth=10)
#rfc.fit(train)
#model_rfc=rfc.fit(train)
#pred_rfc= model_rfc.transform(test)

#GBTClassifier



#Model Evaluation
#evaluator = MulticlassClassificationEvaluator(labelCol="cat_label",predictionCol="prediction",metricName="accuracy")
#accuracy_reg = evaluator.evaluate(pred_reg)
#print("Accuracy-reg : ",accuracy_reg)
#accuracy_nbays = evaluator.evaluate(pred_nbays)
#print("Accuracy-NaiveBayes : ",accuracy_nbays)
#accuracy_rfc = evaluator.evaluate(pred_rfc)
#print("Accuracy-RandomForest : ",accuracy_rfc)

#pickle.dump(rfc,open('mongospark.pkl','wb'))

#news.show(2)

#Loading DataFrame to DataBase
#cv_title.write.format("mongodb://localhost:27017").mode("append")
# .option("database","capstone").option("collection","procnews").save()
#bigram_df.write.format("mongo").mode("append").option("database","capstone").option("collection", "processeddf").save()

