#!/usr/bin/env python
# coding: utf-8

# ### Load the dataset, and establish subsets
# 
# ing - is a dataframe with just the ingredient columns\
# rate - is a dataframe with one column (rating)
# 
# 
# We removed the columns 'ref', 'Unnamed:0' and 'beans' (all have_beans) because they were irrelevant or redundant.
# 

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# load the chocodataset, dropping useless columns
df = pd.read_csv('chocolate.csv').drop(['ref', 'Unnamed: 0', 'beans'], axis=1)

#creating a separate indexed dataframe for just the ingredients
ing = df.iloc[:,8:14]  
#sempre 6 colunas. location subject to changes in indexing order


#separate indexed column with the rating
rate = df.iloc[:,6] #subject to changes

#dataframe with all 4 tastes
tastes = df.iloc[:,14:18]

tastes


# In[2]:




from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, Tokenizer, RegexTokenizer, StopWordsRemover
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, min, max
from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, min, max
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import CountVectorizer
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vector, Vectors
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
from pyspark import SparkContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext

sparky = SparkSession.builder.appName('Choco_Data_Analysis').getOrCreate()  


#If the csv file have a header (column names in the first row) then set header=true. This will use the first row in the csv file as the dataframe's column names. Setting 
sp = (sparky.read
.option("inferSchema","true")
.option("header","true")
.csv('chocolate.csv'))


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


sqlContext = SQLContext(sparky)


#predicting the company location from the ratings, percentage and count of ingredients of the chocolate
FEATURES_COL = ['cocoa_percent']

#selecting from the "main" dataset for this query
sp2 = sp.select('rating', 'cocoa_percent')
df = sp2.toDF('rating','cocoa_percent')


#we need all the features/vars to be of the same type while still keeping 'company_location' as a string
for col in df.columns:
    if col in FEATURES_COL:
        df = df.withColumn(col,df[col].cast('double'))
df.printSchema()


# In[21]:


df.na.drop()

df.show()


# In[26]:


from pyspark.ml.evaluation import ClusteringEvaluator

vecAssembler = VectorAssembler(inputCols=FEATURES_COL, outputCol="percentage")
df_kmeans = vecAssembler.transform(df).select('rating', 'percentage')

k = 10
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("percentage")
model = kmeans.fit(df_kmeans)
centers = model.clusterCenters()

print("Cluster Centers: ")
for center in centers:
    print(center)
    


# In[23]:


transformed = model.transform(df_kmeans).select('rating', 'features')
rows = transformed.collect()
print(rows[:2])


# In[24]:


df_pred = sqlContext.createDataFrame(rows)
df_pred.show()


# In[ ]:


df_pred = df_pred.join(df, 'company_location')
df_pred.show()


# In[ ]:


pddf_pred = df_pred.toPandas().set_index('company_location')
pddf_pred.head()

threedee = plt.figure(figsize=(12,10)).gca(projection='3d')
threedee.scatter(pddf_pred.cocoa_percent, pddf_pred.rating, pddf_pred.counts_of_ingredients, c=pddf_pred.features)
threedee.set_xlabel('x')
threedee.set_ylabel('y')
threedee.set_zlabel('z')
plt.show()


# ### The following 2 cells only run once, because it transforms/replaces the entire dataframe from string to integers. Avoid rerunning the next 2 cells, unless you want to restart the kernel/ ou correr a 1 cell tb

# In[ ]:





# In[ ]:


#created a function that transforms every dataframe that has a similar style to have_/have_not into ones and zeroes

def ingredients_transformed(x):
    
   transformed = x.replace({"have_not*" :0, "have*": 1}, regex=True, inplace=True)
   return transformed


# In[ ]:


#only runs once, unless you restart and rerun all the outputs, from the untransformed ing dataframe.
ingredients_transformed(ing)

#update the original dataset, with transformed ingredient columns, it works
df.update(ing)


# # Exploratory Analyses

# In[ ]:


# Understanding the basic ground information of our chocodata

def all_about_my_data(df):
    print("Here is some Basic Ground Info about your Data:\n")
    
    # Shape of the chocodataframe
    print("Number of Instances:",df.shape[0])
    print("Number of Features:",df.shape[1])
    
    # Summary Stats
    print("\nSummary Stats:")
    print(df.describe())
    
    # Missing Value Inspection
    print("\nMissing Values:")
    print(df.isna().sum())

all_about_my_data(df)



# # Who creates the best Chocolate bars?

# ## What makes a good chocolate?
# 
# We tried to correlate the usage of all ingredients against each other, as represented in the heatmap.
# For example, salt+sugar, and sugar+sweetener without sugar are the least common combination of ingredients, strongly negatively correlated. Salt+sweetener without sugar, lecithin+vanilla/cocoa_butter, and cocoa butter+vanilla are some of the most common combos.
# 
# Finally, sugar and cocoa butter are certain amongst the better rated chocolates.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

#joining rating column with the ingredients dataframe
corr = ing.join(rate)
correlation_mat = corr.corr()
sns.heatmap(correlation_mat, annot = True, cmap="BrBG")
plt.show()


# # Exploring with pyspark. FICA PRO FIM NAO HA PACIENCIA

# import pyspark
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import mean, min, max
# 
# sparky = SparkSession.builder.appName('Choco_Data_Analysis').getOrCreate()  
# 
# spark.conf.set("spark.sql.execution.arrow.enabled", "true")
# 
# spark_df = sqlContext.createDataFrame(df)
# 
# 
# sparky = SparkSession.builder.appName('Data_Analysis').getOrCreate()  
# 
# 
# #If the csv file have a header (column names in the first row) then set header=true. This will use the first row in the csv file as the dataframe's column names. Setting
# sp = (sparky
#   .read       
#   .option("inferSchema","true")                 
#   .option("header","true")                           
#   .csv("chocolate.csv")
#   .drop('ref'))
# 
# 
# sp.select('rating').describe().show()
# 
# 
# from pyspark.sql.functions import *
# 
# 
# df.select(('rating'), df.rating.cast('float'))

# In[ ]:


# describing portuguese companies, rating, and where do they get their beans

sp.select('rating', 'company_location', 'company', 'country_of_bean_origin').filter(df.company_location == 'Portugal').show(10)



#sort by top rating

df = df.select('rating', 'country_of_bean_origin').orderBy('rating', 'country_of_bean_origin', ascending=False).show()


# 
# 

# # Plotting with pandas
# 
# # Where are the best cocoa beans grown?
# 
# The best beans which create the highest rated chocolates are grown in Sao Tome e Principe, Solomon Islands, and Congo.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

#necessary to make the later plot show in descending order, and only show the top 10 highest chocos
grp_order = df.groupby('country_of_bean_origin').rating.agg('mean').sort_values(ascending=False).iloc[:10].index


# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(7, 4))
sns.set_color_codes('pastel')
sns.barplot(x='rating', y='country_of_bean_origin', data=df,
            label="Best cocoa beans", color="b",estimator=np.mean, order=grp_order)

# Add a legend and informative axis and limit
ax.set(xlim=(1, 4), ylabel="Bean origin",
       xlabel="Bean rating")


# # Which countries produce the highest-rated bars?

# In[ ]:


#necessary to make the later plot show in descending order, and only show the top 10 highest chocos
grp_order = df.groupby('company_location').rating.agg('mean').sort_values(ascending=False).iloc[:10].index


# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(6, 4))
sns.set_color_codes('pastel')
sns.barplot(x='rating', y='company_location', data=df,
            label="Best cocoa beans", color="g",estimator=np.mean, order=grp_order)

# Add a legend and informative axis and limit
ax.set(xlim=(0, 4), ylabel="Production country",
       xlabel="Rating")


# # Which company has the highest rating?

# In[ ]:


#necessary to make the later plot show in descending order, and only show the top 10 highest chocos
grp_order = df.groupby('company').rating.agg('mean').sort_values(ascending=False).iloc[:5].index


# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(6, 2))
sns.set_color_codes('pastel')
sns.barplot(x='rating', y='company', data=df,
            label="Best cocoa beans", color="r",estimator=np.mean, order=grp_order)

# Add a legend and informative axis and limit
ax.set(xlim=(3, 4), ylabel="Production Company",
       xlabel="Rating")


# # Spark and flavors

# In[ ]:


from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, min, max

sparky = SparkSession.builder.appName('Choco_Data_Analysis').getOrCreate()  



#If the csv file have a header (column names in the first row) then set header=true. This will use the first row in the csv file as the dataframe's column names. Setting 
sp = (sparky.read
.option("inferSchema","true")
.option("header","true")
.load(tastes))


# In[ ]:




first_ts = df[['rating','first_taste']]
first_ts.dtypes


# In[ ]:


# Import Spark NLP
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("Spark NLP").config("spark.driver.memory","8G").config("spark.driver.maxResultSize", "2G") .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5").config("spark.kryoserializer.buffer.max", "1000M").getOrCreate()


# if you are reading file from hdfs
file_location = 'chocolate.csv'
file_type = "csv"# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

spark_df = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)
# Verify the count
spark_df.count()


# In[ ]:


# Spark NLP requires the input dataframe or column to be converted to document. 

document_assembler = DocumentAssembler()     .setInputCol("first_taste")     .setOutputCol("document")     .setCleanupMode("shrink")

# Split sentence to tokens(array)
tokenizer = Tokenizer()   .setInputCols(["document"])   .setOutputCol("token")

# clean unwanted characters and garbage
normalizer = Normalizer()     .setInputCols(["token"])     .setOutputCol("normalized")

# remove stopwords
stopwords_cleaner = StopWordsCleaner()      .setInputCols("normalized")      .setOutputCol("cleanTokens")      .setCaseSensitive(False)

# stem the words to bring them to the root form.
stemmer = Stemmer()     .setInputCols(["cleanTokens"])     .setOutputCol("stem")

# Finisher is the most important annotator. Spark NLP adds its own structure when we convert each row in the dataframe to document. Finisher helps us to bring back the expected structure viz. array of tokens.
finisher = Finisher() .setInputCols(["stem"]) .setOutputCols(["tokens"]) .setOutputAsArray(True) .setCleanAnnotations(False)# We build a ml pipeline so that each phase can be executed in sequence. This pipeline can also be used to test the model. 
nlp_pipeline = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            normalizer,
            stopwords_cleaner, 
            stemmer, 
            finisher])# train the pipeline
nlp_model = nlp_pipeline.fit(spark_df)# apply the pipeline to transform dataframe.
processed_df  = nlp_model.transform(spark_df)

# nlp pipeline create intermediary columns that we dont need. So lets select the columns that we need.
tokens_df = processed_df.select('publish_date','tokens').limit(10000)
tokens_df.show()


# In[ ]:


from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import CountVectorizer
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vector, Vectors

path = tastes

data = tastes.zipWithIndex().map(lambda (words,idd): Row(idd= idd, words = words.split(" ")))
docDF = spark.createDataFrame(data)
Vector = CountVectorizer(inputCol="words", outputCol="vectors")
model = Vector.fit(docDF)
result = model.transform(docDF)

corpus = result.select("idd", "vectors").rdd.map(lambda (x,y): [x,Vectors.fromML(y)]).cache()

# Cluster the documents into three topics using LDA
ldaModel = LDA.train(corpus, k=3,maxIterations=100,optimizer='online')
topics = ldaModel.topicsMatrix()
vocabArray = model.vocabulary

wordNumbers = 10  # number of words per topic
topicIndices = sc.parallelize(ldaModel.describeTopics(maxTermsPerTopic = wordNumbers))

def topic_render(topic):  # specify vector id of words to actual words
    terms = topic[0]
    result = []
    for i in range(wordNumbers):
        term = vocabArray[terms[i]]
        result.append(term)
    return result

topics_final = topicIndices.map(lambda topic: topic_render(topic)).collect()

for topic in range(len(topics_final)):
    print ("Topic" + str(topic) + ":")
    for term in topics_final[topic]:
        print (term)
    print ('\n')


# In[ ]:




import pyspark
import string
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, Tokenizer, RegexTokenizer, StopWordsRemover

sc = pyspark.SparkContext(appName = "LDA_app")

#Function to load lines in a CSV file, and remove some special characters
def parseLine(line):
    line = line.encode('ascii',errors='ignore')
    line_split = line.replace('"','').replace('.','')    .replace('(','').replace(')','').replace('!','').split(';')
    return line_split
    
sqlContext = SQLContext(sc)

#load dataset, a local CSV file, and load this as a SparkSQL dataframe without external csv libraries. 


sqlContext = SQLContext(sc)

data_set = tastes

labels = data_set.first().replace('"','').split(';')

#create a schema
fields = [StructField(field_name, StringType(), True) for field_name in labels]
schema = StructType(fields)

#get everything but the header:
header = data_set.take(1)[0]
data_set = data_set.filter(lambda line: line != header)

#parse dataset
data_set = data_set.map(parseLine)

#create dataframe
data_df = sqlContext.createDataFrame(data_set, schema)    


#Tokenize the text in the text column
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsDataFrame = tokenizer.transform(data_df)


#remove 20 most occuring documents, documents with non numeric characters, and documents with <= 3 characters
cv_tmp = CountVectorizer(inputCol="words", outputCol="tmp_vectors")
cv_tmp_model = cv_tmp.fit(wordsDataFrame)


top20 = list(cv_tmp_model.vocabulary[0:20])
more_then_3_charachters = [word for word in cv_tmp_model.vocabulary if len(word) <= 3]
contains_digits = [word for word in cv_tmp_model.vocabulary if any(char.isdigit() for char in word)]

stopwords = []  #Add additional stopwords in this list

#Combine the three stopwords
stopwords = stopwords + top20 + more_then_3_charachters + contains_digits

#Remove stopwords from the tokenized list
remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords = stopwords)
wordsDataFrame = remover.transform(wordsDataFrame)

#Create a new CountVectorizer model without the stopwords
cv = CountVectorizer(inputCol="filtered", outputCol="vectors")
cvmodel = cv.fit(wordsDataFrame)
df_vect = cvmodel.transform(wordsDataFrame)

#transform the dataframe to a format that can be used as input for LDA.train. LDA train expects a RDD with lists,
#where the list consists of a uid and (sparse) Vector
def parseVectors(line):
    return [int(line[2]), line[0]]


sparsevector = df_vect.select('vectors', 'text', 'id').map(parseVectors)

#Train the LDA model
model = LDA.train(sparsevector, k=5, seed=1)

#Print the topics in the model
topics = model.describeTopics(maxTermsPerTopic = 15)
for x, topic in enumerate(topics):
    print ('topic nr: ' + str(x))
    words = topic[0]
    weights = topic[1]
    for n in range(len(words)):
        print (cvmodel.vocabulary[words[n]] + ' ' + str(weights[n]))


# In[ ]:




