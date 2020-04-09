#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

#Create a Jupyter Notebook with the code provided that project, and explore all the existing functionalities 
#(you can also go beyond what is provided in the project and explore other functionalities, 
# particularly related to different data visualizations)


# In[2]:


#import dataset

diabete = pd.read_csv('diabetes2.csv')

diabete.head()


# In[3]:


#quick descriptive stats

diabete.describe()


# In[4]:


#SEABORN distrib. histogram  dropna-Remove missing values.

sns.distplot(diabete['Age'].dropna(),kde=True)


# In[5]:


#visualizing data in MATPLOTLIB scatter plot

fig, ax = plt.subplots()
ax.scatter(diabete['Age'], diabete['BMI'])
ax.set_xlabel('Age')
ax.set_ylabel('Body Mass Index')


# In[6]:


#cute easier SEABORN scatterplot, hue parameter distinguishing the diabetic ones (outcome=1)


sns.scatterplot(x='Age', y='BMI', hue='Outcome', data=diabete)


# In[7]:


#cute rainbow boxplot showing chosen feature (BMI- descriptive stats) for each age

plt.subplots(figsize=(20,15))
sns.boxplot(x='Age', y='BMI', data=diabete)


# In[8]:


#simple countplot Outcome is either 0 or 1 (would be Y target)
sns.countplot(x='Outcome',data=diabete)


# In[9]:


# Compute pairwise correlation of columns, excluding NA/null values.
diabete.corr()


# In[27]:


#SEABORN heatmaps are easier

sns.heatmap(diabete.corr(), annot=True, cmap='Blues')


# In[11]:


#another try at correlation heatmaps
# hotred is positive, coolblue is negative The stronger the color, the larger the correlation magnitude. 

fig, ax = plt.subplots(figsize=(10,10))  
ax = sns.heatmap(
    diabete.corr(), 
    vmin=-1, vmax=1, center=0,
    cmap='coolwarm',
    annot=True
)


# In[12]:


#pairplots (or scatter_matrix in PANDAS) may take a bit to load (10sec)

sns.pairplot(diabete, hue = 'Outcome')


# In[13]:


#spliting dataset



x = diabete.drop('Outcome',axis=1)
y = diabete['Outcome']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


# In[14]:


#perform logistic regression

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter=500)

logmodel.fit(x_train,y_train)


# In[15]:


#using model to make prediction

predictions = logmodel.predict(x_test)


# In[29]:


#display confusion matrix in a heatmap!, FP and FN should be as low as possible

from sklearn.metrics import confusion_matrix
confm = confusion_matrix(y_test,predictions)

# heatmap
sns.heatmap(pd.DataFrame(confm), annot=True, cmap='binary', fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[30]:



#array style

confusion_matrix(y_test,predictions)


# In[31]:


#classification reports and score

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[ ]:





# In[ ]:




