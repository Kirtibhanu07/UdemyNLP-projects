#!/usr/bin/env python
# coding: utf-8

# ## Restaurant Review classifier 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


data=pd.read_csv("Desktop/Restaurant_Reviews.tsv",sep='\t',quoting=3)


# In[5]:


data.head()


# In[6]:


data['Liked'].value_counts()


# In[7]:


data.isnull().sum()


# In[8]:


import nltk


# In[9]:


import re


# In[10]:


nltk.download('stopwords')


# In[11]:


from nltk.corpus import stopwords


# In[12]:


review=re.sub('[^a-zA-z]', ' ',data['Review'][0])
review


# In[13]:


review=review.lower()


# In[14]:


review=review.split()


# In[15]:


review


# In[ ]:





# In[16]:


stopwords.words('english')


# In[17]:


preview=[]
for word in review:
    if word not in stopwords.words('english'):
        preview.append(word)
        
       


# In[18]:


preview


# In[19]:


review=[word for word in review if word not in stopwords.words('english')]


# In[20]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[21]:


review= [ps.stem(word) for word in review]


# In[22]:


review


# In[23]:


review=" ".join(review)


# In[24]:


review


# In[45]:


corpus=[]

ps=PorterStemmer()

for i in range(len(data)):
    
    review=re.sub('[^a-zA-z]', ' ',data['Review'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in stopwords.words('english')]
    review= " ".join(review)
    corpus.append(review)    


# In[ ]:





# In[46]:


print(corpus)


# In[47]:


from sklearn.feature_extraction.text import CountVectorizer


# In[48]:


cv=CountVectorizer(max_features=1500)


# In[49]:


x=cv.fit_transform(corpus).toarray()


# In[51]:


print(x)


# In[53]:


x.shape


# In[54]:


y=data.iloc[:, 1].values


# In[56]:


y.shape


# In[57]:


y


# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0,shuffle=True)


# In[60]:


X_train.shape


# In[61]:


X_test.shape


# In[62]:


y_train.shape


# In[63]:


y_test.shape


# In[64]:


from sklearn.naive_bayes import GaussianNB


# In[65]:


classifier=GaussianNB()


# In[66]:


classifier.fit(X_train,y_train)


# In[68]:


y_pred=classifier.predict(X_test)


# In[69]:


y_pred


# In[70]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[71]:


accuracy_score(y_test,y_pred)


# In[72]:


confusion_matrix(y_test,y_pred)


# In[ ]:




