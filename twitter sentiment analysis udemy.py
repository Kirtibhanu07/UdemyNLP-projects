#!/usr/bin/env python
# coding: utf-8

# ## Twitter sentiment analysis
Access token:
1100455510262902790-IimSHnMnkip2J9dGmwTooSEkets3dY

Access token secret:
AjISLjaGmsJWMvtakstaVGWUHrV6xdkU1QblNJnG5JYbg

API key:
pn3ZLO47TA6gJlQkekgN7ZBv5

API key secret:
8gSddSCIEIW26pxk6uF2l1r2LOxgnYtYT4XIvBtiQ4brNv93WT

# In[1]:



get_ipython().system('pip install textblob ')


# In[2]:


from textblob import TextBlob


# In[3]:


get_ipython().system('pip install tweepy')


# In[4]:


import tweepy


# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[6]:


#api key 
consumer_key="pn3ZLO47TA6gJlQkekgN7ZBv5"
#api secret key
consumer_secret="8gSddSCIEIW26pxk6uF2l1r2LOxgnYtYT4XIvBtiQ4brNv93WT"

access_token="1100455510262902790-IimSHnMnkip2J9dGmwTooSEkets3dY"

access_token_secret="AjISLjaGmsJWMvtakstaVGWUHrV6xdkU1QblNJnG5JYbg"


# In[39]:


auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


# In[8]:


type(api)


# In[9]:


search_term = "Money Heist"
no_of_tweets=1000


# In[40]:


tweets=tweepy.Cursor(api.search, q=search_term,).items(no_of_tweets)


# In[11]:


tweets


# In[12]:


a='I am a good cricket player'
b='I am a bad cricket player'
c='I am a cricket player'


# In[13]:


TextBlob(a).sentiment


# In[14]:


TextBlob(b).sentiment


# In[15]:


TextBlob(c).sentiment


# In[41]:


positive=0
negative=0
polarity=0.0
neutral=0


# In[43]:


for tweet in tweets:
    analysis=TextBlob(tweet.text)
    polarity += analysis.sentiment.polarity
    
    if(analysis.sentiment.polarity==0.00):
        neutral+=1
    elif(analysis.sentiment.polarity<0.00):
        negative+=1
    elif(analysis.sentiment.polarity>0.00):
        positive+=1


# In[44]:


positive


# In[45]:


negative


# In[46]:


neutral


# In[47]:


polarity


# In[48]:


def percentage(part,whole):
    return 100*float(part)/float(whole)


# In[49]:


positive=percentage(positive,no_of_tweets)
negative=percentage(negative,no_of_tweets)
neutral=percentage(neutral,no_of_tweets)
polarity=percentage(polarity,no_of_tweets)


# In[50]:


positive=format(positive,'.2f')
negative=format(negative,'.2f')
neutral=format(neutral,'.2f')


# In[51]:


positive


# In[52]:


negative


# In[53]:


neutral


# In[54]:


polarity


# In[ ]:




