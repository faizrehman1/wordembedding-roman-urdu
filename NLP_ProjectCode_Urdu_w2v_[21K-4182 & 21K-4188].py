#!/usr/bin/env python
# coding: utf-8

# # **NLP Final Project**
# ### **Roman Urdu Word embeddings using word2vec**
# 
# **Muhammd Owais Alam Ansari - [21K-4182]**
# <br>
# **Faiz ur Rehman Khan - [21K-4188]**

# In[ ]:





# ### **Importing the required packages**

# In[143]:


import numpy as np
import re 
import pandas as pd
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
nltk.download('punkt')
import multiprocessing
from time import time
import os.path
import pickle
from gensim.test.utils import datapath
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
 
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd  
from gensim.models.callbacks import CallbackAny2Vec


# 

# In[144]:


from google.colab import drive
drive.mount('/content/drive')


# ### **Loading the dataset**

# In[145]:


df = pd.read_csv('/content/drive/MyDrive/NLPProject/chat_data.csv')


# ##**Exploratory Data Analysis**
# **Preparing the text to train the model**

# ### **Shape of data**

# In[146]:


df.shape


# ### **Have a look a data**

# 

# In[147]:


df.head()


# In[148]:


df.info()


# **Removing irrelevant column**

# In[149]:


df = df.drop(['msgType'], axis=1)
df.head(5)


# ### **Dataset contains some numbers and non-english characters. Therefore, Remove all non-english characters and make everything lowercase.**

# In[150]:


df['cleanMsg'] = df['message'].str.replace('[^a-zA-Z]',' ').str.lower()


# In[151]:


df.head()


# In[152]:


df[0:25]


# In[153]:


df['cleanMsg'].replace(' ', np.nan, inplace=True)


# In[154]:


df.isnull().sum()


# ###**Remove Null values**

# In[155]:


df = df.dropna(subset=['cleanMsg'])


# In[156]:


df[0:25]


# In[157]:


df.isnull().sum()


# 

# In[157]:





# ### **Removing the sentence with only single words because Word2Vec uses context words to learn the vector representation of a target word.**

# In[158]:


df = df[df["cleanMsg"].apply(lambda x: len(x.split()) > 1)]
# df = df[df["cleanMsg"].apply(lambda x: len(x.split()) > 2)]


# In[159]:


df.info()


# In[160]:


df[0:25]


# ### **Dropping duplicate rows**
# 

# In[161]:


df['cleanMsg'].is_unique


# In[162]:


duplicate_rows_df = df[df.duplicated()]
print("Number of duplicate rows: ", duplicate_rows_df.shape)


# In[163]:


df = df.drop_duplicates(subset=['cleanMsg'], keep='first')


# In[164]:


df['cleanMsg'].is_unique


# In[165]:


df.head()


# In[166]:


df.info()


# ### **Data Visualization**

# In[167]:


word_cloud = WordCloud(colormap='Blues',width=1000,height=600).generate(str(df["cleanMsg"]))
fig = plt.figure(1, figsize=(10,10))
plt.axis('off')
plt.title("WordCloud for most used words", size = 24)
fig.subplots_adjust(top=2.3)
plt.imshow(word_cloud)
plt.show()


# In[168]:


plt.style.use('ggplot')
plt.figure(figsize=(14,6))
freq=pd.Series(" ".join(df["cleanMsg"]).split()).value_counts()[:30]
freq.plot(kind="bar", color = "teal")
plt.title("30 most frequent words",size=20)


# ## **Model Training**

# **Creating array list of text from data**

# In[169]:


dataset = [i for i in df['cleanMsg']]
print (dataset[0:20])


# #### **Tokenization of roman text**

# In[170]:


roman_lines = list()
t = time()
if os.path.exists('/content/drive/MyDrive/NLPProject/romanTokTT.ob'):
   with open ('/content/drive/MyDrive/NLPProject/romanTokTT.ob', 'rb') as fp:
    roman_lines = pickle.load(fp)
    print('Roman tokens loaded from .ob file')
else:
  for line in dataset:
      # tokenize the text
      tokens = word_tokenize(line)
      tokens = [w.lower() for w in tokens]
      # remove puntuations
      table = str.maketrans('', '', string.punctuation)
      stripped = [w.translate(table) for w in tokens]
      # remove non alphabetic characters
      words = [word for word in stripped if word.isalpha()]
      roman_lines.append(words)

  with open('/content/drive/MyDrive/NLPProject/romanTokTT.ob', 'wb') as fp:
    pickle.dump(roman_lines, fp)
    
  print('Roman tokens created from file')
    
print('Time to load: {} mins'.format(round((time() - t) / 60, 2)))
print(roman_lines[0:5])


# In[171]:


# Callback to print loss after each epoch.

class callback(CallbackAny2Vec):

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_begin(self, model):
        print("Epoch {} start.".format(self.epoch))

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        print("Epoch {} end.".format(self.epoch))
        print("") 
        self.epoch += 1


# ### **Train word2vec model on roman urdu text**

# In[172]:


t = time()
if os.path.exists('/content/drive/MyDrive/NLPProject/my_model500eTTf.bin'):
  w2v_model = Word2Vec.load('/content/drive/MyDrive/NLPProject/my_model500eTTf.bin')
  print('Time to load the model: {} mins'.format(round((time() - t) / 60, 2)))
  
else:
  # w2v_model = Word2Vec(sentences=roman_lines, size=100, window=5, workers=4,
  #                      min_count=2, sg=0, iter=50, compute_loss=True, callbacks=[callback()]) #sg= 1:skip-gram 0:cbow
  

  # init word2vec
  w2v_model = Word2Vec(size=100, window=5, workers=4, min_count=2, sg=0)                   #sg= 1:skip-gram 0:cbow

  # build vocab
  w2v_model.build_vocab(roman_lines, progress_per=10000)

  # train the w2v model
  w2v_model.train(roman_lines, total_examples=w2v_model.corpus_count, epochs=500,
                  report_delay=1, compute_loss=True, callbacks=[callback()])

  temp_file = datapath("/content/drive/MyDrive/NLPProject/my_model500eTTf.bin")
  w2v_model.save(temp_file)
  print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

print(w2v_model)


# **Using the vectors**

# In[173]:


w2v_model['theek']


# In[174]:


w2v_model['kha']


# ### **Testing the model**

# In[175]:


w2v_model.wv.most_similar("theek")


# In[176]:


w2v_model.wv.most_similar("salam")


# In[177]:


w2v_model.wv.most_similar("nhi")


# In[178]:


w2v_model.wv.most_similar("zyada")


# In[179]:


w2v_model.wv.most_similar("bhai")


# In[180]:


w2v_model.wv.most_similar("sai")


# **Cosine Similarity**

# In[181]:


from numpy import dot
from numpy.linalg import norm

def cosine_similarity (model,word1,word2):
  cos_sim = dot(model.wv[word1], model.wv[word2])/(norm(model.wv[word1])*norm(model.wv[word2]))
  return cos_sim


# In[182]:


cosine_similarity(w2v_model,'kaise', 'kese')


# In[183]:


w2v_model.wv.similarity('kaise', 'kese')


# In[184]:


w2v_model.wv.similarity('nhi', 'nahi')


# In[184]:





# ## **Visualization**

# In[185]:


def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 100), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 100 to 19 dimensions with PCA
    reduc = PCA(n_components=19).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))


# In[186]:


tsnescatterplot(w2v_model, 'nhi', [i[0] for i in w2v_model.wv.most_similar(positive=["salam"])])


# In[140]:





# In[ ]:




