# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from gensim.models import word2vec
import logging
import os
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
os.chdir("C:/CS512/data/KBP16/")

"""
######################################################
########## POS tagging for entities ##################
######################################################
"""
seed=open("seed_file.txt")
seed_lines=seed.readlines()
Entity=[]
Type=[]
for m in seed_lines:
    type_split=m.split()
    Entity.append(type_split[0])
    Type.append(type_split[1])
pos_tag=[nltk.pos_tag(nltk.word_tokenize(x))[0] for x in Entity]
pos_tag_df=pd.DataFrame.from_records(pos_tag)
pos_tag_df.columns=["Entity","POS"]
pos_tag_df["Type"]=Type
pos_list=list(pos_tag_df["POS"].unique())
for p in pos_list:
    sub_df=pos_tag_df[pos_tag_df["POS"]==p]
    print(sub_df.head(n=5))
    print("There are " ,sub_df.shape[0], " entities which are ",p)
    
######### REMOVE few POS ###############################

### IN (Preposition),DT( determiner ),PRP(Personal Pronoun) removed
pos_tag_df=pos_tag_df[~pos_tag_df["POS"].isin(["IN","DT","PRP$"])]
  
"""
############################################################
########### WORD2VEC training ##############################
############################################################
"""
# loading raw text
df=pd.read_csv("en/train_raw.txt",sep="\t",header=None,index_col=False)
sentences=df[1]
len(sentences)

def Content_process( sentence ):      
    letters_only = re.sub("[^a-zA-Z0-9]", " ", sentence) 
    words = letters_only.split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   
    return( " ".join( meaningful_words )) 

clean_sentences=[[]]
for i in range(len(sentences)):
    processed_content= Content_process(sentences[i]).split()
    #rec=np.array(processed_content.split(' '))
    clean_sentences.append(processed_content)
    if i%1000==0:
        print ("Processed",i,"sentences")

# training word to vec model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = word2vec.Word2Vec(clean_sentences, size=300)
model.save('300features')

"""
###############################################################
########### learning word embeddings for entities #############
###############################################################
"""
model=word2vec.Word2Vec.load('300features')
np_300=[]
Entity=np.array(pos_tag_df['Entity'])
Type=np.array(pos_tag_df['Type'])
for i in range(len(Entity)):
    e=Content_process(Entity[i])
    if e in model.wv.vocab:
        np1=np.array(model[e])
        np2=np.append(Entity[i],Type[i])
        arr1=np.append(np1,np2)
        np_300.append(arr1)
    
train_data=pd.DataFrame.from_records(np_300)  
pd.unique(train_data[301])  
train_data.to_csv("w2v.csv",header=False,index=False)


"""
##################################################################
############### clustering #######################################
##################################################################

REMOVING NOISE by the assumptions
1. Entities of same Type will fall in to a cluster based on word embeddings
2. Entities that belong to different type will fit into a single cluster based on METAPATH using word embeddings 
"""

"""
##################################################################
############### Auto encoder for noise detection #################
##################################################################
"""
import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
h2o.init(max_mem_size = 7)

wv_df=pd.read_csv("w2v.csv",header=None,index_col=False) 
ty=pd.unique(wv_df[301])
wv=h2o.import_file("w2v.csv")

anomaly_list=[]
for t in range(len(ty)):
    w=wv[wv['C302']==ty[t]]  
    print(t)
    predictors = w.col_names[0:300]
    train = w[predictors]
    dl_model = H2ODeepLearningEstimator(hidden=[400,200,20,200,400], epochs = 800,activation="Tanh",autoencoder=True)
    dl_model.train(x=w.col_names[0:300],training_frame= train)
    e=dl_model.anomaly(train,per_feature=False)
    w['errors']=e
    w2 = w.as_data_frame(use_pandas=True)
    anomaly=np.array(w2[w2['errors']>w2['errors'].quantile(q=0.99)]['C301'])
    anomaly_list.append(anomaly)

with open("Anomaly.txt","w") as wfile:
    for i in range(len(anomaly_list)):
        wfile.write(str(anomaly_list[i]),"\n")

#### t-sne plot for each type
#
#
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
#
#df=pos_tag_df[pos_tag_df["Type"]=="ORG"]
##df=pos_tag_df
#
#vocab =list(df['Entity'])
#vocabulary=[]
#for v in vocab:
#    e=Content_process(v)
#    if e in model.wv.vocab:
#        vocabulary.append(e)
#
#
#model=word2vec.Word2Vec.load('300features')
#wv=[]
#for i in vocabulary:
#    #e=Content_process(vocabulary[i])
#    #if e in model.wv.vocab:
#    np1=np.array(model[i])
#    wv.append(np1)
#tsne = TSNE(n_components=2, random_state=0)
#np.set_printoptions(suppress=True)
#Y = tsne.fit_transform(wv)
#plt.scatter(Y[:, 0], Y[:, 1])
#for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
#    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
#plt.show()
#
#
#### density based clustering
#import numpy as np
#
#from sklearn.cluster import DBSCAN
#from sklearn import metrics
#from sklearn.datasets.samples_generator import make_blobs
#from sklearn.preprocessing import StandardScaler
#
#X = StandardScaler().fit_transform(wv)
#db = DBSCAN(eps=0.2, min_samples=10).fit(X)
#core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#core_samples_mask[db.core_sample_indices_] = True
#labels = db.labels_
#
## Number of clusters in labels, ignoring noise if present.
#n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#import matplotlib.pyplot as plt
#
## Black removed and is used for noise instead.
#unique_labels = set(labels)
#colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
#for k, col in zip(unique_labels, colors):
#    if k == -1:
#        # Black used for noise.
#        col = 'k'
#
#    class_member_mask = (labels == k)
#
#    xy = X[class_member_mask & core_samples_mask]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=14)
#
#    xy = X[class_member_mask & ~core_samples_mask]
#    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=6)
#
#plt.title('Estimated number of clusters: %d' % n_clusters_)
#plt.show()
#
#
#
#import numpy as np
#import matplotlib.pyplot as plt
#from pyclustering.cluster.optics import optics
#
#optics_instance = optics(wv, 0.5, 6)
#optics_instance.process()
#clusters = optics_instance.get_clusters()
#noise = optics_instance.get_noise()
#
#ordering = optics_instance.get_cluster_ordering();
#indexes = [i for i in range(0, len(ordering))];
#plt.bar(indexes, ordering);
#plt.show(); 
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.neighbors import LocalOutlierFactor
#print(__doc__)
#
#np.random.seed(42)
#
## Generate train data
#X = 0.3 * np.random.randn(100, 2)
## Generate some abnormal novel observations
#X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
#X = np.r_[X + 2, X - 2, X_outliers]
#
## fit the model
#clf = LocalOutlierFactor(n_neighbors=20)
#y_pred = clf.fit_predict(X)
#y_pred_outliers = y_pred[200:]
#
## plot the level sets of the decision function
#xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
#Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)
#
#plt.title("Local Outlier Factor (LOF)")
#plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
#
#a = plt.scatter(X[:200, 0], X[:200, 1], c='white')
#b = plt.scatter(X[200:, 0], X[200:, 1], c='red')
#plt.axis('tight')
#plt.xlim((-5, 5))
#plt.ylim((-5, 5))
#plt.legend([a, b],
#           ["normal observations",
#            "abnormal observations"],
#           loc="upper left")
#plt.show()
#
### clustering into 5 groups
#from sklearn.cluster import KMeans
#word_vectors = model.wv.syn0
#num_clusters =  5
## Initalize a k-means object and use it to extract centroids
#kmeans_clustering = KMeans( n_clusters = num_clusters )
#idx = kmeans_clustering.fit_predict( word_vectors )
#word_centroid_map = dict(zip( model.wv.index2word, idx )) 
#
#for cluster in range(num_clusters):
#    #
#    # Print the cluster number  
#    print ("\nCluster %d" % cluster)
#    #
#    # Find all of the words for that cluster number, and print them out
#    words = []
#    for i in range(0,len(word_centroid_map.values())):
#        if( list(word_centroid_map.values())[i] == cluster ):
#            words.append(list(word_centroid_map.keys())[i])
#    print (words)
#
#
