import re
# Analysis
import pandas as pd
import numpy as np
from keras.activations import tanh
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB

from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# print('here')
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix
from time import time


fake_newss = pd.read_csv("DS/Zohaib/might work/CM-Spam/BC/600/BuzzFeed Real 600.csv",low_memory=False)
true_newss = pd.read_csv("DS/Zohaib/might work/CM-Spam/BC/600/BuzzFeed Real vs others.csv",low_memory=False)

fake_newss['label']=0
true_newss['label']=1
df=pd.concat([fake_newss,true_newss],ignore_index=True)
df = df.sample(frac = 1,random_state=24).reset_index(drop=True)
# print(df.head())
df.duplicated().sum()

df.drop_duplicates(inplace=True)

df['txt']=df['title']
# df['txt']=df['title']+' '+df['text']
# print(df['txt'])
# df.drop(columns=['title','text','subject','date'],inplace=True)
# df=df.sample(20000).reset_index(drop='index')

tl=WordNetLemmatizer()
corpus=[]
for i in np.arange(len(df)):
    line =re.sub(r'[^a-zA-Z]',' ',str(df.iloc[i]['txt']))
    line=line.lower()
    line=line.split()
    line=[tl.lemmatize(word) for word in line if word not in stopwords.words('english')]
    line=" ".join(line)
    corpus.append(line)
# c=1
# for i in range(len(df)):
#     line = " ".join(df.iloc[i]['txt'])
#     corpus.append(line)
#     corpus.append(df.iloc[i]['txt'])
    # print(df.iloc[i]['txt'])

x=np.array(corpus)
y=df['label']

import warnings
warnings.filterwarnings("ignore")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=4, shuffle =True)
# x_train, x_test, y_train, y_test = train_test_split(np.array(corpus), df['label'], test_size=0.20, random_state=4, shuffle =True)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

my_tfidf = TfidfVectorizer()
print('here')
# fit the vectorizer and transform X_train into a tf-idf matrix,
# then use the same vectorizer to transform X_test
x_train = my_tfidf.fit_transform(x_train).toarray()
x_test = my_tfidf.transform(x_test).toarray()

train_scores = []
test_scores = []


def algorithm(model):
    # Training model
    model.fit(x_train, y_train)

    # score of train set
    train_model_score = model.score(x_train, y_train)
    train_scores.append(round(train_model_score, 2))
    y_pred_train = model.predict(x_train)

    # score of test set
    test_model_score = model.score(x_test, y_test)
    test_scores.append(round(test_model_score, 2))
    y_pred_test = model.predict(x_test)

    # Printing results
    # print("Train score :", round(train_model_score, 2))
    print("Test score :", round(test_model_score, 2))

    df_model = pd.DataFrame(classification_report(y_pred_test, y_test, digits=2, output_dict=True)).T
    df_model['support'] = df_model.support.apply(int)
    df_model.style.background_gradient(cmap='viridis', subset=pd.IndexSlice['0':'9', :'f1-score'])
    # print(df_model)

    # print("\n----------------------Confusion Matrix---------------------- \n")
    # conf_mat = confusion_matrix(y_test, y_pred_test)
    # # plot_confusion_matrix(conf_mat,
    # # show_normed=True, colorbar=True,
    # # class_names=['Fake', 'Real'])
    # plt.show()
totalStart=time()
t1=time()
print('BNB')
from sklearn.naive_bayes import BernoulliNB
bnv=BernoulliNB()
algorithm(bnv)
t2=time()
print('Time is : ',t2-t1)
#
#
t1=time()
print('GNB')
gnv=GaussianNB()
algorithm(gnv)
t2=time()
print('Time is : ',t2-t1)
#
t1=time()
print('DT')
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
algorithm(dt)
t2=time()
print('Time is : ',t2-t1)
#
t1=time()
print('RF')
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
algorithm(rf)
t2=time()
print('Time is : ',t2-t1)
#
t1=time()
print('MLP')
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=600,activation='tanh',learning_rate='invscaling')
algorithm(mlp)
t2=time()
print('Time is : ',t2-t1)
#
t1=time()
print('SVM')
SVM=svm.SVC()
algorithm(SVM)
t2=time()
print('Time is : ',t2-t1)

t1=time()
print('KNN Centroid')
from sklearn.neighbors import NearestCentroid
clf = NearestCentroid()
algorithm(clf)
t2=time()
print('Time is : ',t2-t1)
#
t1=time()
print('LR')
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
algorithm(LR)
t2=time()
print('Time is : ',t2-t1)

totalEnd=time()
print('Totla Time is : ',totalEnd-totalStart)
# # print('KNN')
# # from sklearn.neighbors import NearestNeighbors
# # KNN=NearestNeighbors(n_neighbors=2)
# # algorithm(KNN)
# #
# # from tensorflow.keras.layers import LSTM
# # ls=LSTM(activation='relu',return_sequences=True)
# # algorithm(ls)
