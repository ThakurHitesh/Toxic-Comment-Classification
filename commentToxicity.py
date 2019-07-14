#Python program to predict the toxicity of the comment
#Author - Hitesh Thakur

#Importing required libraries
import re;
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from nltk.tokenize import word_tokenize;
from nltk.corpus import stopwords;
from nltk.stem import PorterStemmer;
from sklearn.model_selection import train_test_split;
from sklearn.feature_extraction.text import TfidfVectorizer;
from sklearn.metrics import roc_auc_score,accuracy_score;

#Reading data from .csv file using pandas
comment_data=pd.read_csv("train.csv");
raw_X=comment_data['comment_text'];         #Feature
Y=comment_data.iloc[:,2:8];                 #Targets

#Performing stemming and removing stop words
def wordToken(text):
    text=text.lower();
    wordset=word_tokenize(text);
    stop_words=set(stopwords.words('english'));
    filtered_words=[];
    stemming=PorterStemmer();
    for x in wordset:
        if x not in stop_words:
            filtered_words.append(stemming.stem(x));
    text=" ".join(filtered_words);
    return text;

#Cleaning data
def filtering(text):
    text = text.lower();
    text = re.sub(r"what's", "what is ", text);
    text = re.sub(r"\'s", " ", text);
    text = re.sub(r"\'ve", " have ", text);
    text = re.sub(r"can't", "cannot ", text);
    text = re.sub(r"n't", " not ", text);
    text = re.sub(r"i'm", "i am ", text);
    text = re.sub(r"\'re", " are ", text);
    text = re.sub(r"\'d", " would ", text);
    text = re.sub(r"\'ll", " will ", text);
    text = re.sub(r"\'scuse", " excuse ", text);
    text = re.sub('\W', ' ', text);
    text = re.sub('\s+', ' ', text);
    text = text.strip(' ');
    return text;

filt_X=raw_X.apply(filtering);
X=filt_X.apply(wordToken);

#Plotting histogram for the length of the comments
Len_X=X.apply(len);
plt.hist(Len_X,bins=50);
plt.show();
#print(Len_X);

#Splitting data into two sets: Train and Test
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3);

#Transforming text to feature vectors using TfidfVectorizer
TV=TfidfVectorizer(max_features=5000);
X_train=TV.fit_transform(X_train);
X_test=TV.transform(X_test);

column_names=Y_train.columns;

#Training model using training data and calculating the accuracy
from sklearn.linear_model import LogisticRegression;
LR=LogisticRegression(C=12.0);

for x in column_names:
    target=Y_train[x];
    LR.fit(X_train,target);
    Y_pred=LR.predict(X_test);
    Accuracy=accuracy_score(Y_test[x],Y_pred);
    print("Accuracy for ",x,":",Accuracy);


