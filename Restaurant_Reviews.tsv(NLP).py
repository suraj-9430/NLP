# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:40:39 2023

@author: rajsu
"""

import pandas as pd
dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter=("\t"))


#step -2 cleaning the data or  removing the full stop and puntuation marks
"""it use to keep the all letter in lower case"""
"""here we remove the words like that,this these,here,their etc"""
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

dataset_Af=[]
for i in range(1000):
    review=re.sub('[^a-zA-Z]',' ',dataset["Review"][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    dataset_Af.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(dataset_Af).toarray()
y=dataset.iloc[:,1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)


from sklearn.ensemble import RandomForestClassifier
rdc=RandomForestClassifier(n_estimators=150)

rdc.fit(x_train, y_train)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,rdc.predict(x_test))