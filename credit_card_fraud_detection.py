# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

credit_card_data=pd.read_csv("/content/creditcard.csv")

credit_card_data.head()

legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]

sample_legit=legit.sample(492)
new_dataset=pd.concat((sample_legit, fraud),axis=0)

x=new_dataset.drop("Class",axis=1)
y=new_dataset.Class

x.head()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

model=LogisticRegression()
model.fit(x_train,y_train)

train_prediction=model.predict(x_train)
training_accuracy=accuracy_score(train_prediction,y_train)
print("Training Accuracy is:",training_accuracy*100)

test_prediction=model.predict(x_test)
testing_accuracy=accuracy_score(test_prediction,y_test)
print("Testing Accuracy is:",testing_accuracy*100)

