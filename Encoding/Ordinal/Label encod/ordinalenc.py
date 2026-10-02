import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('Encoding/Ordinal/Label encod/customer.csv')
print(df.sample(5))
df=df.iloc[:,2:]
print(df.head())
X=df.drop(['purchased'],axis=1)
y=df['purchased']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

enc=OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']])
enc.fit(X_train)
X_train_enc=enc.transform(X_train)
X_test_enc=enc.transform(X_test)
print(X_train_enc)

le=LabelEncoder()
le.fit(y_train)
y_train_enc=le.transform(y_train)
y_test_enc=le.transform(y_test)
print(y_train)