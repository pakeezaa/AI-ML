import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv('MinMaxScaler/wine_data.csv',header=None,usecols=[0,1,2])
df.columns=['Class Label','Alcohol','Malic Acid']
print(df.head())

sns.kdeplot(df['Alcohol'])
plt.savefig('MinMaxScaler/Alcohol.png',dpi=300,bbox_inches='tight')
plt.show()


sns.kdeplot(df['Malic Acid'])
plt.savefig('MinMaxScaler/Malic_Acid.png',dpi=300,bbox_inches='tight')
plt.show()

color_dict={1:'red',2:'green',3:'blue'}
sns.scatterplot(x=df['Alcohol'],y=df['Malic Acid'],hue=df['Class Label'],palette=color_dict)
plt.savefig('MinMaxScaler/scatterplot.png',dpi=300,bbox_inches='tight')
plt.show()
#trai test split
X_train,X_test,y_train,y_test=train_test_split(df.drop(['Class Label'],axis=1),df['Class Label'],test_size=0.3,random_state=0)

#minmaxscalar
scaler=MinMaxScaler()
scaler.fit(X_train)
scaled_X_train=scaler.transform(X_train)
scaled_X_test=scaler.transform(X_test)

#transform to dataframe
scaled_X_train=pd.DataFrame(scaled_X_train,columns=X_train.columns)
scaled_X_test=pd.DataFrame(scaled_X_test,columns=X_test.columns)

print(np.round(scaled_X_train.describe(),1))

fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(12,5))

ax1.scatter(X_train['Alcohol'],X_train['Malic Acid'],c=y_train)
ax1.set_title('before Scaling')

ax2.scatter(scaled_X_train['Alcohol'],scaled_X_train['Malic Acid'],c=y_train)
ax2.set_title("After Scaling")
plt.savefig('MinMaxScaler/Scatter.png',dpi=300,bbox_inches='tight')
plt.show()


fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(12,5))

#before
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['Alcohol'],ax=ax1)
sns.kdeplot(X_train['Malic Acid'],ax=ax1)

ax2.set_title('After Scaling')
sns.kdeplot(scaled_X_train['Alcohol'],ax=ax2)
sns.kdeplot(scaled_X_train['Malic Acid'],ax=ax2)
plt.savefig('MinMaxScaler/kdeplot_bef_aft.png',dpi=300,bbox_inches='tight')
plt.show()


lr=LogisticRegression()
lr_scaled=LogisticRegression()
lr=lr.fit(X_train,y_train)
print(lr)
lr_scaled=lr_scaled.fit(scaled_X_train,y_train)
print(lr_scaled)
#predict
y_pred=lr.predict(X_test)
print(y_pred)
scaled_y_predict=lr_scaled.predict(scaled_X_test)
print(scaled_y_predict)

print("actualscore:",accuracy_score(y_test,y_pred))
print("scaled score:",accuracy_score(y_test,scaled_y_predict))