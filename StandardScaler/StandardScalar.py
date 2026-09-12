import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_csv('FeatureScaling/StandardScaler/Social_Network_Ads.csv')
#print(df.head(4))

X=df.drop(['Purchased','Gender'],axis=1)
y=df['Purchased']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()

scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)

scaled_X_train=pd.DataFrame(scaled_X_train,columns=X_train.columns)
scaled_X_test=pd.DataFrame(scaled_X_test,columns=X_test.columns)
print(np.round(X_train.describe()))
print(np.round(scaled_X_train.describe()))

#figures to differntiate
fig, (ax1,ax2)=plt.subplots(ncols=2,figsize=(12,5))
ax1.scatter(X_train['Age'],X_train['EstimatedSalary'],color='red')
ax1.set_title("Before SCaling")

ax2.scatter(scaled_X_train['Age'],scaled_X_train['EstimatedSalary'])
ax2.set_title("After SCaling")
plt.savefig('FeatureScaling/StandardScaler/scaling.png',dpi=300,bbox_inches='tight')
plt.show()

fig, (ax1,ax2)=plt.subplots(ncols=2,figsize=(12,5))
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['Age'],ax=ax1)
sns.kdeplot(X_train['EstimatedSalary'],ax=ax1)

ax1.set_title('After Scaling')
sns.kdeplot(scaled_X_train['Age'],ax=ax2)
sns.kdeplot(scaled_X_train['EstimatedSalary'],ax=ax2)
plt.savefig('FeatureScaling/StandardScaler/kdeplot.png',dpi=300,bbox_inches='tight')
plt.show()

#model
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