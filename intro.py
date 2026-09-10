
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import pickle
df=pd.read_csv('placement.csv')
print(df.head())
print(df.info())
#deciding features and output
X=df.drop(['placement','Unnamed: 0'],axis=1)
y=df['placement']

#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)

#feature scaling , so that model do not get dominate on higher values 
scaler=StandardScaler()
scale_X_train=scaler.fit_transform(X_train)
scale_X_test=scaler.transform(X_test)

#model
model=LogisticRegression()
#training
model.fit(scale_X_train,y_train)

#prediction
y_pred=model.predict(scale_X_test)

print(y_test)
#accuracy score
acc_score=accuracy_score(y_test,y_pred)
print(acc_score)

#plott


plot_decision_regions(scale_X_train, y_train.values, clf=model)
plt.xlabel('CGPA')
plt.ylabel('IQ')

plt.savefig('visual.png',dpi=300,bbox_inches='tight')
plt.show()

pickle.dump(model,open('model.pkl','wb'))