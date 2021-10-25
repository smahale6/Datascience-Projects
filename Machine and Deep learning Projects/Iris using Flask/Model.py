from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import pickle

#import data
df = pd.read_csv("iris.csv")

#select independent and dependent variable
X = df.iloc[:,:-1]
y = df.iloc[:,-1]


#Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 50)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Instantiate the model
classifier = RandomForestClassifier()

#Fit the model
classifier.fit(X_train, y_train)
pickle.dump(classifier,open("model.pkl","wb"))
