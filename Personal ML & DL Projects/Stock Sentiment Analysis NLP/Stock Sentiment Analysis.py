# importing the Dataset

import pandas as pd
import re
import nltk
import os
import glob
import re

import seaborn as sns
import matplotlib.pyplot as plt


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm
from sklearn.ensemble  import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


### Importing data
file_path = os.path.abspath("Stock Sentiment Analysis")
df = pd.read_csv(r'C:\Users\D100793\OneDrive - Citizens\Desktop\Krish Naik\Live Implementation Practice\Stock Sentiment Analysis NLP\Stock.csv', encoding = "ISO-8859-1")


###################Feature Engineering#################################
df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%Y/%m/%d')
df.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

## Renaming column names for ease of access
Headline_Cols= [str(i) for i in range(25)]
column_names = ['Date','Label']
column_names.extend(Headline_Cols)
new_Index=[i for i in column_names]
df.columns= new_Index

## Converting to lower case
df.apply(lambda x: x.astype(str).str.lower())

## Concatenating all news and stemming them
stemmer = PorterStemmer()
headlines = []
for row in range(0,len(df.index)):
    headlines.append(' '.join(str(x) for x in df.loc[row,Headline_Cols]))
    
for i in range(0,len(headlines)):
    words = nltk.word_tokenize(headlines[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    headlines[i] = ' '.join(words)

df = df.drop(columns = Headline_Cols)
## Using Bag of words
# Creating the Bag of Words model
cv = CountVectorizer(max_features=4000,ngram_range=(2,2))
headlines_df = cv.fit_transform(headlines).toarray()
headlines_df = pd.DataFrame(headlines_df, columns = cv.get_feature_names() )
df = df.join(headlines_df)
df['Label'] = df['Label'].map({0:'Down', 1:'Up'})

## Breaking data into Train and Test for Model
train = df.loc[df['Date'] < '2015-01-01',:]
test = df.loc[df['Date'] > '2014-12-31',:]
train = train.drop(columns = 'Date')
test = test.drop(columns = 'Date')


X_train = train.iloc[:,1:]
y_train = train.loc[:,'Label']

X_test = test.iloc[:,1:]
y_test = test.loc[:,'Label']

# implement RandomForest Classifier
clf = RandomForestClassifier(n_estimators=250,criterion='entropy')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

labels = list(set(y_train))
labels.sort()
rf_cm = confusion_matrix(y_test,y_pred, labels = labels)
ax= plt.subplot()
sns.heatmap(rf_cm, annot=True, ax = ax,cmap='Blues',fmt="d"); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted');ax.set_ylabel('Actual'); 
ax.set_title('Confusion matrix of the Decision Tree'); 
ax.xaxis.set_ticklabels(labels,rotation = 90); 
ax.yaxis.set_ticklabels(labels,rotation = 0);

print(classification_report(y_test, y_pred))

print("Accuracy:",clf.score(X_test, y_test))

















