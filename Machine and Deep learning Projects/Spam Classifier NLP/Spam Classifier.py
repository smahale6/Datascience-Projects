# importing the Dataset

import pandas as pd
import re
import nltk
import os
import glob

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



messages = pd.read_csv('spam.csv', encoding = "ISO-8859-1")
messages = messages.loc[:,['v1','v2']]
messages.columns = ["label", "message"]
#Data cleaning and preprocessin

lemmatizer = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the Bag of Words model
cv = CountVectorizer(max_features=2500)
X_Bow = cv.fit_transform(corpus).toarray()
X_Bow = pd.DataFrame(X_Bow, columns = cv.get_feature_names() )
y_Bow = pd.get_dummies(messages['label'])
y_Bow = y_Bow.loc[:,['spam']]


# Creating the TFIDF of Words model
tv = TfidfVectorizer(max_features=2500)
X_TFIDF = tv.fit_transform(corpus).toarray()
X_TFIDF = pd.DataFrame(X_TFIDF, columns = tv.get_feature_names() )
y_TFIDF = pd.get_dummies(messages['label'])
y_TFIDF = y_TFIDF.loc[:,['spam']]



## Hyperparametertuning
def model_hyperparameter_tuning(X,y):
    model_params = {

        'Logistic_Regression' : {
            'model': LogisticRegression(solver='liblinear'),
            'params': {
                'C': list(range(1,5))
            }
         },  
        'naive_bayes_gaussian': {
           'model': GaussianNB(),
           'params': {}
         },
        'naive_bayes_multinomial': {
           'model': MultinomialNB(),
           'params': {}
         },
        'Random_Forest': {
            'model': RandomForestClassifier(),
            'params' : {
                'n_estimators': list(range(1,25))
            }
        },
        'Decision_Tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'criterion': ['gini','entropy'],  
                'splitter' : ['best','random'],
                'min_samples_leaf' : list(range(1,25))
            }
        },
        'SVM': {
            'model': svm.SVC(gamma='auto'),
            'params' : {
                'C': list(range(1,10)),
                'kernel': ['rbf','linear']
            }  
        },
        'XG_Boost': {
            'model': XGBClassifier(silent=True, verbosity = 0),
            'params': { 'n_estimators' :  [10, 50, 100, 130], 
                        'criterion': ['gini', 'entropy'], 
                        'max_depth': range(2, 10, 1) }
            }  
    }
    scores = []

    for model_name, mp in model_params.items():
        print('Running ' + model_name)
        clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
        clf.fit(X, y)
        scores.append({
            'model': model_name,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_,
            'best_estimators' : clf.best_estimator_
        })
    hyperparameter_tuning_results = pd.DataFrame(scores,columns=['model','best_score','best_params','best_estimators'])
    return hyperparameter_tuning_results

BOW_Results = model_hyperparameter_tuning(X_Bow,y_Bow)
BOW_Results['Method'] = 'Bag of words'

TFIDF_Results = model_hyperparameter_tuning(X_TFIDF,y_TFIDF) 
TFIDF_Results['Method'] = 'TFIDF'

Combined_results = pd.concat([BOW_Results,TFIDF_Results])
Combined_results.sort_values(by = 'best_score', inplace = True, ascending = False)
Combined_results = Combined_results.reset_index(drop = True)

best_estimator  = Combined_results.loc[0,'best_estimators']
method = Combined_results.loc[0,'Method']


if method == 'TFIDF':
    X_train,X_test,y_train,y_test = train_test_split(X_TFIDF, y_TFIDF, test_size=0.33, random_state=42)
else:
    X_train,X_test,y_train,y_test = train_test_split(X_Bow, y_Bow, test_size=0.33, random_state=42)


best_estimator.fit(X_train, y_train)
y_pred = best_estimator.predict(X_test)

if len(y_test['spam'].unique()) == 1:
    score = accuracy_score(y_test, y_pred)
else:
    score = roc_auc_score(y_test, y_pred) 






