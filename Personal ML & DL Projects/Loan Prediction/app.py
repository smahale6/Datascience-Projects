import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = [ 'Credit_History', 'Property_Area', 'Married', 'LoanAmount'])                         
    
    prediction = model.predict(data_unseen)

    output = prediction
    
    if output == 1:
        label = 'approved'
    else:
        label = 'not approved'

    return render_template('index.html', prediction_text='Loan Status is {}'.format(label))

if __name__ == "__main__":
    app.run(debug=False)