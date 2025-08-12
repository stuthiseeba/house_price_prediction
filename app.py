import os
import pandas as pd
from flask import Flask, render_template,request
import pickle
import numpy as np

app = Flask(__name__, template_folder=os.path.abspath('templates'))
data=pd.read_csv("cleaned_data.csv")
pipe=pickle.load(open("linear_model.pkl",'rb'))

@app.route('/')
def home():
    locations=sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location=request.form.get('location')
    bhk=request.form.get('bhk')
    bath=request.form.get('bath')
    sqft=request.form.get('total_sqft')

    

    input=pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    prediction=pipe.predict(input)[0]*10000


    return str(np.round(prediction,2))



if __name__ == '__main__':
    app.run(debug=True)
