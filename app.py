from flask import Flask , render_template, jsonify, request
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
#creating an app object using the Flask class
app = Flask(__name__)

#load the pickel model 
model = pickle.load(open("breast_cancer_model.pkl", "rb"))

data = pd.read_csv('data.csv')
mean_data = data.drop(['radius_mean', 'texture_mean', 'perimeter_mean' , 'area_mean', 'diagnosis', 'id'], axis=1).mean()

#Define the route to be home 
#use the route() decorator to tell Flask what URL should trigger our function .
@app.route('/')
def Home():
    # print(column_names)
    return render_template("index.html") #<----index.html file should be in 'templates' folder .

@app.route("/predict", methods = ['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()] # fetching the values from the form and convert it to floats 
    # features = [np.array(float_features)] #converting to an array that has the same shape of our prediction data 
    # features = [np.concatenate([float_features[:4], mean_data.tolist()])]
    features = float_features[:4]+mean_data.tolist()

    features = [np.array(features)]
    
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)

    prediction = model.predict(features) #make predictions .

    return render_template("index.html", prediction_text = "Breast cancer prediction {}".format(prediction))


if __name__ == "__main__": #main function 

    app.run(debug=True , port= 5000 , host='0.0.0.0')
