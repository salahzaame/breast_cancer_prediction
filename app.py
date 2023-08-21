from flask import Flask , render_template, jsonify, request
import pickle
import numpy as np
#creating an app object using the Flask class
app = Flask(__name__)

#load the pickel model 
model = pickle.load(open("breast_cancer_model.pkl", "rb"))

column_names = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
#Define the route to be home 
#use the route() decorator to tell Flask what URL should trigger our function .
@app.route('/')
def Home():
    # print(column_names)
    return render_template("index.html") #<----index.html file should be in 'templates' folder .

@app.route("/predict", methods = ['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()] # fetching the values from the form and convert it to floats 
    features = [np.array(float_features)] #converting to an array that has the same shape of our prediction data 
    prediction = model.predict(features) #make predictions .

    return render_template("index.html", prediction_text = "Breast cancer prediction {}".format(prediction))


if __name__ == "__main__": #main function 

    app.run(debug=True)

