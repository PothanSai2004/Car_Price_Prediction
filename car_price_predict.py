import numpy as np
from flask import render_template
from flask_cors import CORS, cross_origin
import pandas as pd
from flask import Flask, request, jsonify
import joblib
# import pickle
app = Flask(__name__)
cors=CORS(app)
car=pd.read_csv("cars_data.csv")
model_filename = 'linear_regression_model.pkl'
model_rf_filename = 'random_forest_regression1_model.pkl'
model_xg_filename = 'xgboost_model1.pkl'
@app.route('/')
def index():
    fuel_type = sorted(car['Fuel'].unique())
    transmission_type = sorted(car['Transmission'].unique())
    body_type = sorted(car['Build'].unique())
    return render_template('index.html', fuel_type=fuel_type, transmission_type=transmission_type,body_type=body_type)

@app.route('/predict', methods=['POST'])
def predict():
    # Load the trained model object from the file
    model = joblib.load(model_filename)
    model1 = joblib.load(model_rf_filename)
    model2 = joblib.load(model_xg_filename)

    company = request.form.get('company', '')
    reviews = float(request.form.get('reviews', 0))
    engine = float(request.form.get('displacement', 0))
    cylinder = float(request.form.get('cylinder', 0))
    seat = float(request.form.get('seat', 0))
    fuel_tank = float(request.form.get('fuel_tank', 0))
    rating = float(request.form.get('rating', 0))
    torque_nm = float(request.form.get('torque_nm', 0))
    torque_bhp = float(request.form.get('torque_rmp', 0))
    power_bhp = float(request.form.get('power_bhp', 0))
    power_rp = float(request.form.get('power_rp', 0))
    min_price = float(request.form.get('min_price', 0))
    max_price = float(request.form.get('max_price', 0))

    # Define the fuel type map
    fuel_type_map = {'CNG': 0, 'Diesel': 1, 'Electric': 2, 'Petrol': 3}

    # Get the selected fuel type from the form data
    fuel_type = request.form.get('fuel_type')

    # Convert the selected fuel type to its corresponding integer value
    fuel_type_int = fuel_type_map[fuel_type]

    # Define the transmission type map
    transmission_type_map = {'Automatic': 0, 'Electric': 1, 'Manual': 2}

    # Get the selected transmission type from the form data
    transmission = request.form.get('transmission')

    # Convert the selected transmission type to its corresponding integer value
    transmission_int = transmission_type_map[transmission]

    # Define the body type map
    body_type_map = {'Convertible': 0, 'Coupe': 1, 'Hatchback': 2, 'Hybrid': 3, 'Luxury': 4, 'MUV': 5, 'Minivan': 6, 'Pickup Truck': 7, 'SUV': 8, 'Sedan': 9, 'Wagon': 10}

    # Get the selected body type from the form data
    body = request.form.get('body')

    # Convert the selected body type to its corresponding integer value
    body_int = body_type_map[body]

    # Make predictions using the loaded model
    prediction = model.predict(pd.DataFrame(columns=['Reviews', 'Engine', 'Cylinders', 'Seat', 'Tank_size', 'Rating', 'Starting_Price', 'Max_price', 'Torque', 'Max_Torque_RPM', 'Power', 'Max_Power_RPM', 'Fuel_n', 'Transmission_n', 'Build_n'],
                                            data=np.array([reviews, engine, cylinder, seat, fuel_tank, rating, min_price, max_price, torque_nm, torque_bhp, power_bhp, power_rp, fuel_type_int, transmission_int, body_int]).reshape(1, 15)))
    # prediction1 = model1.predict(pd.DataFrame(columns=['Reviews', 'Engine', 'Cylinders', 'Seat', 'Tank_size', 'Rating', 'Starting_Price', 'Max_price', 'Torque', 'Max_Torque_RPM', 'Power', 'Max_Power_RPM', 'Fuel', 'Transmission', 'Build'],
    #                                           data=np.array([reviews, engine, cylinder, seat, fuel_tank, rating, min_price, max_price, torque_nm, torque_bhp, power_bhp, power_rp, fuel_type, transmission, body]).reshape(1, 15)))
    # prediction2 = model2.predict(pd.DataFrame(columns=['Reviews', 'Engine', 'Cylinders', 'Seat', 'Tank_size', 'Rating', 'Starting_Price', 'Max_price', 'Torque', 'Max_Torque_RPM', 'Power', 'Max_Power_RPM', 'Fuel', 'Transmission', 'Build'],
    #                                           data=np.array([reviews, engine, cylinder, seat, fuel_tank, rating, min_price, max_price, torque_nm, torque_bhp, power_bhp, power_rp, fuel_type, transmission, body]).reshape(1, 15)))

    return str(np.round(prediction[0], 2))
    # return str(np.round(prediction1[0], 2))
    # return str(np.round(prediction2[0], 2))
    # Combine the predictions into a single string
    # prediction_string = f"Linear Regression Prediction: {np.round(prediction[0], 2)}, Random Forest Prediction: {np.round(prediction1[0], 2)}, XGBoost Prediction: {np.round(prediction2[0], 2)}"
    #
    # return prediction_string

if __name__ == "__main__":
    app.run(debug=True)
