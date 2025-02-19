import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pyfunc
import streamlit as st
import joblib


flight_model_paths = {
    'Linear Regression': 'runs:/cd528040d2f84263bfe0d0822a5a8326/LinearRegression_model',
    'Lasso': 'runs:/0722a606b9194293a2a949450e3281ce/Lasso_model', 
    'Ridge': 'runs:/648e0641fa8a40bba7eeb95bdccbd8c9/Ridge_model', 
    'Xgboost': 'runs:/747cfd2259614f0ebf56a454c4736b6a/XGBoost_model',
    'Random Forest':'runs:/7c21208881d1410baee19f9c1a003493/RandomForest_model',
    'Decision Tree':'runs:/8f788ab9a36f4669b747c31b602a6f6a/DecisionTree_model',
    'KNN':'runs:/12eeb3d6f94d4cb78dfa81fe68ab7bd7/KNeighbors_model',
    'AdaBoost':'runs:/a388d1b3c71644b999474b5fb61b0bc5/AdaBoost_model',
} 

satisfaction_model_paths = {
    "LogisticRegression": 'runs:/6a4b4bfea85946869db7603abef42f2d/LogisticRegression_model',
    "KNeighbors": 'runs:/14c5aa9caacd406ab4fa736bfbacd982/KNeighbors_model',
    "DecisionTree": 'runs:/418b8232ed7e4bc1847a36c414d6a0fa/DecisionTree_model',
    "RandomForest": 'runs:/2f1580d6763341b28c7a99dd2c8ff9f9/RandomForest_model',
    "AdaBoost": 'runs:/88aff6a7196242d19a905a98ac8e4f80/AdaBoost_model',
    "Ridge": 'runs:/59a4572e54764c058cb1b9d25bbc0d41/Ridge_model',
    "Lasso": 'runs:/9282a9a069584ba0b7e2e3bab2d792c6/Lasso_model',
    "XGBoost": 'runs:/074f00c0573446c4a2d41e28682df238/XGBoost_model'
}

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Flight Price", "Passenger Satisfaction Prediction"])

if page == "Predict Flight Price":
    st.title("Flight Price Prediction")
    

    source_cities = ['Delhi', 'Kolkata', 'Mumbai', 'Chennai', 'Banglore']
    destination_cities = ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'Banglore']

    source = st.selectbox('Source', source_cities)

    # Dynamically filter out selected source from destination options
    filtered_destinations = [city for city in destination_cities if city != source]
    destination = st.selectbox('Destination', filtered_destinations)

    input_data = {
        'Dep_Date': st.date_input('Departure Date'),
        'Dep_Time': st.time_input('Departure Time'),
        'Arrival_Date': st.date_input('Arrival Date'),
        'Arrival_Time': st.time_input('Arrival Time'),
        'stops': st.number_input('Total Stops', min_value=0, max_value=4, step=1),
        'airline': st.selectbox('Airline', ['Jet Airways', 'IndiGo', 'Air India', 'Multiple carriers', 'SpiceJet', 'Vistara', 'GoAir', 'Multiple carriers Premium economy', 'Jet Airways Business', 'Vistara Premium economy', 'Trujet', 'Air Asia']),
        'Source': source,
        'Destination': destination
    }



    input_data['Dep_Time'] = pd.to_datetime(str(input_data['Dep_Date']) + ' ' + str(input_data['Dep_Time']))
    input_data['Arrival_Time'] = pd.to_datetime(str(input_data['Arrival_Date']) + ' ' + str(input_data['Arrival_Time']))


    input_df = pd.DataFrame([input_data])

    input_df['Dep_day'] = input_df['Dep_Time'].dt.day
    input_df['Dep_month'] = input_df['Dep_Time'].dt.month
    input_df['Dep_year'] = input_df['Dep_Time'].dt.year
    input_df['Dep_hour'] = input_df['Dep_Time'].dt.hour
    input_df['Dep_min'] = input_df['Dep_Time'].dt.minute
    input_df['Arrival_day'] = input_df['Arrival_Time'].dt.day
    input_df['Arrival_month'] = input_df['Arrival_Time'].dt.month
    input_df['Arrival_year'] = input_df['Arrival_Time'].dt.year
    input_df['Arrival_hour'] = input_df['Arrival_Time'].dt.hour
    input_df['Arrival_min'] = input_df['Arrival_Time'].dt.minute

    input_df['duration'] = input_df['Arrival_Time'] - input_df['Dep_Time']
    input_df['dur_hour'] = input_df['duration'].dt.components.hours
    input_df['dur_min'] = input_df['duration'].dt.components.minutes

    airline_dict = {
        'Air India': 0,
        'GoAir': 0,
        'IndiGo': 0,
        'Jet Airways': 0,
        'Jet Airways Business': 0,    
        'Multiple carriers': 0,
        'Multiple carriers Premium economy': 0,
        'SpiceJet': 0,
        'Trujet': 0,
        'Vistara': 0,    
        'Vistara Premium economy': 0,
    }
    if input_data['airline'] != 'Air Asia':
        airline_dict[input_data['airline']] = 1

    source_dict = {
        'Chennai': 0,
        'Delhi': 0,
        'Kolkata': 0,
        'Mumbai': 0,

    }
    if input_data['Source'] != 'Banglore':
        source_dict[input_data['Source']] = 1

    destination_dict = {
        'Cochin': 0,
        'Delhi': 0,
        'Hyderabad': 0,
        'Kolkata': 0,
        'New_Delhi': 0,
    }
    if input_data['Destination'] != 'Banglore':
        destination_dict[input_data['Destination']] = 1

    features = [
        input_df['stops'].values[0], input_df['Dep_hour'].values[0], input_df['Dep_min'].values[0],
        input_df['Dep_day'].values[0], input_df['Dep_month'].values[0], input_df['Dep_year'].values[0],
        input_df['Arrival_hour'].values[0], input_df['Arrival_min'].values[0], input_df['dur_hour'].values[0],
        input_df['dur_min'].values[0], input_df['Arrival_day'].values[0], input_df['Arrival_month'].values[0],
        input_df['Arrival_year'].values[0], *airline_dict.values(), *source_dict.values(), *destination_dict.values()
    ]

    scaler = joblib.load('scaler_path.pkl')

    scaled_features = scaler.transform([features])
    selected_model = st.sidebar.selectbox("Select Model", list(flight_model_paths.keys()))
    logged_model = flight_model_paths[selected_model]
    
    st.sidebar.subheader("Model Performance")
    try:
        model_metrics = mlflow.get_run(logged_model.split("/")[1]).data.metrics
        r2_score = float(model_metrics.get('r2', 0))
        mae = float(model_metrics.get('mae', 0))
        mse = float(model_metrics.get('mse', 0))
        st.sidebar.write(f"**R² Score:** {r2_score:.4f}")
        st.sidebar.write(f"**MAE:** {mae:.4f}")
        st.sidebar.write(f"**MSE:** {mse:.4f}")
    except Exception as e:
        st.sidebar.error(f"Error loading model metrics: {str(e)}")

    if st.button('Predict'):
        logged_model = flight_model_paths[selected_model]
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        prediction = loaded_model.predict(pd.DataFrame(scaled_features))
        st.write(f'Your Flight price is Rs. {round(prediction[0])}')
if page == "Passenger Satisfaction Prediction":    
        st.title("✈️ Passenger Satisfaction Prediction")
        st.write("Rate each aspect of your flight experience on a scale of 1 to 5.")

        st.sidebar.header("User Input Features")
        
        Age = st.sidebar.slider("Age", 18, 80, 30)
        Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
        Customer_Type = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
        Type_of_Travel = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal travel"])
        Class = st.sidebar.selectbox("Class", ["Eco", "Eco Plus", "Business"])
        Flight_Distance = st.sidebar.number_input("Flight Distance", 0, 10000, 500)
        Departure_Delay = st.sidebar.number_input("Departure Delay (minutes)", 0, 600, 0)
        Arrival_Delay = st.sidebar.number_input("Arrival Delay (minutes)", 0, 600, 0)
        
        categories = [
             'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness'
        ]
        order = ['Age', 'Class', 'Flight Distance', 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
       'Gender_Male', 'Customer Type_disloyal Customer',
       'Type of Travel_Personal Travel']
        ratings = {}
        with st.form("rating_form"):
            col1, col2 = st.columns(2)
            
            for i, category in enumerate(categories):
                with col1 if i % 2 == 0 else col2:
                    ratings[category] = st.slider(f"{category}", 1, 5, 3)
            
            submitted = st.form_submit_button("Submit Ratings")

        
        data = {
            "Age": Age,
            "Class": 0 if Class == "Eco" else (1 if Class == "Eco Plus" else 2),
            "Flight Distance": Flight_Distance,
            **ratings,
            "Departure Delay in Minutes": Departure_Delay,
            "Arrival Delay in Minutes": Arrival_Delay,
            "Gender_Male": 1 if Gender == "Male" else 0,
            "Customer Type_disloyal Customer": 1 if Customer_Type == "Disloyal Customer" else 0,
            'Type of Travel_Personal Travel': 1 if Type_of_Travel == "Personal travel" else 0,
            
        }
        
        st.write("### User Input Features:")
        st.write(data)
        
        selected_model = st.sidebar.selectbox("Select Model", list(satisfaction_model_paths.keys()))
        logged_model = satisfaction_model_paths[selected_model]

        st.sidebar.subheader("Model Performance")
        try:
            loaded_model = mlflow.pyfunc.load_model(logged_model)
            model_metrics = mlflow.get_run(logged_model.split("/")[1]).data.metrics
            st.sidebar.write(f"**Accuracy:** {model_metrics.get('accuracy', 'N/A'):.4f}")
            st.sidebar.write(f"**Precision:** {model_metrics.get('precision', 'N/A'):.4f}")
            st.sidebar.write(f"**Recall:** {model_metrics.get('recall', 'N/A'):.4f}")
            st.sidebar.write(f"**F1 Score:** {model_metrics.get('f1_score', 'N/A'):.4f}")
        except Exception as e:
            st.sidebar.error(f"Error loading model metrics: {str(e)}")

        if st.button("Predict Satisfaction"):
            try:
                input_df = (pd.DataFrame([data]))
                prediction = loaded_model.predict(input_df)
                probability = loaded_model.predict_proba(input_df)[:, 1] if hasattr(loaded_model, "predict_proba") else [0.5]
                
                st.write("### Prediction:")
                st.write("✅ **Satisfied**" if prediction[0] == 1 else "❌ **Not Satisfied**")

                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")



