from flask import Flask, request, render_template
import pickle
import pandas as pd
import re
from car_data_prep import prepare_data

app = Flask(__name__)

# Load the trained model
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the train columns
with open('train_columns.pkl', 'rb') as f:
    train_columns = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'manufactor': request.form['manufactor'],
            'Year': int(request.form['Year']),
            'model': request.form['model'],
            'Hand': int(request.form['Hand']),
            'Gear': request.form['Gear'],
            'capacity_Engine': float(request.form['capacity_Engine']),
            'Engine_type': request.form['Engine_type'],
            'Prev_ownership': request.form['Prev_ownership'],
            'Curr_ownership': request.form['Curr_ownership'],
            'Area': request.form['Area'],
            'City': request.form['City'],
            'Description': request.form['Description'],
            'Color': request.form['Color'],
            'Km': float(request.form['Km']),
            'Test': request.form['Test']
        }

        # Validate Test format
        if not re.match(r'\d{2}\.\d{2}\.\d{2}', data['Test']):
            return render_template('index.html', prediction="Invalid Test format. Correct format: 20.01.22", **request.form)

        # Validate Area and City format
        if not re.match(r'^[א-ת\s]+$', data['Area']):
            return render_template('index.html', prediction="Invalid Area format. Only Hebrew text is allowed.", **request.form)
        if not re.match(r'^[א-ת\s]+$', data['City']):
            return render_template('index.html', prediction="Invalid City format. Only Hebrew text is allowed.", **request.form)
        if not re.match(r'^[א-ת\s]+$', data['Color']):
            return render_template('index.html', prediction="Invalid Color format. Only Hebrew text is allowed.", **request.form)

        # Create a DataFrame from the data
        data_df = pd.DataFrame([data])

        # Print original data for debugging
        print("Original data:")
        print(data_df)

        # Prepare the data
        prepared_data = prepare_data(data_df, train_columns)

        # Predict the car price
        prediction = model.predict(prepared_data)[0]

        # If prediction is less then 2000, set it to 2000
        if prediction < 2000:
            prediction = 2000.0
        
        # Print prediction for debugging
        print("Prediction:")
        print(prediction)

        # Return the prediction
        return render_template('index.html', prediction=f"{prediction:.2f}", **request.form)

    except Exception as e:
        return render_template('index.html', prediction=str(e), **request.form)

if __name__ == '__main__':
    app.run(debug=True)
