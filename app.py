from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime
import os

app = Flask(__name__)
model = joblib.load('fraud_model.pkl')
gov_data = pd.read_csv('mock_government_records.csv')
log_file = 'prediction_log.csv'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    dob = request.form['dob']
    mobile = request.form['mobile']
    gender = request.form['gender']
    aadhaar = request.form['aadhaar']
    pan = request.form['pan']

    # Check if the record exists in mock government data
    matched = gov_data[
        (gov_data['Aadhaar'].astype(str) == aadhaar) &
        (gov_data['pan'].astype(str) == pan)
    ]

    # Create hashed input for prediction
    combined = aadhaar + pan
    hash_feature = pd.DataFrame([[hash(combined)]], columns=["HashFeature"])

    # Predict using the trained model
    prediction = model.predict(hash_feature)[0]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Result Message
    if not matched.empty and prediction == 1:
        result = "✅ Identity Verified - Legitimate"
    elif matched.empty and prediction == -1:
        result = "⚠️ Identity Suspicious - Synthetic Identity Detected"
    else:
        result = "⚠️ Identity Not Found in Government Records - Verification Needed"

    # Log the prediction
    log_entry = {
        'Name': name,
        'DOB': dob,
        'Mobile': mobile,
        'Gender': gender,
        'Aadhaar': aadhaar,
        'pan': pan,
        'Prediction': result,
        'Timestamp': timestamp
    }

    if os.path.exists(log_file):
        pd.DataFrame([log_entry]).to_csv(log_file, mode='a', index=False, header=False)
    else:
        pd.DataFrame([log_entry]).to_csv(log_file, index=False)

    return render_template('result.html', result=result, details={
        'Name': name,
        'DOB': dob,
        'Mobile': mobile,
        'Gender': gender,
        'Aadhaar': aadhaar,
        'PAN': pan,
        'Time': timestamp
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
