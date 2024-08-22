from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and encoders
model = joblib.load('improved_student_performance_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

# Define the expected features
features = [
    'Study Hours per Week', 
    'Attendance Rate', 
    'Previous Grades', 
    'Participation in Extracurricular Activities', 
    'Parent Education Level'
]

@app.route('/student_performance/sample', methods=['GET'])
def sample():
    sample_data = {
        "Study Hours per Week": 15,
        "Attendance Rate": 85,
        "Previous Grades": 90,
        "Participation in Extracurricular Activities": "Yes",  # Use string values
        "Parent Education Level": "Bachelor"
    }
    return jsonify(sample_data)

@app.route('/student_performance/explain', methods=['GET'])
def explain():
    explanation = {
        "Study Hours per Week": "Number of hours the student studies per week.",
        "Attendance Rate": "Percentage of classes attended by the student.",
        "Previous Grades": "Average of the student's previous grades.",
        "Participation in Extracurricular Activities": "Participation in extracurricular activities (No or Yes).",
        "Parent Education Level": "Parent's highest education level (Associate, Bachelor, Doctorate, High School, Master)."
    }
    return jsonify(explanation)

@app.route('/student_performance/evaluate', methods=['POST'])
def evaluate():
    try:
        input_data = request.get_json()
        print("Received input data:", input_data)

        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])

        # Ensure the input data has all the required columns
        missing_features = [feature for feature in features if feature not in df.columns]
        if missing_features:
            return jsonify({"error": f"Missing required features: {', '.join(missing_features)}"}), 400

        # Handle categorical features with label encoders
        for col in ['Participation in Extracurricular Activities', 'Parent Education Level']:
            if col in df:
                value = df[col].values[0]
                if value not in label_encoders[col].classes_:
                    return jsonify({"error": f"Invalid value for {col}: {value}"}), 400
                df[col] = label_encoders[col].transform(df[col])

        # Ensure the input data has the correct order of features
        df = df[features]

        # Debug: Print the transformed DataFrame
        print("Transformed DataFrame:", df)

        # Make prediction
        prediction = model.predict(df)
        print("Model prediction:", prediction)

        # Check for NaN and decode prediction
        if pd.isna(prediction[0]):
            return jsonify({"error": "Prediction resulted in NaN."}), 500

        result = target_encoder.inverse_transform([prediction[0]])[0]
        return jsonify({"Prediction": result})

    except KeyError as e:
        return jsonify({"error": f"Key error: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Value error: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
