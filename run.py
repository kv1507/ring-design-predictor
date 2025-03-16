from flask import Flask, request, render_template
import pandas as pd
import pickle

# Load trained model & encoder
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

app = Flask(__name__, template_folder="templates")

def process_input_data(input_data):
    input_df = pd.DataFrame([input_data])

    categorical_features = ['Party Name', 'First Color of Diamond', 'Second Color of Diamond', 
                            'Third Color of Diamond', 'Shape', 'Type']

    # Apply one-hot encoding
    input_df = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)

    # Ensure all expected features exist
    missing_cols = set(feature_names) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Add missing columns with default value 0

    # Ensure the order of columns matches training data
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    return input_df

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "GET":
        return render_template("index2.html")  # Show the form page on GET requests

    try:
        form_data = {key.capitalize() if key.lower() == "type" else key: value for key, value in request.form.to_dict().items()}

        # Handle missing fields gracefully
        required_fields = ['Quantity', 'Number of Diamonds', 'Shape', 'First Color of Diamond', 'Second Color of Diamond', 'Third Color of Diamond']
        for field in required_fields:
            if field not in form_data or form_data[field] == "":
                return f"Error: Missing field '{field}' in input."

        # Standardize the key naming
        if 'type' in form_data:
            form_data['Type'] = form_data.pop('type') 

        # Convert numerical values safely
        form_data['Quantity'] = int(form_data.get('Quantity', 0))
        form_data['Number of Diamonds'] = int(form_data.get('Number of Diamonds', 0))

        # Process input
        input_df = process_input_data(form_data)
        prediction = model.predict(input_df)[0]
        predicted_choice = label_encoder.inverse_transform([prediction])[0]

        return render_template("index2.html", prediction=predicted_choice)

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/aboutproject')
def about_project():
    return render_template("aboutproject.html")

@app.route('/aboutapi')
def about_api():
    return render_template("aboutapi.html")

@app.route('/home')
def home_page():
    return render_template("home.html")

@app.route('/adv')
def adv():
    return render_template("adv.html")

@app.route('/navbar')
def navbar():
    return render_template("navbar.html")



if __name__ == '__main__':
    app.run(debug=True)
