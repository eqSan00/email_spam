from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Muat model dari file pickle
with open('modelSpam_detection.pkl', 'rb') as file:
    model = pickle.load(file)

# Muat vectorizer dari file pickle
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html', message="")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        
        # Preprocess the message using the loaded vectorizer
        message_transformed = vectorizer.transform([message])
        
        # Make prediction
        prediction = model.predict(message_transformed)[0]  # Assuming model.predict returns a list/array

        if prediction == 1:
            result = "Spam"
        else:
            result = "Not Spam"
            
        # Return the result and preserve the input message
        return render_template('index.html', prediction=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)
