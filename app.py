from flask import Flask, render_template, request
import pickle as pkl
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

model = pkl.load(open('Cmodel.pkl', 'rb'))
model_accuracy = pkl.load(open('model_accuracy.pkl', 'rb'))  # load accuracy from pkl file

@app.route('/')
def home():
    return render_template('index.html', accuracy=round(model_accuracy * 100, 2))  # pass accuracy in %

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        prediction = model.predict([np.array(data)])

        input_labels = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_dict = {label: value for label, value in zip(input_labels, data)}
        input_dict['label'] = prediction[0]

        filename = 'Crop_recommendation.csv'
        df = pd.DataFrame([input_dict])

        if not os.path.isfile(filename):
            df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, mode='a', header=False, index=False)

        return render_template('result.html', crop=prediction[0])

    except Exception as e:
        return f"Error in prediction. Check inputs. <br><br> {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
