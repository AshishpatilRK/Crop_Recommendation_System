from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle
import os
from crop_recommendn_prdn.pipeline.prediction_pipeline import PredictPipline,CustomData

# from crop_recommendn_prdn.utils.util import load_object
# importing model

# preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
# model_path = os.path.join("artifacts","model.pkl")
# preprocessor = load_object(preprocessor_path)
# model = load_object(model_path)
# model = pickle.load(open('model.pkl','rb'))
# sc = pickle.load(open('standscaler.pkl','rb'))
# ms = pickle.load(open('minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    data = CustomData(
            N = int(request.form.get("Nitrogen")),
            P = int(request.form.get("Phosporus")),
            K = int(request.form.get("Potassium")),
            temperature = int(request.form.get("Temperature")),
            humidity = int(request.form.get("Humidity")),
            ph = int(request.form.get("Ph")),
            rainfall = int(request.form.get("Rainfall"))
            )

    final_data = data.get_data_as_data_frame()
    predict_pipline = PredictPipline()
    pred = predict_pipline.predict(final_data)
    # N = request.form['Nitrogen']
    # P = request.form['Phosporus']
    # K = request.form['Potassium']
    # temp = request.form['Temperature']
    # humidity = request.form['Humidity']
    # ph = request.form['Ph']
    # rainfall = request.form['Rainfall']

    # feature_list = [N, P, K, temp, humidity, ph, rainfall]
    # single_pred = np.array(feature_list).reshape(1, -1)

    # scaled_features = ms.transform(single_pred)
    # final_features = preprocessor.transform(single_pred)
    # prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if pred[0] in crop_dict:
        crop = crop_dict[pred[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result)




# python main
if __name__ == "__main__":
    app.run(debug=True)