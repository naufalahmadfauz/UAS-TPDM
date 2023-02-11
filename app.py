from flask import Flask, render_template, request
import pickle
import sklearn
import pandas as pd

modeL_DT = open('models_for_apppy/model_DT.pkl', 'rb')
modeL_GNB = open('models_for_apppy/model_GNB.pkl', 'rb')
modeL_KNN = open('models_for_apppy/model_KNN.pkl', 'rb')
modeL_LR = open('models_for_apppy/model_LR.pkl', 'rb')
modeL_RF = open('models_for_apppy/model_RF.pkl', 'rb')
modeL_SVM = open('models_for_apppy/model_SVM.pkl', 'rb')

model_load_DT = pickle.load(modeL_DT, encoding='bytes')
model_load_GNB = pickle.load(modeL_GNB, encoding='bytes')
model_load_KNN = pickle.load(modeL_KNN, encoding='bytes')
model_load_LR = pickle.load(modeL_LR, encoding='bytes')
model_load_RF = pickle.load(modeL_RF, encoding='bytes')
model_load_SVM = pickle.load(modeL_SVM, encoding='bytes')

app = Flask(__name__)


@app.route('/')
def index():  # put application's code here
    return render_template('input_prediksi.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the insurance cost based on user inputs
    and render the result to the html page
    '''
    pm10, pm25, so2, co, o3, no2 = [x for x in request.form.values()]

    data = []

    data.append(int(pm10))
    data.append(int(pm25))
    data.append(int(so2))
    data.append(int(co))
    data.append(int(o3))
    data.append(int(no2))

    x_test = pd.DataFrame([data], columns=['PM10', 'PM2.5', 'SO2', 'CO', 'O3', 'NO2'])
    print(data)
    dt_prediction = model_load_DT.predict(x_test)
    gnb_prediction = model_load_GNB.predict(x_test)
    knn_prediction = model_load_KNN.predict(x_test)
    lr_prediction = model_load_LR.predict(x_test)
    svm_prediction = model_load_SVM.predict(x_test)
    rf_prediction = model_load_RF.predict(x_test)
    output = []

    output.append(dt_prediction[0])
    output.append(gnb_prediction[0])
    output.append(knn_prediction[0])
    output.append(lr_prediction[0])
    output.append(svm_prediction[0])
    output.append(rf_prediction[0])

    return render_template('hasil_prediksi.html', hasil_pred=output)


if __name__ == '__main__':
    app.run(debug=True)
