from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__) 
model = pickle.load(open('models/Final_Model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():

    val = []

    int_rate        = float(request.form['int_rate'])
    val = np.append(val, int_rate)

    grade           = float(request.form['grade'])
    val = np.append(val, grade)

    sub_grade       = float(request.form['sub_grade'])
    val = np.append(val, sub_grade)

    total_rec_prncp = float(request.form['total_rec_prncp'])
    val = np.append(val, total_rec_prncp)

    total_pymnt_inv = float(request.form['total_pymnt_inv'])
    val = np.append(val, total_pymnt_inv)

    total_pymnt     = float(request.form['total_pymnt'])
    val = np.append(val, total_pymnt)

    out_prncp       = float(request.form['out_prncp'])
    val = np.append(val, out_prncp)

    out_prncp_inv   = float(request.form['out_prncp_inv'])
    val = np.append(val, out_prncp_inv)

    last_pymnt_amnt = float(request.form['last_pymnt_amnt'])
    val = np.append(val, last_pymnt_amnt)

    val = val.reshape(9,)

    val_predict = model.predict([val])
    return render_template('predict.html', prediction=val_predict)

if __name__ == "__main__":
    app.run(debug=True)