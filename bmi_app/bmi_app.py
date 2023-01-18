#Import Flask library and required callbacks
from flask import Flask,render_template,request
import pickle


#Launch App and Load the Model
app = Flask(__name__)
model = pickle.load(open('bmi_calc.pkl','rb'))

#Index or Landing Page
@app.route('/')
def home():
    result = '<Enter your Weight in Kgs and Height in metres>'
    return render_template('Mini_ML_pagee.html',**locals())

def isfloat(arg):
    try:
        float(arg)
        return True
    except:
        return False

#Function called on form submission
@app.route('/predict',methods=['POST','GET'])
def predict():
    wkg = request.form['weight']
    hm = request.form['height']
    
    if(isfloat(wkg)==True and isfloat(hm)==True and float(wkg)>0 and float(wkg)<120 and float(hm)>=1.4 and float(hm)<=2.1):
        result = round(model.predict([[wkg,hm]])[0],2)
        bmi = ""
        if result>=30:
            bmi="Obesity"
        elif result<30 and result>=25:
            bmi="Overweight"
        elif result<25 and result>=18.5:
            bmi="Healthy"
        else:
            bmi="Underweight"
        result = ""+str(result)+"("+bmi+")"
    else:
        result="Enter valid data"
    return render_template('predict.html',**locals())

if __name__=="__main__":
    app.run(debug=True)
