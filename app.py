from flask import Flask, request, jsonify, render_template, session, url_for, redirect
from flask_wtf import FlaskForm
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from wtforms import TextField, SubmitField

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'


class FlowerForm(FlaskForm):

    sepal_length = TextField('Sepal Length')
    sepal_width = TextField('Sepal Width')
    petal_length = TextField('Petal Length')
    petal_width = TextField('Petal Width')
    submit = SubmitField("Predict")

def return_prediction(model,scaler,sample_json):
    flower = list()

    for i in sample_json:
        flower.append(sample_json[i])
    flower = [flower]
    print(flower)
    flower = scaler.transform(flower)
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    class_index = model.predict_classes(flower)

    return classes[class_index][0]

flower_model = load_model('final_iris_model.h5')
flower_scaler = joblib.load('iris_scaler.pkl')

@app.route("/prediction/")
def prediction():
    content = {}
    content['sepal_length'] = float(session['sepal_length'])
    content['sepal_width'] = float(session['sepal_width'])
    content['petal_width'] = float(session['petal_width'])
    content['petal_length'] = float(session['petal_length'])

    results = return_prediction(flower_model, flower_scaler, content)

    return render_template('prediction.html', results = results)




@app.route("/", methods = ['GET', 'POST'])
def index():
    form = FlowerForm()

    if form.validate_on_submit():
        session["sepal_length"] = form.sepal_length.data
        session["sepal_width"] = form.sepal_width.data
        session["petal_length"] = form.petal_length.data
        session["petal_width"] = form.petal_width.data

        return redirect(url_for("prediction"))
    else:
        return render_template('home.html', form=form)

if __name__ == '__main__':
    app.run()
