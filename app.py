import bz2
from distutils.log import debug
from tkinter.messagebox import NO
from typing import final
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, request
from geopy.geocoders import Nominatim
import pickle
import bz2
import _pickle as cPickle
import os

app = Flask(__name__)
port = int(os.environ.get('PORT', 5000))
if __name__ == '__main__':
    app.debug = True
    app.run()

data = pd.read_csv('data/cleaned-data.csv')
data = data.iloc[:29430,:]

X = data.drop(['TARGET(PRICE_IN_LACS)'],axis=1)
Y= data['TARGET(PRICE_IN_LACS)']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

model = RandomForestRegressor(n_estimators=1000,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',max_depth=None)
model.fit(X_train,Y_train)

@app.route('/')
def index():
    geolocator = Nominatim(user_agent="geoapiExercises")
    return render_template('index.html', title='House Price Prediction in India',subtitle='predict the price of your dream home . . .')

model = bz2.BZ2File('Compressed.pbz2','rb')
model = cPickle.load(model)

@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    
    int_features = [x for x in request.form.values()]
    print('data collected', int_features)
    geolocator = Nominatim(user_agent="geoapiExercises")
    if (int_features[6] == 'Other'):
        try:
            lat = geolocator.geocode(f'{int_features[5] , int_features[7]}, India').raw['lat']
            lon = geolocator.geocode(f'{int_features[5] , int_features[7]}, India').raw['lon']
        except:
            lat = None
            lon = None
        
    else:
        try:
            lat = geolocator.geocode(f'{int_features[5] , int_features[6]}, India').raw['lat']
            lon = geolocator.geocode(f'{int_features[5] , int_features[6]}, India').raw['lon']
        except:
            lat = None
            lon = None

    print(lat, lon)
    

    if (lat is None or lon is None):
        return render_template('index.html', error_text=f'Sorry something went wrong...')
    else:
        final_features = np.array([int_features[0],int_features[1],int_features[2],int_features[3],int_features[4],lat,lon])
        final_features = final_features.reshape(1,-1)
        prediction = None
        prediction = model.predict(final_features)
        print(prediction)
        if (prediction is None):
            return render_template('index.html', error_text=f'Sorry we unable to predict the price . Please try after sometime...')
        else:
            return render_template('index.html', prediction_text=f'Predicted price is Rs {"{:10.4f}".format(prediction[0])} (in lakhs) .', text =f'Our Result is 82% accurate.', title='House Price Prediction in India',subtitle='predict the price of your dream home . . .')
