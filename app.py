import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from datetime import datetime , timedelta
import pandas as pd
import pycaret
import requests
from wwo_hist import retrieve_hist_data
from pycaret.regression import *

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    print(int_features)
    date = int_features[0]
    converted_datetime=pd.to_datetime(date).date()

    def dateaddition (converted_datetime):
        if int(str(converted_datetime).split("-")[0])%4 ==0:
            if (int(str(converted_datetime).split("-")[1])) ==2:
                added_days=29
            elif int(str(converted_datetime).split("-")[1]) %2 ==0 and int(str(converted_datetime).split("-")[1]) <=6:   
                added_days=30
            elif int(str(converted_datetime).split("-")[1]) %2 ==1 and int(str(converted_datetime).split("-")[1]) <=6:
                added_days=31
            elif  int(str(converted_datetime).split("-")[1]) %2 ==0 and int(str(converted_datetime).split("-")[1]) >=7:
                added_days=31
            elif int(str(converted_datetime).split("-")[1]) %2 ==1 and int(str(converted_datetime).split("-")[1]) >=7:
                added_days=30  
        elif int(str(converted_datetime).split("-")[0])%4 !=0:
            if (int(str(converted_datetime).split("-")[1])) ==2:
                added_days=28
            elif int(str(converted_datetime).split("-")[1]) %2 ==0 and int(str(converted_datetime).split("-")[1]) <=6:   
                added_days=30
            elif int(str(converted_datetime).split("-")[1]) %2 ==1 and int(str(converted_datetime).split("-")[1]) <=6:
                added_days=31
            elif  int(str(converted_datetime).split("-")[1]) %2 ==0 and int(str(converted_datetime).split("-")[1]) >=7:
                added_days=31
            elif int(str(converted_datetime).split("-")[1]) %2 ==1 and int(str(converted_datetime).split("-")[1]) >=7:
                added_days=30  

        return added_days


    end_date= pd.to_datetime(converted_datetime+timedelta(dateaddition(converted_datetime)-1)).date().strftime("%d-%b-%Y")


    start_date = converted_datetime.strftime("%d-%b-%Y")

    print(start_date,"   ",end_date)





    frequency = 24
    start_date = start_date
    end_date = end_date
    api_key = '12b2c18a34194a8ca93113127200405'
    location_list = ['Mlawa']
    hist_weather_data = retrieve_hist_data(api_key,
                                            location_list,
                                            start_date,
                                            end_date,
                                            frequency,
                                            location_label = False,
                                            export_csv = True,
                                            store_df = True)

    
    monthly_weather_data=pd.read_csv("Mlawa.csv")

    monthly_weather_data=monthly_weather_data.drop(["date_time","totalSnow_cm","sunHour","uvIndex.1","uvIndex","moon_illumination","moonrise","moonset","sunrise","DewPointC","sunset","WindChillC","WindGustKmph","precipMM","pressure","visibility","winddirDegree","windspeedKmph","tempC"],axis=1)


    monthly_weather_data["avg_temp"]=(monthly_weather_data["maxtempC"]+monthly_weather_data["mintempC"])/2

    monthly_weather_data=monthly_weather_data.drop(["maxtempC","mintempC"],axis=1)

    monthly_weather_data=monthly_weather_data[["avg_temp","FeelsLikeC","HeatIndexC","cloudcover","humidity",]]
    # monthly_weather_data=monthly_weather_data[["avg_temp","FeelsLikeC","HeatIndexC","cloudcover","humidity","ishol/week"]]

    

    def cat_heat(heatindex):
        
        if heatindex < -2:
            return (0)
        elif heatindex >=-1 and heatindex<=14:
            return (1)
        else:
            return (2)
        



    def cat_cloud(cloudcover):
        # monthly_cloudcover=[]
        # for cloudcover in data["cloudcover"]:
        if cloudcover < 25:
            return(0)
        elif cloudcover >=25 and cloudcover<50:
            return(1)
        elif cloudcover >=50 and cloudcover<75:
            return(1)    
        else:
            return(3)
        
            

            

    def mean_data(data):
        values=[]
        for key,value in data.iteritems():
            values.append(value.mean())
            print(key)
            return values

    monthly_averages=np.around(mean_data(monthly_weather_data),2)
    monthly_averages[2]=cat_heat(monthly_averages[2])
    monthly_averages[3]=cat_cloud(monthly_averages[3])
        
    print(monthly_averages)

    


    loaded_model=load_model("Final_Mod")
    unseen_data=pd.read_csv("GROUP_OF_DATASETS/SWEETS.csv")

    prediction=predict_model(loaded_model, data= unseen_data.head(5))

    output = prediction
    print(jsonify(output))
    s=""

    for row in output:
        s+='Quantity of product {} predicted is  {}\n'.format(row['name'],row['Label'])


    return render_template('index.html', prediction_text=s)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    
    date=data.Date #enter the month and year for which you want to predict the sales of the product [EXAMPLE : 02-2020 (FEB 2020)]: "

    converted_datetime=pd.to_datetime(date).date()

    def dateaddition (converted_datetime):
        if int(str(converted_datetime).split("-")[0])%4 ==0:
            if (int(str(converted_datetime).split("-")[1])) ==2:
                added_days=29
            elif int(str(converted_datetime).split("-")[1]) %2 ==0 and int(str(converted_datetime).split("-")[1]) <=6:   
                added_days=30
            elif int(str(converted_datetime).split("-")[1]) %2 ==1 and int(str(converted_datetime).split("-")[1]) <=6:
                added_days=31
            elif  int(str(converted_datetime).split("-")[1]) %2 ==0 and int(str(converted_datetime).split("-")[1]) >=7:
                added_days=31
            elif int(str(converted_datetime).split("-")[1]) %2 ==1 and int(str(converted_datetime).split("-")[1]) >=7:
                added_days=30  
        elif int(str(converted_datetime).split("-")[0])%4 !=0:
            if (int(str(converted_datetime).split("-")[1])) ==2:
                added_days=28
            elif int(str(converted_datetime).split("-")[1]) %2 ==0 and int(str(converted_datetime).split("-")[1]) <=6:   
                added_days=30
            elif int(str(converted_datetime).split("-")[1]) %2 ==1 and int(str(converted_datetime).split("-")[1]) <=6:
                added_days=31
            elif  int(str(converted_datetime).split("-")[1]) %2 ==0 and int(str(converted_datetime).split("-")[1]) >=7:
                added_days=31
            elif int(str(converted_datetime).split("-")[1]) %2 ==1 and int(str(converted_datetime).split("-")[1]) >=7:
                added_days=30  

        return added_days


    end_date= pd.to_datetime(converted_datetime+timedelta(dateaddition(converted_datetime)-1)).date().strftime("%d-%b-%Y")


    start_date = converted_datetime.strftime("%d-%b-%Y")

    print(start_date,"   ",end_date)





    frequency = 24
    start_date = start_date
    end_date = end_date
    api_key = '12b2c18a34194a8ca93113127200405'
    location_list = ['Mlawa']
    hist_weather_data = retrieve_hist_data(api_key,
                                            location_list,
                                            start_date,
                                            end_date,
                                            frequency,
                                            location_label = False,
                                            export_csv = True,
                                            store_df = True)

    
    monthly_weather_data=pd.read_csv("Month_Weather_data.csv")

    monthly_weather_data=monthly_weather_data.drop(["date_time","totalSnow_cm","sunHour","uvIndex.1","uvIndex","moon_illumination","moonrise","moonset","sunrise","DewPointC","sunset","WindChillC","WindGustKmph","precipMM","pressure","visibility","winddirDegree","windspeedKmph","tempC"],axis=1)


    monthly_weather_data["avg_temp"]=(monthly_weather_data["maxtempC"]+monthly_weather_data["mintempC"])/2

    monthly_weather_data=monthly_weather_data.drop(["maxtempC","mintempC"],axis=1)

    monthly_weather_data=monthly_weather_data[["avg_temp","FeelsLikeC","HeatIndexC","cloudcover","humidity",]]
    # monthly_weather_data=monthly_weather_data[["avg_temp","FeelsLikeC","HeatIndexC","cloudcover","humidity","ishol/week"]]

    

    def cat_heat(heatindex):
        
        if heatindex < -2:
            return (0)
        elif heatindex >=-1 and heatindex<=14:
            return (1)
        else:
            return (2)
        



    def cat_cloud(cloudcover):
        # monthly_cloudcover=[]
        # for cloudcover in data["cloudcover"]:
        if cloudcover < 25:
            return(0)
        elif cloudcover >=25 and cloudcover<50:
            return(1)
        elif cloudcover >=50 and cloudcover<75:
            return(1)    
        else:
            return(3)
        
            

            

    def mean_data(data):
        values=[]
        for key,value in data.iteritems():
            values.append(value.mean())
            print(key)
            return values

    monthly_averages=np.around(mean_data(monthly_weather_data),2)
    monthly_averages[2]=cat_heat(monthly_averages[2])
    monthly_averages[3]=cat_cloud(monthly_averages[3])
        
    print(monthly_averages)

    


    loaded_model=load_model("Final_Mod")
    unseen_data=pd.read_csv("GROUP_OF_DATASETS/SWEETS.csv")

    prediction=predict_model(loaded_model, data= unseen_data.head(5))

    output = prediction
    print(jsonify(output))

    return jsonify(output)


    

if __name__ == "__main__":
    app.run(debug=True)


