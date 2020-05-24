# PG-StockManagement Model 

**P&G** is one of the largest and amongst the fastest-growing consumer goods companies globally.

# Predictive Analysis of Stock

This project aims at enabling **real-time and even predictive out of stock in retail stores in a cost-effective manner** by predicting the quantity of sales of each product in that month.

### Theme: Artificial Intelligence and Machine Learning

## STOCK MANAGEMENT

- Since **competition** is increasing day by day among retailers at the market, **companies are focusing more predictive analytics techniques** in order to **decrease their costs** and **increase their productivity and profit**. 

- Excessive stocks (overstock) and out- of-stock (stockouts) are very serious problems for retailers.




## The Problem 

- Excessive stock levels can cause **revenue loss** because of company capital bound to stock surplus. 
- Excess inventory can also lead to **increased storage, labor, and insurance costs, and quality reduction and degradation** depending on the type of the product. 
- Out-of-stock products can result in **loss for sales** and **reduced customer satisfaction and store loyalty**.


## Solution

- Sales and customer loss is a critical problem for retailers. Considering competition and financial constraints in the retail industry, it is very crucial to have an **accurate demand forecasting and inventory control system for management of effective operations**. 

- Here comes **predictive analytics** where the **out-of-stock predictions are more accurate**.

- Predictive analytics is the utilization of information, factual calculations and machine learning methods to distinguish the probability of future results in light of chronicled information.


## Weather Predictive Stock Fulfilment

- **Weather** plays a **crucial** role in the **accurate predictiveness of the stock**.

- Weather influences the **consumer psychology, habits, product preference, and behavior**. It affects four basic purchasing decisions: **what, where, when and in what quantity** to buy. 

- Results of studies show that weather has significant effect on store traffic and sales of many product categories and store types. 

- If companies are informed about the changes in weather ahead of time, it adds predictability in sales with the application of relevant trade promotions. After all, small changes to retail sales would increase revenues by a large percentage.


## Idea and Implementation:

- With the help of weather predictive stock fulfillment, stocking and early purchase would be more meaningful. Combining historical data with past and predicted weather patterns we  can project the levels of demand and correct levels of supply. This adds more efficiency and accuracy to the planning process and eliminates gut-based planning.

- We have used various time series and regression techniques like Exponential smoothing model and SVR ( Support Vector Regression) and Holt-Trend Method,
Holt-Winters Seasonal Models will be used. 

- Logistic regression , Lasso Regression and ElasticNet Regression models will be used. A multilayer feedforward artificial neural network (MLFANN) is employed as a deep learning algorithm to train the data as well.

- The output from each model is then integrated together using decision integration. This approach is being used by the way of combining the strengths of different algorithms into a single collaborated method philosophy. 

- An ensemble system will be used  to give better accuracy. We use an integration technique so as to predict the best model out of all. ie. We shall use boosting as it predicts the best algorithm as the weights of each model( which get assigned to each model) are changed depending on the output of the previously acquired knowledge. 


## FINAL RESULT:

The final result is a **number of units sold for each product of each category** so as to know and procure the required goods. 


## Dataset

The Dataset we have is **data of sale of various FMCG goods of a SuperMarket** located in a city in Poland with a population of 30000 people. The store is located in the prime locality of the city and offer various products like general food-and basic chemistry, hygienic articles, fresh bread,sweets, local vegetables, dairy, basic meat(ham,sausages), newspaper, home chemistry etc. 

- The nearest competition is a small grocery store and sells the same products as this SuperMarket. The area of the shop is 120m^2 and was opened in 2009. 

- The Data deals with products sold  in a time period of 12 days over various weather conditions for each particular day. It has 13000 entries ranging over various product descriptions like price, product number, quantity sold etc.

- We predict the number of products sold for each category everyday based on this data. The inputs taken by the model would be the weather and the probable quantity sold each day would be the final output. This would lead to better predicting of stock and easy ordering of the necessary goods. 


## Flowchart

A brief flowchart showing each step in the model building process.



## Training, Testing and Deployment


The final decision of the system is based on the best algorithms of  the day by gaining more weight. This makes our forecasts more reliable with respect to the trend changes and seasonality behavior.



## TECHNOLOGIES USED 

- To implement the regression models and select the best performing one, we have used **pycaret** library in **python**

- We used the **wwo-hist API** to procure weather data for our dataset dates.

- **Pickle** was used to save the model and **Flask and REST API** to create an interface for the model.

- Finally the web application was deployed on the server using **Heroku CLI.**



## WORKFLOW

### Data Analysis


- First we analyse the data to understand its structure

- The dataset has several columns. We check the corelations of these coumns with quantity and remove the unrequired ones.
 
- We further scale the data to remove its skewness.

- We add a column separately to check whether that day is a weekday or holiday. This affects the sales predictions significantly.

- We change the categorical values to numeric. 


### Weather Data API


- We use the wwo-hist historical weather data retrieval  API.

- It is an API developed by World Weather online and gives the weather data for a specific location from a start date to end date.
 
- The api returns a csv of date and their respective weather conditions like temperature, precipitation, cloudcover and a lot more.

- We merge the obtained csv date wise to our dataset and again visualise the data to analyse the corelations between weather data and sales.


### PYCARET TO TRAIN SEVERAL REGRESSION MODELS


We train several regression models. 
 
They include 
- Random Forest
- Extreme Gradient boosting
- CatBoost Regressor
- Ridge Regression
- Bayesian Ridge
- Linear Regression
- Random Sample consensus
- Orthogonal Matching Pursuit
- Lasso Regression
- Extra trees Regessor
- Elastic Net
- K neighbors Regressor
- Support vector machine
- Decision Tree
- AdaBoost Regressor
- Lasso Least Angle Regression
- Gradient Boostinf regressor 



## REAL WORLD SCENARIO

###### Working of Prototype in Real World Scenario

- Small changes to retail sales would increase revenues by a large percentage.

- We can train the models on Data from inventories oF P&G over a certain time period. Using that we can obtain the predictions for each quantity in a separate group for the next month or later.

- The model will give better results when we have a lot more focused data.


## Benefits

- It’s better to utilize real-time information to move stock where it's required before it's past the point of no return. 

- Furthermore, utilizing predictive analytics to choose what to stock and where in light of information about provincial contrasts in inclinations, climate, and so on.
 
- Retail has moved toward becoming as much about envisioning clients' needs as it is about just stocking decent items.

- Real-time and predictive out of stock in retail stores in a cost-effective manner.


## DEMO VIDEO

Click here to view DEMO VIDEO : https://youtu.be/oqBa5oczFZo

<img src="https://github.com/sbis04/Healthcare-poseAI/blob/master/PoseAI%20images/screen3.jpg" height="500"  alt="Screenshot"/>


## TEST IT LIVE!

- Our web app is live on heroku.

- Enter the date in this format : Eg Feb 2020 should be entered as 02-2020

- Enter the category from this list of categories we trained our model n. Each category has a lot of products.

- The web application outputs the names and predicted quantites for every product in that group.

#### Click Here to test it live! : pgstockprediction.herokuapp.com





