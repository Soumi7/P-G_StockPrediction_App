import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Date':2, 'Product name':9, 'Product Group':6})

print(r.json())