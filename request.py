import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'R&D Spend':2, 'Administration':9, 'Marketing Spend':6})

print(r.json())