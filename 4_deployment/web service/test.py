import requests


ride = {
    'PULocationID': 10,
    'DOLocationID': 30,
    'trip_distance': 40
}

ride = dict(ride)

url = "http://localhost:9696/predict"
response = requests.post(url, json=ride)
print(response.json())