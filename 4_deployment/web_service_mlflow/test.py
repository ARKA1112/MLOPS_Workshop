import requests


ride = {
    'PULocationID': 40,
    'DOLocationID': 100,
    'trip_distance': 40
}

ride = dict(ride)

url = "http://localhost:9696/predict"
response = requests.post(url, json=ride)
print(response.json())