import predict


ride ={
    "PULocationID": 10,
    "DOLocationID": 30,
    "trip_distance": 40,
}


features = predict.prepare_data(ride)
pred = predict.predict(features)
print(pred[0])