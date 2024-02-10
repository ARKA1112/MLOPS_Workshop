import pickle
import sklearn
import mlflow
from mlflow.client import MlflowClient
from flask import Flask, request, jsonify

#declare the run_id

RUN_ID = "7f6c9b37be7041d89ed1abc9d4f14836"
MLFLOW_TRACKING_URI = 'http://localhost:5000'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
#----download the artifact

path = client.download_artifacts(run_id=RUN_ID,path='dv.bin')

with open(path,'rb') as f_out:
    dv=pickle.load(f_out)
print(f"downloading the dict vectorizer in {path}")
##############################
logged_model = f'runs:/{RUN_ID}/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
# import pandas as pd
# loaded_model.predict(pd.DataFrame(data))

############################

def prepare_data(ride):
    features = {}
    features["PU_DO"] = '%s_%s' % (ride['PULocationID'] , ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform(features)
    preds = loaded_model.predict(X)
    return preds[0]


app = Flask('Web_application')


@app.route("/predict",methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_data(ride)
    pred = predict(features)


    result = {
        'duration': pred
    }

    return jsonify(result)



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
    