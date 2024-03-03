from expected_score_model.predict import predict_xscore
from AFLPy.AFLData_Client import upload_data

from flask import Flask, request

app = Flask(__name__)


@app.route("/model/expectedscore/predict", methods=["GET", "POST"])
def predict(ID = None):
    data = request.json
        
    shots = predict_xscore(data)
    
    upload_data(Dataset_Name="CG_Expected_Score", Dataset=shots, overwrite=True)
    
    return shots.to_json(orient='records')
