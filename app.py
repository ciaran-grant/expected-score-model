from expected_score_model.predict import predict_xscore
from AFLPy.AFLData_Client import upload_data

from flask import Flask, request

app = Flask(__name__)

@app.route("/model/expectedscore/predict", methods=["GET", "POST"])
def predict(ID = None):

    shots = predict_xscore(ID = request.json['ID'])
    
    upload_data(Dataset_Name="CG_Expected_Score", Dataset=shots, overwrite=True, update_if_identical=True)
    
    return shots.to_json(orient='records')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)