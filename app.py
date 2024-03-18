from expected_score_model.predict import predict_xscore
from AFLPy.AFLData_Client import load_data, upload_data

from flask import Flask, request

app = Flask(__name__)

@app.route("/model/expectedscore/predict", methods=["GET", "POST"])
def predict(ID = None):
    
    # Load chain data
    chains = load_data(Dataset_Name='AFL_API_Match_Chains', ID = request.json['ID'])

    shots = predict_xscore(chains)
    
    # upload_data(Dataset_Name="CG_Expected_Score", Dataset=shots, overwrite=True, update_if_identical=True)
    
    return shots.to_json(orient='records')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=False)