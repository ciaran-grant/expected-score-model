import joblib
import pandas as pd
from expected_score_model.domain.contracts.modelling_data_contract import (
    ModellingDataContract,
)

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/predict/set_shots", methods=["POST"])
def predict_set_shots():
    data = request.json
    set_shots = pd.DataFrame(data, index=list(range(len(data))))

    initial_state_values = ["ballUp", "centreBounce", "kickIn", "possGain", "throwIn"]
    set_shots = pd.get_dummies(set_shots)
    set_shots.columns = [x.replace("Initial_State_", "") for x in list(set_shots)]
    missing_cols = list(set(initial_state_values) - set(list(set_shots)))
    for col in missing_cols:
        set_shots[col] = 0

    set_goal_features = set_shots[ModellingDataContract.feature_list_set_goal]
    set_shots["xGoals"] = goal_set_model.predict_proba(
        set_goal_features, calibrate=True
    )

    set_behind_features = set_shots[ModellingDataContract.feature_list_set_behind]
    set_shots["xBehinds"] = behind_set_model.predict_proba(
        set_behind_features, calibrate=True
    )

    set_miss_features = set_shots[ModellingDataContract.feature_list_set_miss]
    set_shots["xMiss"] = miss_set_model.predict_proba(set_miss_features, calibrate=True)

    set_shots["xGoals_normalised"] = set_shots["xGoals"] / (
        set_shots["xGoals"] + set_shots["xBehinds"] + set_shots["xMiss"]
    )
    set_shots["xBehinds_normalised"] = set_shots["xBehinds"] / (
        set_shots["xGoals"] + set_shots["xBehinds"] + set_shots["xMiss"]
    )
    set_shots["xMiss_normalised"] = set_shots["xMiss"] / (
        set_shots["xGoals"] + set_shots["xBehinds"] + set_shots["xMiss"]
    )

    set_shots["xScore"] = (
        set_shots["xGoals_normalised"] * 6 + set_shots["xBehinds_normalised"]
    )

    return jsonify({"prediction": list(set_shots["xScore"])})


@app.route("/predict/open_shots", methods=["POST"])
def predict_open_shots():
    data = request.json
    open_shots = pd.DataFrame(data, index=list(range(len(data))))

    initial_state_values = ["ballUp", "centreBounce", "kickIn", "possGain", "throwIn"]
    open_shots = pd.get_dummies(open_shots)
    open_shots.columns = [x.replace("Initial_State_", "") for x in list(open_shots)]
    missing_cols = list(set(initial_state_values) - set(list(open_shots)))
    for col in missing_cols:
        open_shots[col] = 0

    open_goal_features = open_shots[ModellingDataContract.feature_list_open_goal]
    open_shots["xGoals"] = goal_open_model.predict_proba(
        open_goal_features, calibrate=True
    )

    open_behind_features = open_shots[ModellingDataContract.feature_list_open_behind]
    open_shots["xBehinds"] = behind_open_model.predict_proba(
        open_behind_features, calibrate=True
    )

    open_miss_features = open_shots[ModellingDataContract.feature_list_open_miss]
    open_shots["xMiss"] = miss_open_model.predict_proba(
        open_miss_features, calibrate=True
    )

    open_shots["xGoals_normalised"] = open_shots["xGoals"] / (
        open_shots["xGoals"] + open_shots["xBehinds"] + open_shots["xMiss"]
    )
    open_shots["xBehinds_normalised"] = open_shots["xBehinds"] / (
        open_shots["xGoals"] + open_shots["xBehinds"] + open_shots["xMiss"]
    )
    open_shots["xMiss_normalised"] = open_shots["xMiss"] / (
        open_shots["xGoals"] + open_shots["xBehinds"] + open_shots["xMiss"]
    )

    open_shots["xScore"] = (
        open_shots["xGoals_normalised"] * 6 + open_shots["xBehinds_normalised"]
    )

    return jsonify({"prediction": list(open_shots["xScore"])})


if __name__ == "__main__":
    models_file_path = "/Users/ciaran/Documents/Projects/AFL/git-repositories/expected-score-model/model_outputs/models"

    goal_set_model = joblib.load(models_file_path + "/expected_goal_set.joblib")
    goal_open_model = joblib.load(models_file_path + "/expected_goal_open.joblib")
    behind_set_model = joblib.load(models_file_path + "/expected_behind_set.joblib")
    behind_open_model = joblib.load(models_file_path + "/expected_behind_open.joblib")
    miss_set_model = joblib.load(models_file_path + "/expected_miss_set.joblib")
    miss_open_model = joblib.load(models_file_path + "/expected_miss_open.joblib")

    app.run(host="0.0.0.0", port=8000, debug=True)
