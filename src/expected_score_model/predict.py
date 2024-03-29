import pandas as pd
import numpy as np
import joblib
from expected_score_model.domain.preprocessing.preprocessing import expected_score_response_processing, split_shots

def predict_xscore(chains):

    # Preprocess
    chains['Quarter_Duration'] = chains['Period_Duration']
    chains['Quarter_Duration_Chain_Start'] = chains['Period_Duration_Chain_Start']
    chains['Shot_At_Goal'] = np.where(chains['Shot_At_Goal'] == "TRUE", True, False)

    chains = expected_score_response_processing(chains)
    df_set_shots, df_open_shots = split_shots(chains)

    goal_set_preproc = joblib.load("model_outputs/preprocessors/set_goal_preproc.joblib")
    behind_set_preproc = joblib.load("model_outputs/preprocessors/set_behind_preproc.joblib")
    miss_set_preproc = joblib.load("model_outputs/preprocessors/set_miss_preproc.joblib")
    
    goal_open_preproc = joblib.load("model_outputs/preprocessors/open_goal_preproc.joblib")
    behind_open_preproc = joblib.load('model_outputs/preprocessors/open_behind_preproc.joblib')
    miss_open_preproc = joblib.load("model_outputs/preprocessors/open_miss_preproc.joblib")

    set_goal_features = goal_set_preproc.transform(chains)
    set_behind_features = behind_set_preproc.transform(chains)
    set_miss_features = miss_set_preproc.transform(chains)
    open_goal_features = goal_open_preproc.transform(chains)
    open_behind_features = behind_open_preproc.transform(chains)
    open_miss_features = miss_open_preproc.transform(chains)

    # Load models
    expected_goal_set_model = joblib.load("model_outputs/models/expected_goal_set.joblib")
    expected_behind_set_model = joblib.load("model_outputs/models/expected_behind_set.joblib")
    expected_miss_set_model = joblib.load("model_outputs/models/expected_miss_set.joblib")

    expected_goal_open_model = joblib.load("model_outputs/models/expected_goal_open.joblib")
    expected_behind_open_model = joblib.load("model_outputs/models/expected_behind_open.joblib")
    expected_miss_open_model = joblib.load("model_outputs/models/expected_miss_open.joblib")

    # Score models
    df_set_shots['xGoals'] = expected_goal_set_model.predict_proba(set_goal_features, calibrate=True)
    df_set_shots['xBehinds'] = expected_behind_set_model.predict_proba(set_behind_features, calibrate=True)
    df_set_shots['xMiss'] = expected_miss_set_model.predict_proba(set_miss_features, calibrate=True)

    df_open_shots['xGoals'] = expected_goal_open_model.predict_proba(open_goal_features, calibrate=True)
    df_open_shots['xBehinds'] = expected_behind_open_model.predict_proba(open_behind_features, calibrate=True)
    df_open_shots['xMiss'] = expected_miss_open_model.predict_proba(open_miss_features, calibrate=True)

    # Expected Score
    df_shots = pd.concat([df_set_shots, df_open_shots], axis=0)
    df_shots = df_shots.sort_values(by = ['Match_ID', "Chain_Number", "Order"])

    df_shots['xGoals_normalised'] = df_shots['xGoals'] / (df_shots['xGoals'] + df_shots['xBehinds'] + df_shots['xMiss'])
    df_shots['xBehinds_normalised'] = df_shots['xBehinds'] / (df_shots['xGoals'] + df_shots['xBehinds'] + df_shots['xMiss'])
    df_shots['xMiss_normalised'] = df_shots['xMiss'] / (df_shots['xGoals'] + df_shots['xBehinds'] + df_shots['xMiss'])

    df_shots['xScore'] = df_shots['xGoals_normalised']*6 + df_shots['xBehinds_normalised']

    return df_shots