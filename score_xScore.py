import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from expected_score_model.config import *
from expected_score_model.domain.preprocessing.preprocessing import expected_score_response_processing, split_shots

# Load chain data
chain_data = pd.read_csv(chain_file_path, low_memory=False)

# Preprocess
chain_data = expected_score_response_processing(chain_data)
df_set_shots, df_open_shots = split_shots(chain_data)

goal_set_preproc = joblib.load(exp_goal_set_preprocessor_file_path)
behind_set_preproc = joblib.load(exp_behind_set_preprocessor_file_path)
miss_set_preproc = joblib.load(exp_miss_set_preprocessor_file_path)
goal_open_preproc = joblib.load(exp_goal_open_preprocessor_file_path)
behind_open_preproc = joblib.load(exp_behind_open_preprocessor_file_path)
miss_open_preproc = joblib.load(exp_miss_open_preprocessor_file_path)

set_goal_features = goal_set_preproc.transform(chain_data)
set_behind_features = behind_set_preproc.transform(chain_data)
set_miss_features = miss_set_preproc.transform(chain_data)
open_goal_features = goal_open_preproc.transform(chain_data)
open_behind_features = behind_open_preproc.transform(chain_data)
open_miss_features = miss_open_preproc.transform(chain_data)

# Load models
expected_goal_set_model = joblib.load(exp_goal_set_model_file_path)
expected_behind_set_model = joblib.load(exp_behind_set_model_file_path)
expected_miss_set_model = joblib.load(exp_miss_set_model_file_path)

expected_goal_open_model = joblib.load(exp_goal_open_model_file_path)
expected_behind_open_model = joblib.load(exp_behind_open_model_file_path)
expected_miss_open_model = joblib.load(exp_miss_open_model_file_path)

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

# Merge xScore to Chain
chain_data = chain_data.merge(df_shots[['Match_ID', "Chain_Number", "Order", 'xGoals', 'xBehinds', 'xMiss', 'xGoals_normalised', 'xBehinds_normalised', 'xMiss_normalised', 'xScore']], how = "left", on = ['Match_ID', "Chain_Number", "Order"])

# Export chain
chain_data.to_csv(xScore_chain_output_path, index=False)