import joblib
import warnings
warnings.filterwarnings('ignore')

def predict_xscore_from_chains(chains):
    
    chains = chains.reset_index(drop=True)

    xscore_preproc = joblib.load("model_outputs/preprocessors/xscore_preprocessor_25.joblib")
    shots_features = xscore_preproc.transform(chains)

    xscore_model = joblib.load("model_outputs/models/catboost_xscore_model_25.joblib")   
    model_features = xscore_model.cb_clf.feature_names_
    
    shots = chains.loc[shots_features.index]
    shots['predicted_result'] = xscore_model.predict(shots_features[model_features]).flatten()
    shots[[f'{x}_probas' for x in xscore_model.classes_]] = xscore_model.predict_proba(shots_features[model_features])
    shots['xscore'] = shots['goal_probas']*6 + shots['behind_probas']
    shots = shots.merge(shots_features, left_index=True, right_index=True)

    return shots