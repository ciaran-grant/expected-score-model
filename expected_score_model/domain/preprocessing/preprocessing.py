import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

final_state_map = {
    'goal':'goal',
    'behind':'behind',
    'turnover':'miss',
    'rushed':'miss',
    'outOfBounds':'miss',
    'ballUpCall':'miss',
    'endQuarter':'miss',
    'rushedOpp':'miss',
}

def expected_score_response_processing(chain_data):
    
    chain_data['Final_State'] = chain_data['Final_State'].replace(final_state_map)

    chain_data['Goal'] = np.where((chain_data['Shot_At_Goal'] == True) & (chain_data['Final_State'] == "goal"), 1, 0)
    chain_data['Behind'] = np.where((chain_data['Shot_At_Goal'] == True) & (chain_data['Final_State'] == "behind"), 1, 0)
    chain_data['Miss'] = np.where((chain_data['Shot_At_Goal'] == True) & (chain_data['Final_State'] == "miss"), 1, 0)

    chain_data['Score'] = np.where(chain_data['Goal']==1, 6,
                                np.where(chain_data['Behind']==1, 1, 
                                            0))
    
    return chain_data

def split_shots(chain_data):
    
    chain_data['Event_Type1'] = chain_data['Description'].shift(1)
    df_shots = chain_data[chain_data['Shot_At_Goal'] == True]
    df_shots['Set_Shot'] = df_shots['Event_Type1'].apply(lambda x: ("Mark" in x) or ("Free" in x))
    df_set_shots = df_shots[df_shots['Set_Shot']]
    df_open_shots = df_shots[~df_shots['Set_Shot']]
    
    return df_set_shots, df_open_shots

def expected_score_feature_engineering(chain_data):
        
    chain_data['Event_Type0'] = chain_data['Description']
    chain_data['Event_Type1'] = chain_data['Description'].shift(1)
    chain_data['Event_Type2'] = chain_data['Description'].shift(2)
    chain_data['Event_Type3'] = chain_data['Description'].shift(3)

    chain_data['x0'] = chain_data['x']
    chain_data['x1'] = chain_data['x'].shift(1)
    chain_data['x2'] = chain_data['x'].shift(2)
    chain_data['x3'] = chain_data['x'].shift(3)

    chain_data['y0'] = chain_data['y']
    chain_data['y1'] = chain_data['y'].shift(1)
    chain_data['y2'] = chain_data['y'].shift(2)
    chain_data['y3'] = chain_data['y'].shift(3)

    chain_data['Quarter_Duration0'] = chain_data['Quarter_Duration']
    chain_data['Quarter_Duration1'] = chain_data['Quarter_Duration'].shift(1)
    chain_data['Quarter_Duration2'] = chain_data['Quarter_Duration'].shift(2)
    chain_data['Quarter_Duration3'] = chain_data['Quarter_Duration'].shift(3)
    
    chain_data['Time_Since_Last_Action'] = chain_data['Quarter_Duration0'] - chain_data['Quarter_Duration1']
    chain_data['Distance_Since_Last_Action'] = ((chain_data['x1'] - chain_data['x0'])**2 + (chain_data['y1'] - chain_data['y0'])**2)**0.5
    
    chain_data['Chain_Duration'] = chain_data['Quarter_Duration'] - chain_data['Quarter_Duration_Chain_Start']
    
    chain_data['Distance_to_Right_Goal_x'] = chain_data['Venue_Length']/2 - chain_data['x0']
    chain_data['Distance_to_Middle_y'] = abs(chain_data['y0'])

    chain_data['Distance_to_Middle_Goal'] = (chain_data['Distance_to_Right_Goal_x']**2 + chain_data['Distance_to_Middle_y']**2)**0.5
    chain_data['Angle_to_Middle_Goal'] = np.arctan2(chain_data['Distance_to_Middle_y'], chain_data['Distance_to_Right_Goal_x'])
    chain_data['Angle_to_Middle_Goal_degrees'] = np.degrees(chain_data['Angle_to_Middle_Goal'])

    chain_data['Visible_Goal_Angle'] = (6.4*chain_data['Distance_to_Right_Goal_x']) / (chain_data['Distance_to_Right_Goal_x']**2 + chain_data['Distance_to_Middle_y']**2-(6.4/2)**2)
    chain_data['Visible_Goal_Angle_degrees'] = np.degrees(chain_data['Visible_Goal_Angle'])

    chain_data['Visible_Behind_Angle'] = ((6.4*3)*chain_data['Distance_to_Right_Goal_x']) / (chain_data['Distance_to_Right_Goal_x']**2 + chain_data['Distance_to_Middle_y']**2-((3*6.4)/2)**2)
    chain_data['Visible_Behind_Angle_degrees'] = np.degrees(chain_data['Visible_Behind_Angle'])
    
    chain_data['Squared_Distance_to_Middle_Goal'] = chain_data['Distance_to_Right_Goal_x']**2
    chain_data['Log_Distance_to_Middle_Goal'] = np.log(chain_data['Distance_to_Right_Goal_x'])
    
    return chain_data


def get_stratified_train_test_val_columns(data, response):
        
    X, y = data.drop(columns=[response]), data[response]
    X_modelling, X_test, y_modelling, y_test = train_test_split(X, y, test_size = 0.2, random_state=2407)
    X_train, X_val, y_train, y_val = train_test_split(X_modelling, y_modelling, test_size = 0.2, random_state=2407)
    X_train[response+'TrainingSet'] = True
    X_test[response+'TestSet'] = True
    X_val[response+'ValidationSet'] = True
    
    if [response+'TrainingSet', response+'TestSet', response+'ValidationSet'] not in list(data):
        data = pd.merge(data, X_train[response+'TrainingSet'], how="left", left_index=True, right_index=True) 
        data = pd.merge(data, X_test[response+'TestSet'], how="left", left_index=True, right_index=True) 
        data = pd.merge(data, X_val[response+'ValidationSet'], how="left", left_index=True, right_index=True)
        data[[response+'TrainingSet', response+'TestSet', response+'ValidationSet']] = data[[response+'TrainingSet', response+'TestSet', response+'ValidationSet']].fillna(False) 
        
    return data