from dataclasses import dataclass

@dataclass
class ModellingDataContract:
    """ Holds details for defining modelling terms in a single place.
    """
    
    ID_COL = "Match_ID"
    RESPONSE_GOAL = "Goal"
    RESPONSE_BEHIND = "Behind"
    RESPONSE_MISS = "Miss"
    RESPONSE_MULTICLASS = "Kick_Outcome"
    
    GOAL_TRAINING_SET = "GoalTrainingSet"
    BEHIND_TRAINING_SET = "BehindTrainingSet"
    MISS_TRAINING_SET = "MissTrainingSet"
        
    team_list = [
        'Adelaide',
        'Brisbane Lions',
        'Carlton',
        'Collingwood',
        'Essendon',
        'Fremantle',
        'Geelong',
        'Gold Coast',
        'Greater Western Sydney',
        'Hawthorn',
        'Melbourne',
        'North Melbourne',
        'Port Adelaide',
        'Richmond',
        'St Kilda',
        'Sydney',
        'West Coast',
        'Western Bulldogs'
    ]
    set_goal_modelling_feature_list = [
        'x0',
        'Distance_to_Middle_Goal',
        'Angle_to_Middle_Goal',
        'Visible_Goal_Angle',
        'ballUp', 
        'centreBounce', 
        'kickIn', 
        'possGain', 
        'throwIn'
    ]
    open_goal_modelling_feature_list = [
        'x0',
        'x1',
        'x2',
        'x3',
        'Distance_Since_Last_Action',
        'Distance_to_Middle_Goal',
        'Angle_to_Middle_Goal',
        'Visible_Goal_Angle',
        'ballUp',
        'centreBounce',
        'kickIn',
        'possGain',
        'throwIn',
        'Distance_to_Right_Goal_x',
        'Distance_to_Middle_y',
        'Chain_Duration',
        'Time_Since_Last_Action'
    ]
    
    expected_score_modelling_feature_list = [
        'x0',
        'y0',
        'Distance_to_Middle_Goal',
        'Angle_to_Middle_Goal',
        'Visible_Goal_Angle',
        'Set_Shot'
    ]
    
    monotone_constraints_set_goal = {
        'Distance_to_Middle_Goal':-1,
        'Angle_to_Middle_Goal':-1,
        'Visible_Goal_Angle':1,
        'x0':-1,
    }
    monotone_constraints_open_goal = {
        'Angle_to_Middle_Goal': -1,
        'Distance_Since_Last_Action': 1,
        'Distance_to_Middle_Goal': -1,
        'Visible_Goal_Angle': 1, 
        'x0': -1
    }
    
    monotone_constraints_score = {
        'x0': -1,
        'Distance_to_Middle_Goal': -1,
        'Angle_to_Middle_Goal': -1,
        'Visible_Goal_Angle': 1, 
    }