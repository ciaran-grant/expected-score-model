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
    TRAIN_TEST_SPLIT_COL = "ModellingFilter"
        
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
    
    # Set Goal
    feature_list_set_goal = [
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
    monotone_constraints_set_goal = {
        'Distance_to_Middle_Goal':-1,
        'Angle_to_Middle_Goal':-1,
        'Visible_Goal_Angle':1,
        'x0':-1,
    }
    
    # Set Behind
    feature_list_set_behind = [
        'x0',
        'y0',
        'Distance_to_Middle_Goal',
        'Angle_to_Middle_Goal',
        'Visible_Goal_Angle'
    ]
    monotone_constraints_set_behind = {
        'Distance_to_Middle_Goal':1,
        'Angle_to_Middle_Goal':1,
        'Visible_Goal_Angle':-1,
        'x0':1,
    }
    # Set Miss
    feature_list_set_miss = [
        'x0',
        'y0',
        'Distance_to_Middle_Goal',
        'Angle_to_Middle_Goal',
        'Visible_Goal_Angle'
    ]
    monotone_constraints_set_miss = {
        'Distance_to_Middle_Goal':1,
        'Angle_to_Middle_Goal':1,
        'Visible_Goal_Angle':-1,
        'x0':1,
    }
    
    # Open Goal
    feature_list_open_goal = [
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
    monotone_constraints_open_goal = {
        'Angle_to_Middle_Goal': -1,
        'Distance_Since_Last_Action': 1,
        'Distance_to_Middle_Goal': -1,
        'Visible_Goal_Angle': 1, 
        'x0': -1
    }
    # Open Behind
    feature_list_open_behind = [
        'x0',
        'y0',
        'Distance_Since_Last_Action',
        'Distance_to_Middle_Goal',
        'Angle_to_Middle_Goal',
        'Visible_Goal_Angle'
    ]
    monotone_constraints_open_behind = {
        'Angle_to_Middle_Goal': 1,
        'Distance_Since_Last_Action': -1,
        'Distance_to_Middle_Goal': 1,
        'Visible_Goal_Angle': -1, 
        'x0': 1
    }
    # Open Miss
    feature_list_open_miss = [
        'x0',
        'y0',
        'Distance_Since_Last_Action',
        'Distance_to_Middle_Goal',
        'Angle_to_Middle_Goal',
        'Visible_Goal_Angle'
    ]    
    monotone_constraints_open_miss = {
        'Angle_to_Middle_Goal': 1,
        'Distance_Since_Last_Action': -1,
        'Distance_to_Middle_Goal': 1,
        'Visible_Goal_Angle': -1, 
        'x0': 1
    }

