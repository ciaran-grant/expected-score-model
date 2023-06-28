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
        'Distance_Since_Last_Action',
        'Distance_to_Middle_Goal',
        'Angle_to_Middle_Goal',
        'Visible_Goal_Angle',
        'possGain', 
    ]

    monotone_constraints_set_goal = {
        'Distance_to_Middle_Goal':-1,
        'Angle_to_Middle_Goal':-1,
        'Visible_Goal_Angle':1,
        'x0':-1,
    }
    monotone_constraints_open_goal = {
        'Distance_to_Middle_Goal':-1,
        'Angle_to_Middle_Goal':-1,
        'Visible_Goal_Angle':1,
        'x0':-1,
        'Distance_Since_Last_Action':1
    }
    monotone_constraints_behind = {
        'Distance_to_Middle_Goal':1,
        'Angle_to_Middle_Goal':1
    }
    monotone_constraints_miss = {
        'Distance_to_Middle_Goal':1,
        'Angle_to_Middle_Goal':1
    }