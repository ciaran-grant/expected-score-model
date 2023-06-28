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
    
    modelling_feature_list = [
        # 'Initial_State',
        # 'Quarter',
        # 'Quarter_Duration',
        # 'Team',
        # 'x',
        # 'y',
        # 'Current_Margin',
        # 'Event_Type0',
        # 'Event_Type1',
        # 'Event_Type2',
        # 'Event_Type3',
        # 'x0',
        # 'x1',
        # 'x2',
        # 'x3',
        # 'y0',
        # 'y1',
        # 'y2',
        # 'y3',
        # 'Quarter_Duration0',
        # 'Quarter_Duration1',
        # 'Quarter_Duration2',
        # 'Quarter_Duration3',
        # 'Chain_Duration',
        # 'Time_Since_Last_Action',
        # 'Distance_Since_Last_Action',
        # 'Distance_to_Right_Goal_x',
        # 'Distance_to_Middle_y',
        'Distance_to_Middle_Goal',
        'Angle_to_Middle_Goal',
        # 'Angle_to_Middle_Goal_degrees',
        # 'Visible_Goal_Angle',
        # 'Visible_Goal_Angle_degrees',
        # 'Visible_Behind_Angle',
        # 'Visible_Behind_Angle_degrees',
        # 'Squared_Distance_to_Middle_Goal',
        # 'Log_Distance_to_Middle_Goal'
    ]
    
    monotone_constraints_goal = {
        'Distance_to_Middle_Goal':-1,
        'Angle_to_Middle_Goal':-1
    }
    monotone_constraints_behind = {
        'Distance_to_Middle_Goal':1,
        'Angle_to_Middle_Goal':1
    }
    monotone_constraints_miss = {
        'Distance_to_Middle_Goal':1,
        'Angle_to_Middle_Goal':1
    }