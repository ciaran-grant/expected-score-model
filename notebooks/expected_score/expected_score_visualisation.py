import joblib
import numpy as np
import pandas as pd

import sys
sys.path.append("..")
sys.path.append("../..")

model_location = '/Users/ciaran/Documents/Projects/AFL/git-repositories/expected-score-model/model_outputs/models/'
expected_goal_set_version = 7
expected_behind_set_version = 5
expected_miss_set_version = 4

expected_goal_open_version = 7
expected_behind_open_version = 6
expected_miss_open_version = 5
expected_goal_set_model = joblib.load(model_location+"expected_goal_set_v"+str(expected_goal_set_version)+".joblib")
expected_behind_set_model = joblib.load(model_location+"expected_behind_set_v"+str(expected_behind_set_version)+".joblib")
expected_miss_set_model = joblib.load(model_location+"expected_miss_set_v"+str(expected_miss_set_version)+".joblib")

expected_goal_open_model = joblib.load(model_location+"expected_goal_open_v"+str(expected_goal_open_version)+".joblib")
expected_behind_open_model = joblib.load(model_location+"expected_behind_open_v"+str(expected_behind_open_version)+".joblib")
expected_miss_open_model = joblib.load(model_location+"expected_miss_open_v"+str(expected_miss_open_version)+".joblib")

def calculate_x0(x):
    
    return x

def calculate_y0(y):
    
    return y

def calculate_distance_goal_x(x, goal_x):
    
    return goal_x - x

def calculate_distance_middle_y(y):
    
    return abs(y)

def calculate_distance_middle_goal(x, y, goal_x):
    
    distance_to_goal_x = goal_x - x
    distance_to_middle_y = abs(y)
    
    return (distance_to_goal_x**2 + distance_to_middle_y**2)**0.5

def calculate_angle_middle_goal(x, y, goal_x):
    
    distance_to_goal_x = goal_x - x
    distance_to_middle_y = abs(y)

    return np.arctan2(distance_to_middle_y, distance_to_goal_x)

def calculate_visible_goal_angle(x, y, goal_x):
    
    distance_to_goal_x = goal_x - x
    distance_to_middle_y = abs(y)

    return (6.4*distance_to_goal_x) / (distance_to_goal_x**2 + distance_to_middle_y**2-(6.4/2)**2)

def calculate_distance_last_action(x0, y0, x1, y1):
    
    return ((x1 - x0)**2 + (y1 - y0)**2)**0.5

def predict_expected_goal_set(x_pos, y_pos, pitch, calibrated, initial_state = "kick_in"):
        
    goal_x = pitch.dim.pitch_length/2

    x0 = calculate_x0(x_pos)
    distance_to_middle_goal = calculate_distance_middle_goal(x_pos, y_pos, goal_x)
    angle_to_middle_goal = calculate_angle_middle_goal(x_pos, y_pos, goal_x)
    visible_goal_angle = calculate_visible_goal_angle(x_pos, y_pos, goal_x)
    ball_up = 0
    centre_bounce = 0
    kick_in = 0
    poss_gain = 0
    throw_in = 0
    if initial_state == "ball_up":
        ball_up = 1
    if initial_state == "centre_bounce":
        centre_bounce = 1
    if initial_state == "kick_in":
        kick_in = 1
    if initial_state == "poss_gain":
        poss_gain = 1
    if initial_state == "throw_in":
        throw_in = 1

    position_dict = {
        'x0':x0,
        'Distance_to_Middle_Goal':distance_to_middle_goal,
        'Angle_to_Middle_Goal':angle_to_middle_goal,
        'Visible_Goal_Angle':visible_goal_angle,
        'ballUp':ball_up,
        'centreBounce':centre_bounce,
        'kickIn':kick_in,
        'possGain':poss_gain,
        'throwIn':throw_in
    }
    position_df = pd.DataFrame.from_dict(position_dict, orient='index').T

    if calibrated:
        return expected_goal_set_model.xgb_cal.predict(expected_goal_set_model.predict_proba(position_df)[:,1])
    else:
        return expected_goal_set_model.predict_proba(position_df)[:,1]
    
def predict_expected_behind_set(x_pos, y_pos, pitch, calibrated):
    
    goal_x = pitch.dim.pitch_length/2
    
    x0 = calculate_x0(x_pos)
    y0 = calculate_y0(y_pos)
    distance_to_middle_goal = calculate_distance_middle_goal(x_pos, y_pos, goal_x)
    angle_to_middle_goal = calculate_angle_middle_goal(x_pos, y_pos, goal_x)
    visible_goal_angle = calculate_visible_goal_angle(x_pos, y_pos, goal_x)
    
    position_dict = {
        'x0':x0,
        'y0':y0,
        'Distance_to_Middle_Goal':distance_to_middle_goal,
        'Angle_to_Middle_Goal':angle_to_middle_goal,
        'Visible_Goal_Angle':visible_goal_angle
    }
    position_df = pd.DataFrame.from_dict(position_dict, orient='index').T

    if calibrated:
        return expected_behind_set_model.xgb_cal.predict(expected_behind_set_model.predict_proba(position_df)[:,1])
    else:
        return expected_behind_set_model.predict_proba(position_df)[:,1]
    
def predict_expected_miss_set(x_pos, y_pos, pitch, calibrated):
    
    goal_x = pitch.dim.pitch_length/2

    x0 = calculate_x0(x_pos)
    y0 = calculate_y0(y_pos)
    distance_to_middle_goal = calculate_distance_middle_goal(x_pos, y_pos, goal_x)
    angle_to_middle_goal = calculate_angle_middle_goal(x_pos, y_pos, goal_x)
    visible_goal_angle = calculate_visible_goal_angle(x_pos, y_pos, goal_x)
    
    position_dict = {
        'x0':x0,
        'y0':y0,
        'Distance_to_Middle_Goal':distance_to_middle_goal,
        'Angle_to_Middle_Goal':angle_to_middle_goal,
        'Visible_Goal_Angle':visible_goal_angle
    }
    position_df = pd.DataFrame.from_dict(position_dict, orient='index').T

    if calibrated:
        return expected_miss_set_model.xgb_cal.predict(expected_miss_set_model.predict_proba(position_df)[:,1])
    else:
        return expected_miss_set_model.predict_proba(position_df)[:,1]
    
def predict_expected_goal_open(x_pos, y_pos, pitch, calibrated, initial_state = "kick_in"):
    
    goal_x = pitch.dim.pitch_length/2
    
    x0 = calculate_x0(x_pos)
    x1 = x0-5
    x2 = x1-5
    x3 = x2-5
    y0 = calculate_y0(y_pos)
    y1 = y0+5
    distance_since_last_action = calculate_distance_last_action(x0, y0, x1, y1)
    distance_to_middle_goal = calculate_distance_middle_goal(x_pos, y_pos, goal_x)
    angle_to_middle_goal = calculate_angle_middle_goal(x_pos, y_pos, goal_x)
    visible_goal_angle = calculate_visible_goal_angle(x_pos, y_pos, goal_x)
    ball_up = 0
    centre_bounce = 0
    kick_in = 0
    poss_gain = 0
    throw_in = 0
    if initial_state == "ball_up":
        ball_up = 1
    if initial_state == "centre_bounce":
        centre_bounce = 1
    if initial_state == "kick_in":
        kick_in = 1
    if initial_state == "poss_gain":
        poss_gain = 1
    if initial_state == "throw_in":
        throw_in = 1
    distance_to_goal = calculate_distance_goal_x(x_pos, goal_x)
    distance_to_middle = calculate_distance_middle_y(y_pos)
    chain_duration = 10
    time_since_last_action = 10

    position_dict = {
        'x0':x0,
        'x1':x1,
        'x2':x2,
        'x3':x3,
        'Distance_Since_Last_Action':distance_since_last_action,
        'Distance_to_Middle_Goal':distance_to_middle_goal,
        'Angle_to_Middle_Goal':angle_to_middle_goal,
        'Visible_Goal_Angle':visible_goal_angle,
        'ballUp':ball_up,
        'centreBounce':centre_bounce,
        'kickIn':kick_in,
        'possGain':poss_gain,
        'throwIn':throw_in,
        "Distance_to_Right_Goal_x":distance_to_goal,
        'Distance_to_Middle_y':distance_to_middle,
        'Chain_Duration':chain_duration,
        'Time_Since_Last_Action':time_since_last_action
    }
    position_df = pd.DataFrame.from_dict(position_dict, orient='index').T

    if calibrated:
        return expected_goal_open_model.xgb_cal.predict(expected_goal_open_model.predict_proba(position_df)[:,1])
    else:
        return expected_goal_open_model.predict_proba(position_df)[:,1]
    
def predict_expected_behind_open(x_pos, y_pos, pitch, calibrated):

    goal_x = pitch.dim.pitch_length/2
    
    x0 = calculate_x0(x_pos)
    x1 = x0-5
    y0 = calculate_y0(y_pos)
    y1 = y0+5
    distance_since_last_action = calculate_distance_last_action(x0, y0, x1, y1)
    distance_to_middle_goal = calculate_distance_middle_goal(x_pos, y_pos, goal_x)
    angle_to_middle_goal = calculate_angle_middle_goal(x_pos, y_pos, goal_x)
    visible_goal_angle = calculate_visible_goal_angle(x_pos, y_pos, pitch.dim.pitch_length/2)
    
    position_dict = {
        'x0':x0,
        'y0':y0,
        'Distance_Since_Last_Action':distance_since_last_action,
        'Distance_to_Middle_Goal':distance_to_middle_goal,
        'Angle_to_Middle_Goal':angle_to_middle_goal,
        'Visible_Goal_Angle':visible_goal_angle
    }
    position_df = pd.DataFrame.from_dict(position_dict, orient='index').T

    if calibrated:
        return expected_behind_open_model.xgb_cal.predict(expected_behind_open_model.predict_proba(position_df)[:,1])
    else:
        return expected_behind_open_model.predict_proba(position_df)[:,1]
    
def predict_expected_miss_open(x_pos, y_pos, pitch, calibrated):
    
    goal_x = pitch.dim.pitch_length/2

    x0 = calculate_x0(x_pos)
    x1 = x0-5
    y0 = calculate_y0(y_pos)
    y1 = y0+5
    distance_since_last_action = calculate_distance_last_action(x0, y0, x1, y1)
    distance_to_middle_goal = calculate_distance_middle_goal(x_pos, y_pos, goal_x)
    angle_to_middle_goal = calculate_angle_middle_goal(x_pos, y_pos, goal_x)
    visible_goal_angle = calculate_visible_goal_angle(x_pos, y_pos, goal_x)
    
    position_dict = {
        'x0':x0,
        'y0':y0,
        'Distance_Since_Last_Action':distance_since_last_action,
        'Distance_to_Middle_Goal':distance_to_middle_goal,
        'Angle_to_Middle_Goal':angle_to_middle_goal,
        'Visible_Goal_Angle':visible_goal_angle
    }
    position_df = pd.DataFrame.from_dict(position_dict, orient='index').T

    if calibrated:
        return expected_miss_open_model.xgb_cal.predict(expected_miss_open_model.predict_proba(position_df)[:,1])
    else:
        return expected_miss_open_model.predict_proba(position_df)[:,1]
    
def get_expected_pitch_probability(pitch, outcome, shot_type, calibrated=True, initial_state="kick_in"):
    
    probs = list()
    x_size = int(1+pitch.dim.pitch_length/2)
    y_size = int(1+pitch.dim.pitch_width)
    y_min = -int(1+pitch.dim.pitch_width/2)
    y_max = int(1+pitch.dim.pitch_width/2)
    for x_pos in range(0, x_size):
        for y_pos in range(y_min, y_max):
            
            if (outcome=="goal") & (shot_type =="set"):
                xy_prob = predict_expected_goal_set(x_pos, y_pos, pitch, calibrated=calibrated, initial_state=initial_state)
            if (outcome=="goal") & (shot_type =="open"):
                xy_prob = predict_expected_goal_open(x_pos, y_pos, pitch, calibrated=calibrated, initial_state=initial_state)
            if (outcome=="behind") & (shot_type =="set"):
                xy_prob = predict_expected_behind_set(x_pos, y_pos, pitch, calibrated=calibrated)
            if (outcome=="behind") & (shot_type =="open"):
                xy_prob = predict_expected_behind_open(x_pos, y_pos, pitch, calibrated=calibrated)
            if (outcome=="miss") & (shot_type =="set"):
                xy_prob = predict_expected_miss_set(x_pos, y_pos, pitch, calibrated=calibrated)
            if (outcome=="miss") & (shot_type =="open"):
                xy_prob = predict_expected_miss_open(x_pos, y_pos, pitch, calibrated=calibrated)

            probs.append(xy_prob)
            
    expected_probs = np.array(probs).reshape(x_size, y_size)

    return expected_probs

def clip_image_to_pitch_boundary(pitch, ax, im):
    
    from matplotlib.patches import Arc, Polygon
    
    # Clip to pitch boundary
    top_theta_start = 0
    top_boundary_y = pitch.dim.behind_top
    bottom_boundary_y = pitch.dim.behind_bottom
    if pitch.vertical:
        top_theta_start = 180
        top_boundary_y = pitch.dim.behind_bottom
        bottom_boundary_y = pitch.dim.behind_top
    top_theta_end = top_theta_start + 180
    top_boundary_arc = Arc((0, top_boundary_y), 
                                    width = pitch.dim.pitch_length, 
                                    height = pitch.dim.boundary_width, 
                                    theta1=top_theta_start, theta2=top_theta_end
                                    )
    top_vertices = pitch._reverse_vertices_if_vertical(top_boundary_arc.get_verts()[:-1])

    bottom_theta_start = top_theta_end
    bottom_theta_end = bottom_theta_start + 180
    bottom_boundary_arc = Arc((0, bottom_boundary_y), 
                                width = pitch.dim.pitch_length, 
                                height = pitch.dim.boundary_width, 
                                theta1=bottom_theta_start, theta2=bottom_theta_end
                                )
    bottom_vertices = pitch._reverse_vertices_if_vertical(bottom_boundary_arc.get_verts()[:-1])

    pitch_boundary_vertices = np.concatenate([top_vertices, bottom_vertices])

    A = Polygon(pitch_boundary_vertices, color= "w", zorder=-1, transform=ax.transData)
    im.set_clip_path(A)