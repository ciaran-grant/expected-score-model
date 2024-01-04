import requests
import json

url = "http://localhost:8000/predict/open_shots"

headers = {"Content-type": "application/json", "Accept": "application/json"}

def test_single_request():
    
    shot_data = [
        {
            "x0": 43.0,
            "Distance_to_Middle_y": 18.0,
            "Distance_Since_Last_Action": 8.06225774829855,
            "Visible_Goal_Angle": 0.14072119613016712,
            "x1": 50.0,
            "Distance_to_Right_Goal_x": 37.0,
            "Distance_to_Middle_Goal": 41.14608122288197,
            "Time_Since_Last_Action": 40.0,
            "x2": 47.0,
            "x3": 54.0,
            "Chain_Duration": 43.0,
            "Angle_to_Middle_Goal": 0.4527784718235359,
            "Initial_State": "possGain",
            "y0": 18.0,
        }
    ]
    shot_json = json.dumps(shot_data)

    response = requests.post(url, data=shot_json, headers=headers)
    prediction = response.json()["prediction"]

    assert prediction == [2.7930132191037083]

def test_multiple_requests():
    
    multiple_shot_data = [
        {
            "x0": 43.0,
            "Distance_to_Middle_y": 18.0,
            "Distance_Since_Last_Action": 8.06225774829855,
            "Visible_Goal_Angle": 0.14072119613016712,
            "x1": 50.0,
            "Distance_to_Right_Goal_x": 37.0,
            "Distance_to_Middle_Goal": 41.14608122288197,
            "Time_Since_Last_Action": 40.0,
            "x2": 47.0,
            "x3": 54.0,
            "Chain_Duration": 43.0,
            "Angle_to_Middle_Goal": 0.4527784718235359,
            "Initial_State": "possGain",
            "y0": 18.0,
        },
        {
            "x0": 24.0,
            "Distance_to_Middle_y": 16.0,
            "Distance_Since_Last_Action": 8.06225774829855,
            "Visible_Goal_Angle": 0.09841785235305388,
            "x1": 16.0,
            "Distance_to_Right_Goal_x": 61.0,
            "Distance_to_Middle_Goal": 63.06346010171025,
            "Time_Since_Last_Action": 0.0,
            "x2": 14.0,
            "x3": -17.0,
            "Chain_Duration": 0.0,
            "Angle_to_Middle_Goal": 0.25651661264432357,
            "Initial_State": "possGain",
            "y0": -16.0,
        },
    ]
    multiple_shot_json = json.dumps(multiple_shot_data)

    multiple_response = requests.post(url, data=multiple_shot_json, headers=headers)
    predictions = multiple_response.json()["prediction"]
    
    assert predictions == [
        2.7930132191037083,
        2.376009102138349
    ]
