from expected_score_model.predict import predict_xscore_from_chains
from expected_score_model.visualisation.plot_team_rolling_averages import create_team_rolling, plot_team_rolling_ax, plot_all_team_rolling_figure
from AFLPy.AFLData_Client import load_data, upload_data

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot

import matplotlib.pyplot as plt
from flask import Flask, request, send_file
import io
import os

app = Flask(__name__)

@app.route("/model/expectedscore/predict", methods=["GET", "POST"])
def predict(ID=None):
    # Load chain data
    chains = load_data(Dataset_Name='AFL_API_Match_Chains', ID=request.json['ID'])

    shots = predict_xscore_from_chains(chains)
    shots = shots.drop_duplicates(subset=['Chain_Number', 'Period_Duration_Chain_Start'])
    
    upload_data(Dataset_Name="CG_Expected_Score", Dataset=shots, overwrite=True, update_if_identical=True)
    
    return shots.to_json(orient='records')

@app.route("/model/expectedscore/plot_team_rolling_xscore", methods=["GET", "POST"])
def plot_team_rolling_xscore():
    # Get data from request body
    data = request.get_json()  # Ensure the request has JSON
    if not data or 'team' not in data:
        return {"error": "Missing 'team' in request body"}, 400

    team = data['team']
    window = data.get('window', 10)  # Default to 10 if not provided
    years = data.get('years', None)

    # Load shots with expected score data
    shots = load_data(Dataset_Name="CG_Expected_Score")
    shots['Year'] = shots['Match_ID'].apply(lambda x: int(x.split("_")[1]))
    shots['Round'] = shots['Match_ID'].apply(lambda x: x.split("_")[2])

    team_rolling = create_team_rolling(shots, team, window, metric='xscore')

    # Generate plot
    fig, ax = plt.subplots()
    ax = plot_team_rolling_ax(ax=ax, team=team, team_rolling=team_rolling, annotate=True, years=years)

    # Convert plot to image
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)  # Free memory

    return send_file(img, mimetype='image/png')

@app.route("/model/expectedscore/plot_all_team_rolling_xscore", methods=["POST"])
def plot_all_team_rolling_xscore():
    data = request.get_json()  # Ensure the request has JSON

    window = data.get('window', 10)  # Default to 10 if not provided
    metric = data.get('metric', 'xscore')
    annotate = data.get('annotate', True)
    add_title = data.get('add_title', True)
    years = data.get('years', None)
    
    style = data.get('style', 'dark')

    # Load shots with xscore
    shots = load_data(Dataset_Name="CG_Expected_Score")
    shots['Year'] = shots['Match_ID'].apply(lambda x: int(x.split("_")[1]))
    shots['Round'] = shots['Match_ID'].apply(lambda x: x.split("_")[2])

    style_path = os.path.join(os.path.dirname(__file__), 'src', 'expected_score_model', 'visualisation', 'styles', f'{style}.mplstyle')
    plt.style.use(style_path)

    fig, ax = plot_all_team_rolling_figure(
        shots, 
        window=window, 
        metric = metric, 
        annotate = annotate,
        add_title=add_title,
        years=years)

    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=False)