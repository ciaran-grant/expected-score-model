from highlight_text import ax_text, fig_text
from viz.afl_colours import team_colourmaps, team_colours
import matplotlib.pyplot as plt
import numpy as np

def get_quarter_duration(chain_data, quarter):
    
    quarter_data = chain_data[chain_data['Quarter'] == quarter]
    
    if "endQuarter" in list(set(quarter_data['Final_State'])):
        duration = quarter_data[(quarter_data['Final_State'] == "endQuarter")]['Quarter_Duration'].iloc[0]
    else:
        duration = quarter_data['Quarter_Duration'].max()
        
    return duration

def get_all_quarter_durations(match_chain):
    
    duration_dict = dict()
    duration_dict['duration_q1'] = get_quarter_duration(match_chain, 1)
    duration_dict['duration_q2'] = get_quarter_duration(match_chain, 2)
    duration_dict['duration_q3'] = get_quarter_duration(match_chain, 3)
    duration_dict['duration_q4'] = get_quarter_duration(match_chain, 4)
    
    return duration_dict

def get_match_chains(chain_data, match_id, extra_row=True):
    
    match_chain = chain_data[chain_data['Match_ID'] == match_id]
    
    duration_dict = get_all_quarter_durations(match_chain)
    duration_q1, duration_q2, duration_q3, duration_q4 = duration_dict['duration_q1'], duration_dict['duration_q2'], duration_dict['duration_q3'], duration_dict['duration_q4']
    
    match_chain['Duration'] = np.where(match_chain['Quarter'] == 1, match_chain['Quarter_Duration'],
                                np.where(match_chain['Quarter'] == 2, duration_q1 + match_chain['Quarter_Duration'],
                                         np.where(match_chain['Quarter'] == 3, duration_q1 + duration_q2 + match_chain['Quarter_Duration'],
                                                  np.where(match_chain['Quarter'] == 4, duration_q1 + duration_q2 + duration_q3 + match_chain['Quarter_Duration'],
                                                           0))))
    if extra_row:
        home_team = list(set(match_chain['Home_Team']))[0]
        match_chain = match_chain.append(match_chain.iloc[-1])
        match_chain.iloc[-1, match_chain.columns.get_loc('Shot_At_Goal')] = True
        match_chain.iloc[-1, match_chain.columns.get_loc('Team')] = home_team
        match_chain.iloc[-1, match_chain.columns.get_loc('Score')] = 0
        match_chain.iloc[-1, match_chain.columns.get_loc('xScore')] = 0

        away_team = list(set(match_chain['Away_Team']))[0]
        match_chain = match_chain.append(match_chain.iloc[-1])
        match_chain.iloc[-1, match_chain.columns.get_loc('Shot_At_Goal')] = True
        match_chain.iloc[-1, match_chain.columns.get_loc('Team')] = away_team
        match_chain.iloc[-1, match_chain.columns.get_loc('Score')] = 0
        match_chain.iloc[-1, match_chain.columns.get_loc('xScore')] = 0    
    return match_chain

def get_shots(chain_data):
    
    shots = chain_data[chain_data['Shot_At_Goal'] == True]
    
    return shots
    
def get_cumulative_score(shots):
    
    home = list(set(shots['Home_Team']))[0]
    home_shots = shots[shots['Team'] == home]
    home_shots['Score_cum'] = home_shots['Score'].cumsum()
    home_shots['xScore_cum'] = home_shots['xScore'].cumsum()
    
    away = list(set(shots['Away_Team']))[0]
    away_shots = shots[shots['Team'] == away]
    away_shots['Score_cum'] = away_shots['Score'].cumsum()
    away_shots['xScore_cum'] = away_shots['xScore'].cumsum()
    
    return home_shots, away_shots

def get_score(shots):
    
    home = list(set(shots['Home_Team']))[0]
    home_shots = shots[shots['Team'] == home]
    
    away = list(set(shots['Away_Team']))[0]
    away_shots = shots[shots['Team'] == away]
    
    return home_shots, away_shots

def get_team(shots):
    
    return list(set(shots['Team']))[0]

def plot_cumulative_match_story(home_shots, away_shots, duration_dict, expected=True):
        
    fig = plt.figure()
    ax = plt.subplot(111)

    home_team = get_team(home_shots)
    away_team = get_team(away_shots)
    duration_q1, duration_q2, duration_q3, duration_q4 = duration_dict['duration_q1'], duration_dict['duration_q2'], duration_dict['duration_q3'], duration_dict['duration_q4']

    if expected:
        cumulative = "xScore_cum"
    else:
        cumulative = "Score_cum"
    
    ax.step(home_shots['Duration'], home_shots[cumulative], label = home_team, c=team_colours[home_team]['positive'], zorder=2)
    ax.step(away_shots['Duration'], away_shots[cumulative], label = away_team, c=team_colours[away_team]['positive'], zorder=2)

    # Axes Limits
    ax_top = max(home_shots.iloc[-1][cumulative], away_shots.iloc[-1][cumulative])
    ax.set_ylim(0, ax_top+10)
    ax.set_xlim(0)
    # Quarters
    ax.axvline(duration_q1, c="w", zorder=1, alpha=0.3, ls="--")
    ax.axvline(duration_q1+duration_q2, c="w", zorder=1, alpha=0.3, ls="--")
    ax.axvline(duration_q1+duration_q2+duration_q3, c="w", zorder=1, alpha=0.3, ls="--")
    ax.axvline(duration_q1+duration_q2+duration_q3+duration_q4, c="w", zorder=1, alpha=0.3, ls="--")

    # Legend at end of lines
    ax_text(home_shots.iloc[-1]['Duration']+50, 
            home_shots.iloc[-1][cumulative], 
            s=f"<{round(home_shots.iloc[-1][cumulative], 1)}>",
            highlight_textprops=[
                {'size':'8', 'bbox':{'edgecolor':team_colours[home_team]['positive'], 'facecolor':team_colours[home_team]['positive'], 'pad':1}, 'color':"white"}]
    )
    ax_text(away_shots.iloc[-1]['Duration']+50, 
            away_shots.iloc[-1][cumulative], 
            s=f"<{round(away_shots.iloc[-1][cumulative], 1)}>",
            highlight_textprops=[
                {'size':'8', 'bbox':{'edgecolor':team_colours[away_team]['positive'], 'facecolor':team_colours[away_team]['positive'], 'pad':1}, 'color':"white"}]
    )

    # Title
    fig_text(x=0.15, y=0.95,
            s=f'<{home_team}> v <{away_team}>',
            highlight_textprops=[
                {'size':'16', 'bbox':{'edgecolor':team_colours[home_team]['positive'], 'facecolor':team_colours[home_team]['positive'], 'pad':1}, 'color':"white"},
                {'size':'16', 'bbox':{'edgecolor':team_colours[away_team]['positive'], 'facecolor':team_colours[away_team]['positive'], 'pad':1}, 'color':"white"}],
            size=16,
            font="Karla"
            )

    # Axis Labels
    ax.set_xlabel("Game Duration (Mins)", font="Karla")
    if expected:
        ax.set_ylabel("Expected Score", font="Karla")
    else:
        ax.set_ylabel("Score", font="Karla")

    return fig, ax

def plot_lollipop_match_story(home_shots, away_shots, duration_dict, expected=True):
        
    fig = plt.figure()
    ax = plt.subplot(111)

    home_team = get_team(home_shots)
    away_team = get_team(away_shots)
    duration_q1, duration_q2, duration_q3, duration_q4 = duration_dict['duration_q1'], duration_dict['duration_q2'], duration_dict['duration_q3'], duration_dict['duration_q4']

    if expected:
        score = "xScore"
    else:
        score = "Score"
    
    (markers, stemlines, baseline) = ax.stem(home_shots['Duration'], home_shots[score], basefmt=" ")#, c=team_colours[home_team]['positive'], zorder=2)
    plt.setp(stemlines, linestyle="-", color=team_colours[home_team]['positive'])
    plt.setp(markers, color=team_colours[home_team]['positive'], zorder=3)
    (markers, stemlines, baseline) = ax.stem(away_shots['Duration'], -1*away_shots[score], basefmt=" ")#, c=team_colours[away_team]['positive'], zorder=2)
    plt.setp(stemlines, linestyle="-", color=team_colours[away_team]['positive'], zorder=3)
    plt.setp(markers, color=team_colours[away_team]['positive'])

    ax.axhline(0, ls="--", c="w", lw=1)
    ax.set_xlim(0, duration_q1+duration_q2+duration_q3+duration_q4)
    ax.set_ylim(-6, 6)
    # Quarters
    ax.axvline(duration_q1, c="w", zorder=1, alpha=0.3, ls="--")
    ax.axvline(duration_q1+duration_q2, c="w", zorder=1, alpha=0.3, ls="--")
    ax.axvline(duration_q1+duration_q2+duration_q3, c="w", zorder=1, alpha=0.3, ls="--")
    ax.axvline(duration_q1+duration_q2+duration_q3+duration_q4, c="w", zorder=1, alpha=0.3, ls="--")

    # Legend at end of lines
    ax_text(duration_q1+duration_q2+duration_q3+duration_q4-500, 
            5.5, 
            s=f"<{round(home_shots[score].sum(), 1)}>",
            highlight_textprops=[
                {'size':'8', 'bbox':{'edgecolor':team_colours[home_team]['positive'], 'facecolor':team_colours[home_team]['positive'], 'pad':1}, 'color':"white"}]
    )
    ax_text(duration_q1+duration_q2+duration_q3+duration_q4-500, 
            -5, 
            s=f"<{round(away_shots[score].sum(), 1)}>",
            highlight_textprops=[
                {'size':'8', 'bbox':{'edgecolor':team_colours[away_team]['positive'], 'facecolor':team_colours[away_team]['positive'], 'pad':1}, 'color':"white"}]
    )

    # Title
    fig_text(x=0.15, y=0.95,
            s=f'<{home_team}> v <{away_team}>',
            highlight_textprops=[
                {'size':'16', 'bbox':{'edgecolor':team_colours[home_team]['positive'], 'facecolor':team_colours[home_team]['positive'], 'pad':1}, 'color':"white"},
                {'size':'16', 'bbox':{'edgecolor':team_colours[away_team]['positive'], 'facecolor':team_colours[away_team]['positive'], 'pad':1}, 'color':"white"}],
            size=16,
            font="Karla"
            )

    # Axis Labels
    ax.set_xlabel("Game Duration (Mins)", font="Karla")
    if expected:
        ax.set_ylabel("Expected Score", font="Karla")
    else:
        ax.set_ylabel("Score", font="Karla")

    return fig, ax