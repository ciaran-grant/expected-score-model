from viz.afl_colours import team_colourmaps, team_colours
from mplfooty.pitch import VerticalPitch
import matplotlib.pyplot as plt
from highlight_text import fig_text

## Chain Filters
def get_venue_dimensions(chain_data, match_id):
    
    # Get match chain information
    match = chain_data[chain_data['Match_ID'] == match_id]

    return list(set(match['Venue_Width']))[0], list(set(match['Venue_Length']))[0]

def get_match(chain_data, match_id):
    
    return chain_data[chain_data['Match_ID'] == match_id]

def get_team(chain_data, team):
    
    return chain_data[chain_data['Team'] == team]

def get_player(chain_data, player):
    
    return chain_data[chain_data['Player'] == player]

def get_year(chain_data, year):
    
    return chain_data[chain_data['Year'] == year]

def get_shots(chain_data, set_shot):
    
    shots = chain_data[chain_data['Shot_At_Goal'] == True]
    shots = shots[shots['Set_Shot'] == set_shot]
    
    return shots

def get_shot_outcome(shots, final_state):
    
    return shots[shots['Final_State'] == final_state]


# Vertical Shot Map
def plot_vertical_shot_map(chain_data, pitch_width, pitch_length):

    set_shots = get_shots(chain_data, set_shot=True)
    open_shots = get_shots(chain_data, set_shot=False)

    set_goals = get_shot_outcome(set_shots, "goal")
    set_behinds = get_shot_outcome(set_shots, "behind")
    set_misses = get_shot_outcome(set_shots, "miss")
    open_goals = get_shot_outcome(open_shots, "goal")
    open_behinds = get_shot_outcome(open_shots, "behind")
    open_misses = get_shot_outcome(open_shots, "miss")

    team = list(chain_data['Team'].unique())[0]
    cmap = team_colourmaps[team]
    norm = plt.Normalize(vmin=0, vmax=6)
    size_ratio=3

    pitch = VerticalPitch(pitch_width=pitch_width, pitch_length=pitch_length, 
                        line_colour="white", pitch_colour="#121212", 
                        line_width=0.5,
                        half = True, pad_bottom=-5)
    fig, ax = pitch.draw()
    fig.dpi = 300
    pitch.scatter(set_misses['x'], set_misses['y'], ax=ax, s=(set_misses['xScore']**2)*size_ratio, c=cmap(norm(set_misses['xScore'])), alpha=0.3, marker="s")
    pitch.scatter(set_behinds['x'], set_behinds['y'], ax=ax, s=(set_behinds['xScore']**2)*size_ratio, c=cmap(norm(set_behinds['xScore'])), marker="s")
    pitch.scatter(set_goals['x'], set_goals['y'], ax=ax, s=(set_goals['xScore']**2)*size_ratio, c=cmap(norm(set_goals['xScore'])), ec="white", marker="s")

    pitch.scatter(open_misses['x'], open_misses['y'], ax=ax, s=(open_misses['xScore']**2)*size_ratio, c=cmap(norm(open_misses['xScore'])), alpha=0.3)
    pitch.scatter(open_behinds['x'], open_behinds['y'], ax=ax, s=(open_behinds['xScore']**2)*size_ratio, c=cmap(norm(open_behinds['xScore'])))
    pitch.scatter(open_goals['x'], open_goals['y'], ax=ax, s=(open_goals['xScore']**2)*size_ratio, c=cmap(norm(open_goals['xScore'])), ec="white")

    ## Manual Legend
    legend_ax = fig.add_axes([0.75, 0.5, 0.2, 0.3])
    legend_ax.axis("off")
    plt.xlim([0, 7])
    plt.ylim([0, 1])
    legend_ax.xaxis.set_tick_params(color="white")
    for size in [1, 2, 3, 4, 5, 6]:
        legend_ax.scatter(size, 0.95, s=(size**2)*size_ratio, c=cmap(norm(size)))
    for size in [1, 6]:
        legend_ax.text(size-0.14, 0.82, str(size), color="white", fontsize=8, font='Karla')
        
    legend_ax.scatter(3.5, 0.65, s=(4**2)*size_ratio, c=cmap(norm(4)), ec="white")
    legend_ax.scatter(3.5, 0.53, s=(4**2)*size_ratio, c=cmap(norm(4)))
    legend_ax.scatter(3.5, 0.41, s=(4**2)*size_ratio, c=cmap(norm(4)), alpha=0.3)
    legend_ax.scatter(5, 0.65, s=(4**2)*size_ratio, c=cmap(norm(4)), ec="white", marker="s")
    legend_ax.scatter(5, 0.53, s=(4**2)*size_ratio, c=cmap(norm(4)), marker = "s")
    legend_ax.scatter(5, 0.41, s=(4**2)*size_ratio, c=cmap(norm(4)), alpha=0.3, marker = "s")

    legend_ax.text(2.8, 0.73, "Open", color="white", fontsize=8, font='Karla')
    legend_ax.text(4.5, 0.73, "Set", color="white", fontsize=8, font='Karla')
    legend_ax.text(1, 0.62, "Goal", color="white", fontsize=8, font='Karla')
    legend_ax.text(1, 0.5, "Behind", color="white", fontsize=8, font='Karla')
    legend_ax.text(1, 0.38, "Miss", color="white", fontsize=8, font='Karla')
    
    num_shots = set_shots.shape[0] + open_shots.shape[0]
    score = set_shots['Score'].sum() + open_shots['Score'].sum()
    xScore = set_shots['xScore'].sum() + open_shots['xScore'].sum()

    fig.text(0.05, 0.78, 'Shots: ', color="white", font="Karla", size=8)
    fig.text(0.05, 0.75, 'Goals: ', color="white", font="Karla", size=8)
    fig.text(0.05, 0.72, 'Behinds: ', color="white", font="Karla", size=8)
    fig.text(0.05, 0.69, 'Misses: ', color="white", font="Karla", size=8)
    fig.text(0.05, 0.66, 'Score: ', color="white", font="Karla", size=8)

    fig.text(0.13, 0.78, f'{set_shots.shape[0] + open_shots.shape[0]}', color="white", font="Karla", size=8)
    fig.text(0.13, 0.75, f'{set_goals.shape[0] + open_goals.shape[0]}', color="white", font="Karla", size=8)
    fig.text(0.13, 0.72, f'{set_behinds.shape[0] + open_behinds.shape[0]}', color="white", font="Karla", size=8)
    fig.text(0.13, 0.69, f'{set_misses.shape[0] + open_misses.shape[0]}', color="white", font="Karla", size=8)
    fig.text(0.13, 0.66, f"{set_shots['Score'].sum() + open_shots['Score'].sum()}", color="white", font="Karla", size=8)

    fig.text(0.18, 0.75, 'xG: ', color="white", font="Karla", size=8)
    fig.text(0.18, 0.72, 'xB: ', color="white", font="Karla", size=8)
    fig.text(0.18, 0.69, 'xM: ', color="white", font="Karla", size=8)
    fig.text(0.18, 0.66, 'xS: ', color="white", font="Karla", size=8)

    fig.text(0.22, 0.75, f"{round(set_shots['xGoals_normalised'].sum() + open_shots['xGoals_normalised'].sum(), 1)}", color="white", font="Karla", size=8)
    fig.text(0.22, 0.72, f"{round(set_shots['xBehinds_normalised'].sum() + open_shots['xBehinds_normalised'].sum(), 1)}", color="white", font="Karla", size=8)
    fig.text(0.22, 0.69, f"{round(set_shots['xMiss_normalised'].sum() + open_shots['xMiss_normalised'].sum(), 1)}", color="white", font="Karla", size=8)
    fig.text(0.22, 0.66, f"{round(set_shots['xScore'].sum() + open_shots['xScore'].sum(), 1)}", color="white", font="Karla", size=8)

    fig.text(0.05, 0.6, "xS / Shot: ", color="white", font="Karla", size=8)
    fig.text(0.05, 0.55, "Score - xS: ", color="white", font="Karla", size=8)
    fig_text(0.15, 0.62, s=f'<{round(xScore / num_shots, 2)}>',
             highlight_textprops=[
                 {'size':'8', 'bbox':{'edgecolor':team_colours[team]['positive'], 'facecolor':team_colours[team]['positive'], 'pad':1}, 'color':"white"}]
    )
    if score-xScore > 0:
        sign = "+"
    elif score-xScore < 0:
        sign = "-"
    fig_text(0.15, 0.57, s=f'<{sign}{round(score-xScore, 2)}>',
             highlight_textprops=[
                 {'size':'8', 'bbox':{'edgecolor':team_colours[team]['positive'], 'facecolor':team_colours[team]['positive'], 'pad':1}, 'color':"white"}]
    )
    return fig, ax
