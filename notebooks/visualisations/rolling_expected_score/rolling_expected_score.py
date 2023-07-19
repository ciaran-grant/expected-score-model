import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from highlight_text import ax_text

def get_rolling_data(team, window, data):
    
    df_team = data[data['Team'] == team]
    df_team_for = df_team.groupby('Round_ID')['Score', 'xScore'].sum().rename(columns={'Score':"For",
                                                                                    "xScore":"xFor"})
    df_opp = data[((data['Home_Team'] == team) | (data['Away_Team'] == team)) & (data['Team'] != team)]
    df_team_agg = df_opp.groupby('Round_ID')['Score', 'xScore'].sum().rename(columns={'Score':"Against",
                                                                                    "xScore":"xAgainst"})

    df_rolling = pd.merge(df_team_for, df_team_agg, left_index=True, right_index=True)
    df_rolling['rollingFor'] = df_rolling['For'].rolling(window=window, min_periods=0).mean()
    df_rolling['rollingxFor'] = df_rolling['xFor'].rolling(window=window, min_periods=0).mean()
    df_rolling['rollingAgainst'] = df_rolling['Against'].rolling(window=window, min_periods=0).mean()
    df_rolling['rollingxAgainst'] = df_rolling['xAgainst'].rolling(window=window, min_periods=0).mean()

    df_rolling['rollingDiff'] = df_rolling['rollingFor'] - df_rolling['rollingAgainst']
    df_rolling['rollingxDiff'] = df_rolling['rollingxFor'] - df_rolling['rollingxAgainst']
    
    # Filling in missing Finals games with previous round (no change)
    finals_2021 = ['2021F1', '2021F2', '2021F3', '2021F4']
    finals_2021_prev = ['202123', '2021F1', '2021F2', '2021F3' ]
    finals_2022 = ['2022F1', '2022F2', '2022F3', '2022F4']
    finals_2022_prev = ['202223', '2022F1', '2022F2', '2022F3']
    for index in range(len(finals_2021)):
        if finals_2021[index] not in df_rolling.index:
            # print(finals_2021[index])
            df_rolling.loc[finals_2021[index]] = df_rolling.loc[finals_2021_prev[index]]

    for index in range(len(finals_2022)):
        if finals_2022[index] not in df_rolling.index:
            # print(finals_2022[index])
            df_rolling.loc[finals_2022[index]] = df_rolling.loc[finals_2022_prev[index]]       

    # Filling in 202101 if necessary
    if '202101' not in df_rolling.index:
        df_rolling.loc['202101'] = df_rolling.loc['202102']
        
    # Relative performance compared to Expected
    df_rolling['rollingForRelative'] = df_rolling['rollingFor'] - df_rolling['rollingxFor']
    df_rolling['rollingAgainstRelative'] = df_rolling['rollingAgainst'] - df_rolling['rollingxAgainst']
    df_rolling['rollingDiffRelative'] = df_rolling['rollingDiff'] - df_rolling['rollingxDiff']

    df_rolling = df_rolling.sort_index()
    
    return df_rolling

def plot_rolling_average(ax, data, positive_line, negative_line, color_for, color_against):
    
    x = data.index
    y_for = data[positive_line]
    y_against = data[negative_line]

    ax.set_facecolor("#121212")
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("white")

    line_for = ax.plot(x, y_for, label = positive_line, color = color_for, lw=1.5)
    line_against = ax.plot(x, y_against, label = negative_line, color= color_against, lw=1.5)
    ax.set_ylim(0, 125)
    ax.plot([22, 22], [ax.get_ylim()[0], ax.get_ylim()[1]], ls=":", lw=1.25, color="white", zorder=2)
    ax.plot([25, 25], [ax.get_ylim()[0], ax.get_ylim()[1]], ls=":", lw=1.25, color="white", zorder=2)
    plt.axvspan(22, 25, color=line_for[0].get_color(), alpha=0.3)

    ax.plot([48, 48], [ax.get_ylim()[0], ax.get_ylim()[1]], ls=":", lw=1.25, color="white", zorder=2)
    ax.plot([51, 51], [ax.get_ylim()[0], ax.get_ylim()[1]], ls=":", lw=1.25, color="white", zorder=2)
    plt.axvspan(48, 51, color=line_for[0].get_color(), alpha=0.3)

    ax.fill_between(x, y_against, y_for, where = y_for > y_against, interpolate=True, alpha=0.85, zorder=3, color=line_for[0].get_color())
    ax.fill_between(x, y_against, y_for, where = y_against >= y_for, interpolate=True, alpha=0.85, zorder=3, color=line_against[0].get_color())

    ax.tick_params(color="white", length=5, which="major", labelsize=6, labelcolor="white")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    
    plt.axvspan(0, 10, color="grey", alpha=0.3)

    ax_text(
        x=7, y=20,
        s = "2021",
        color="white"
    )
    ax_text(
        x=35, y=20,
        s = "2022",
        color="white"
    )

    return ax

## Expected Score
def get_expected_rolling_plot(ax, team, data, window=10, color_for="#0057B8", color_against = "#989898"):
    
    df_rolling = get_rolling_data(team=team, window=window, data = data)
    
    ax = plot_rolling_average(ax, data=df_rolling, positive_line="rollingxFor", negative_line="rollingxAgainst", color_for = color_for, color_against = color_against)
    
    for_number = df_rolling['rollingxFor'].iloc[-1]
    against_number = df_rolling['rollingxAgainst'].iloc[-1]
    text_colour_for = "white"
    text_colour_against = "white"
    if color_for == "white":
        text_colour_for = "black"
    if color_against == "white":
        text_colour_against = "black"
    
    ax_text(
        x=0, y=140,
        s=f'<{team}>\n<xS for: {for_number:.1f}>  <xS against: {against_number:.1f}>',
        highlight_textprops=[
            {'color':'white', 'weight':'bold', 'font':'DM Sans'},
            {'size':'10', 'bbox':{'edgecolor':color_for, 'facecolor':color_for, 'pad':1}, 'color':text_colour_for},
            {'size':'10', 'bbox':{'edgecolor':color_against, 'facecolor':color_against, 'pad':1}, 'color':text_colour_against},
        ],
        font="Karla",
        ha="left",
        size=14,
    )
    
    return ax


## Actual Scores
def get_actual_rolling_plot(ax, team, data, window=10, color_for="#0057B8", color_against = "#989898"):
    
    df_rolling = get_rolling_data(team=team, window=window, data = data)
    
    ax = plot_rolling_average(ax, data=df_rolling, positive_line="rollingFor", negative_line="rollingAgainst", color_for = color_for, color_against = color_against)
    
    for_number = df_rolling['rollingFor'].iloc[-1]
    against_number = df_rolling['rollingAgainst'].iloc[-1]
    text_colour_for = "white"
    text_colour_against = "white"
    if color_for == "white":
        text_colour_for = "black"
    if color_against == "white":
        text_colour_against = "black"
    
    ax_text(
        x=0, y=140,
        s=f'<{team}>\n<For: {for_number:.1f}>  <Against: {against_number:.1f}>',
        highlight_textprops=[
            {'color':'white', 'weight':'bold', 'font':'DM Sans'},
            {'size':'10', 'bbox':{'edgecolor':color_for, 'facecolor':color_for, 'pad':1}, 'color':text_colour_for},
            {'size':'10', 'bbox':{'edgecolor':color_against, 'facecolor':color_against, 'pad':1}, 'color':text_colour_against},
        ],
        font="Karla",
        ha="left",
        size=14,
    )
    
    return ax

# Expected Score Differences
def plot_rolling_difference(ax, data, difference_line, color_for, color_against, reverse=False):
    
    x = data.index
    y_diff = data[difference_line]
    y_neutral = [0]*len(x)

    ax.set_facecolor("#121212")
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("white")

    line_diff = ax.plot(x, y_diff, label = difference_line, color = color_for, lw=1.5)
    line_neutral = ax.plot(x, y_neutral, color= "white", lw=1.5)
    ax.set_ylim(-20, 20)
    ax.plot([22, 22], [ax.get_ylim()[0], ax.get_ylim()[1]], ls=":", lw=1.25, color="white", zorder=2)
    ax.plot([25, 25], [ax.get_ylim()[0], ax.get_ylim()[1]], ls=":", lw=1.25, color="white", zorder=2)
    plt.axvspan(22, 25, color=line_diff[0].get_color(), alpha=0.3)

    ax.plot([48, 48], [ax.get_ylim()[0], ax.get_ylim()[1]], ls=":", lw=1.25, color="white", zorder=2)
    ax.plot([51, 51], [ax.get_ylim()[0], ax.get_ylim()[1]], ls=":", lw=1.25, color="white", zorder=2)
    plt.axvspan(48, 51, color=line_diff[0].get_color(), alpha=0.3)

    positive_colour = color_for
    negative_colour = "grey"
    if reverse:
        positive_colour = "grey"
        negative_colour = color_for
    ax.fill_between(x, y_neutral, y_diff, where = y_diff > y_neutral, interpolate=True, alpha=0.85, zorder=3, color=positive_colour)
    ax.fill_between(x, y_neutral, y_diff, where = y_neutral >= y_diff, interpolate=True, alpha=0.85, zorder=3, color=negative_colour)

    ax.tick_params(color="white", length=5, which="major", labelsize=6, labelcolor="white")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    
    plt.axvspan(0, 10, color="grey", alpha=0.3)

    ax_text(
        x=7, y=-15,
        s = "2021",
        color="white"
    )
    ax_text(
        x=35, y=-15,
        s = "2022",
        color="white"
    )

def get_difference_rolling_plot(ax, team, data, window=10, color_for="#0057B8", color_against = "#989898", reverse=False):
    
    df_rolling = get_rolling_data(team=team, window=window, data = data)
    
    ax = plot_rolling_difference(ax, data=df_rolling, difference_line="rollingDiffRelative", color_for = color_for, color_against = color_against, reverse=reverse)
    
    diff_number = df_rolling['rollingDiffRelative'].mean()
    text_colour_for = "white"
    # text_colour_against = "white"
    if color_for == "white":
        text_colour_for = "black"
    # if color_against == "white":
        # text_colour_against = "black"
    
    ax_text(
        x=0, y=23,
        s=f'<{team}>\n<Score Difference Relative to Expected: {diff_number:.1f}>',
        highlight_textprops=[
            {'color':'white', 'weight':'bold', 'font':'DM Sans'},
            {'size':'10', 'bbox':{'edgecolor':color_for, 'facecolor':color_for, 'pad':1}, 'color':text_colour_for},
            # {'size':'10', 'bbox':{'edgecolor':color_against, 'facecolor':color_against, 'pad':1}, 'color':text_colour_against},
        ],
        font="Karla",
        ha="left",
        size=14,
    )
    
    return ax

def get_for_difference_rolling_plot(ax, team, data, window=10, color_for="#0057B8", color_against = "#989898"):
    
    df_rolling = get_rolling_data(team=team, window=window, data = data)
    
    ax = plot_rolling_difference(ax, data=df_rolling, difference_line="rollingForRelative", color_for = color_for, color_against = color_against)
    
    diff_number = df_rolling['rollingForRelative'].mean()
    text_colour_for = "white"
    # text_colour_against = "white"
    if color_for == "white":
        text_colour_for = "black"
    # if color_against == "white":
        # text_colour_against = "black"
    
    ax_text(
        x=0, y=23,
        s=f'<{team}>\n<Score For Relative to Expected: {diff_number:.1f}>',
        highlight_textprops=[
            {'color':'white', 'weight':'bold', 'font':'DM Sans'},
            {'size':'10', 'bbox':{'edgecolor':color_for, 'facecolor':color_for, 'pad':1}, 'color':text_colour_for},
            # {'size':'10', 'bbox':{'edgecolor':color_against, 'facecolor':color_against, 'pad':1}, 'color':text_colour_against},
        ],
        font="Karla",
        ha="left",
        size=14,
    )
    
    return ax

def get_against_difference_rolling_plot(ax, team, data, window=10, color_for="#0057B8", color_against = "#989898"):
    
    df_rolling = get_rolling_data(team=team, window=window, data = data)
    
    ax = plot_rolling_difference(ax, data=df_rolling, difference_line="rollingAgainstRelative", color_for = color_for, color_against = color_against, reverse=True)
    
    diff_number = df_rolling['rollingAgainstRelative'].mean()
    text_colour_for = "white"
    # text_colour_against = "white"
    if color_for == "white":
        text_colour_for = "black"
    # if color_against == "white":
        # text_colour_against = "black"
    
    ax_text(
        x=0, y=23,
        s=f'<{team}>\n<Score Against Relative to Expected: {diff_number:.1f}>',
        highlight_textprops=[
            {'color':'white', 'weight':'bold', 'font':'DM Sans'},
            {'size':'10', 'bbox':{'edgecolor':color_for, 'facecolor':color_for, 'pad':1}, 'color':text_colour_for},
            # {'size':'10', 'bbox':{'edgecolor':color_against, 'facecolor':color_against, 'pad':1}, 'color':text_colour_against},
        ],
        font="Karla",
        ha="left",
        size=14,
    )
    
    return ax