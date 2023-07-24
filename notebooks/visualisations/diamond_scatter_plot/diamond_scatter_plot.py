import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
import adjustText

from viz.afl_colours import team_colours
team_list = list(team_colours.keys())

def round_decimals_up(number:float, decimals:int=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

def round_decimals_down(number:float, decimals:int=2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor

def get_diamond_plot_extents(data, x, y, x_pad, y_pad):
    
    extent_dict = {}
    extent_dict['x_mean'] = data[x].mean()
    extent_dict['y_mean'] = data[y].mean()
    extent_dict['x_max'] = data[x].max()
    extent_dict['y_max'] = data[y].max()
    extent_dict['x_min'] = data[x].min()
    extent_dict['y_min'] = data[y].min()

    extent_dict['x_extent_min'] = max(round_decimals_down(extent_dict['x_mean']-x_pad, 2), 0)
    extent_dict['x_extent_max'] = round_decimals_down(extent_dict['x_mean']+x_pad, 2)
    extent_dict['y_extent_min'] = max(round_decimals_down(extent_dict['y_mean']-y_pad, 2), 0)
    extent_dict['y_extent_max'] = round_decimals_down(extent_dict['y_mean']+y_pad, 2)
    
    return extent_dict


def groupby_team_expected_score(data):
    
    xScoreFor = data.groupby('Team').agg(
        numGames = ("Match_ID", "nunique"),
        numShotsFor = ("Team", "count"),
        xScoreFor = ("xScore", "sum")
    )
    data['Opponent'] = np.where(data['Team'] == data['Home_Team'], data['Away_Team'], data['Home_Team'])
    xScoreAgg = data.groupby('Opponent').agg(
        numShotsAgg = ("Opponent", "count"),
        xScoreAgg = ("xScore", "sum")
    )
    team_groupby = pd.merge(xScoreFor, xScoreAgg, left_index=True, right_index=True)

    team_groupby['xScoreDiff'] = team_groupby['xScoreFor'] - team_groupby['xScoreAgg']
    team_groupby['xScoreForPerShot'] = team_groupby['xScoreFor'] / team_groupby['numShotsFor']
    team_groupby['xScoreAggPerShot'] = team_groupby['xScoreAgg'] / team_groupby['numShotsAgg']
    team_groupby['xScoreForPerGame'] = team_groupby['xScoreFor'] / team_groupby['numGames']
    team_groupby['xScoreAggPerGame'] = team_groupby['xScoreAgg'] / team_groupby['numGames']

    team_groupby['xScoreForPerGameNorm'] = team_groupby['xScoreForPerGame'] / team_groupby['xScoreForPerGame'].mean()
    team_groupby['xScoreAggPerGameNorm'] = team_groupby['xScoreAggPerGame'] / team_groupby['xScoreAggPerGame'].mean()
     
    return team_groupby

def plot_diamond_scatter_plot(data, x, y, x_pad=0.2, y_pad=0.2, nticks = 8):
    
    extent_dict = get_diamond_plot_extents(data, x, y, x_pad, y_pad)
    
    fig = plt.figure(dpi=300)
    plot_extents = extent_dict['x_extent_min'], extent_dict['x_extent_max'], extent_dict['y_extent_min'], extent_dict['y_extent_max']
    transform = Affine2D().rotate_deg(135)
    helper = floating_axes.GridHelperCurveLinear(transform, plot_extents,
                                                grid_locator1=MaxNLocator(nbins=nticks),
                                                grid_locator2=MaxNLocator(nbins=nticks)
                                                )
    
    ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=helper)
    fig.add_subplot(ax)
    ax.axis[:].line.set_color("#FFFFFF")
    ax.axis[:].label.set_color("#FFFFFF")
    ax.axis[:].major_ticks.set_color("#FFFFFF")
    ax.axis[:].major_ticklabels.set(fontsize=6)
    ax.axis[:].label.set(fontsize=6)
    ax.axis['left'].major_ticklabels.set_visible(False)
    ax.axis['left'].label.set_visible(False)

    ax.axis['right'].major_ticklabels.set_visible(True)
    ax.axis['right'].label.set_visible(True)
    ax.axis['right'].label.set_axis_direction("top")
    ax.axis['right'].major_ticklabels.set_axis_direction("left")

    ax.axis['bottom'].set_axis_direction("bottom")
    ax.axis['bottom'].major_ticklabels.set_axis_direction("right")
    ax.axis['bottom'].label.set_axis_direction("top")

    ax.grid(visible=True, lw=0.2, ls=":", color="lightgrey")

    aux_ax = ax.get_aux_axes(transform)
    for i, team in enumerate(team_list):
        text = list()
        text.append(aux_ax.annotate(team, (data[x][i], data[y][i]), size=4, zorder=3))
        aux_ax.scatter(data[x][i], data[y][i], zorder=2, c=team_colours[team]['positive'], ec="w", lw=0.3)
                
        adjustText.adjust_text(text, ax=ax)

    # Quadrants
    aux_ax.vlines(x=extent_dict['x_mean'], ymin=extent_dict['y_extent_min'], ymax=extent_dict['y_extent_max'], color="w", lw=0.5, ls="--")
    aux_ax.hlines(y=extent_dict['y_mean'], xmin=extent_dict['x_extent_min'], xmax=extent_dict['x_extent_max'], color="w", lw=0.5, ls="--")

    return fig, ax

def groupby_player_expected_score(data, min_games = 20):
    
    player_groupby = data.groupby('Player').agg(
        numGames = ("Match_ID", "nunique"),
        numShots = ("Player", "count"),
        Score = ("Score", "sum"),
        xScore = ("xScore", "sum")
    )
    player_groupby = player_groupby[player_groupby['numGames'] > min_games]
    
    player_groupby['numGamesNorm'] = player_groupby['numGames'] / player_groupby['numGames'].max()    
    player_groupby['numShotsNorm'] = player_groupby['numShots'] / player_groupby['numShots'].max()    
    
    player_groupby['ScoreNorm'] = player_groupby['Score'] / player_groupby['Score'].max()    
    player_groupby['ScorePerShot'] = player_groupby['Score'] / player_groupby['numShots']
    player_groupby['ScorePerShotNorm'] = player_groupby['ScorePerShot'] / player_groupby['ScorePerShot'].max()
    player_groupby['ScorePerGame'] = player_groupby['Score'] / player_groupby['numGames']
    player_groupby['ScorePerGameNorm'] = player_groupby['ScorePerGame'] / player_groupby['ScorePerGame'].max()
    
    player_groupby['xScoreNorm'] = player_groupby['xScore'] / player_groupby['xScore'].max()    
    player_groupby['xScorePerShot'] = player_groupby['xScore'] / player_groupby['numShots']
    player_groupby['xScorePerShotNorm'] = player_groupby['xScorePerShot'] / player_groupby['xScorePerShot'].max()
    player_groupby['xScorePerGame'] = player_groupby['xScore'] / player_groupby['numGames']
    player_groupby['xScorePerGameNorm'] = player_groupby['xScorePerGame'] / player_groupby['xScorePerGame'].max()
    
    player_groupby['xScoreDiff'] = player_groupby['Score'] - player_groupby['xScore']
    player_groupby['xScoreDiffNorm'] = player_groupby['xScoreDiff'] / player_groupby['xScoreDiff'].max()
    player_groupby['xScoreDiffPerShot'] = player_groupby['xScoreDiff'] / player_groupby['numShots']
    player_groupby['xScoreDiffPerShotNorm'] = player_groupby['xScoreDiffPerShot'] / player_groupby['xScoreDiffPerShot'].max()
    player_groupby['xScoreDiffPerGame'] = player_groupby['xScoreDiff'] / player_groupby['numGames']
    player_groupby['xScoreDiffPerGameNorm'] = player_groupby['xScoreDiffPerGame'] / player_groupby['xScoreDiffPerGame'].max()
    
    return player_groupby

def plot_player_diamond_scatter_plot(data, x, y, x_pad, y_pad, nticks = 8, share_extent = False, annotate=False):
    
    extent_dict = get_diamond_plot_extents(data, x, y, x_pad, y_pad)
    
    if share_extent:
        extent_dict['y_extent_min'] = extent_dict['x_extent_min']
        extent_dict['y_extent_max'] = extent_dict['x_extent_max']
    
    fig = plt.figure(dpi=300)
    plot_extents = extent_dict['x_extent_min'], extent_dict['x_extent_max'], extent_dict['y_extent_min'], extent_dict['y_extent_max']
    scaler = (extent_dict['y_extent_max'] - extent_dict['y_extent_min']) / (extent_dict['x_extent_max'] - extent_dict['x_extent_min']) 
    transform = Affine2D().scale(scaler, 1).rotate_deg(45)
    helper = floating_axes.GridHelperCurveLinear(transform, plot_extents,
                                                grid_locator1=MaxNLocator(nbins=nticks),
                                                grid_locator2=MaxNLocator(nbins=nticks)
                                                )
    
    ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=helper)
    fig.add_subplot(ax)
    ax.axis[:].line.set_color("#FFFFFF")
    ax.axis[:].label.set_color("#FFFFFF")
    ax.axis[:].major_ticks.set_color("#FFFFFF")
    ax.axis[:].major_ticklabels.set(fontsize=6)
    ax.axis[:].label.set(fontsize=6)
    
    ax.axis['left'].major_ticklabels.set_visible(False)
    ax.axis['left'].label.set_visible(False)
    ax.axis['bottom'].major_ticklabels.set_visible(False)
    ax.axis['bottom'].label.set_visible(False)
    
    ax.axis['right'].major_ticklabels.set_visible(True)
    ax.axis['right'].label.set_visible(True)
    ax.axis['right'].label.set_axis_direction("top")
    ax.axis['right'].major_ticklabels.set_axis_direction("top")

    ax.axis['top'].major_ticklabels.set_visible(True)
    ax.axis['top'].label.set_visible(True)
    ax.axis['top'].set_axis_direction("top")
    ax.axis['top'].major_ticklabels.set_axis_direction("top")
    ax.axis['top'].label.set_axis_direction("left")

    ax.grid(visible=True, lw=0.2, ls=":", color="lightgrey")

    aux_ax = ax.get_aux_axes(transform)
    aux_ax.scatter(data[x], data[y], c=(data[x] + data[y])/2, cmap="RdPu", alpha=0.9, zorder=3, ec="w", lw=0.2)

    if annotate:
        text = list()
        data['sort'] = (data[x+"Norm"] + data[y+"Norm"]) / 2
        data_top5 = list(data.sort_values(by="sort", ascending=False).head(10).index)
        for i, name in enumerate(data_top5):
            text.append(aux_ax.annotate(name, (data.loc[name][x], data.loc[name][y]), size=4, zorder=3))
            
        adjustText.adjust_text(text, ax=ax)

    # Quadrants
    aux_ax.vlines(x=extent_dict['x_mean'], ymin=extent_dict['y_extent_min'], ymax=extent_dict['y_extent_max'], color="w", lw=0.5, ls="--")
    aux_ax.hlines(y=extent_dict['y_mean'], xmin=extent_dict['x_extent_min'], xmax=extent_dict['x_extent_max'], color="w", lw=0.5, ls="--")

    return fig, ax