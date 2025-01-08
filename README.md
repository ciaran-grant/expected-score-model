# Expected Score Model

** Updating this as part of preparation for 2025 season, with a full extra year of shots from 2024 to train on. **
New features:
1. Convert from 6 models down to 1 multiclass model.
2. Build other model types including Catboost, NN and Logisti Regression.
3. Add hyperparameter tuning, model evaluation and calibration for each.
4. Clean code.
5. Update visualisations.

expected-score-model is a Python library with useful functions and notebooks for tuning, training and evaluating multiple XGBoost models to predict the Expected Score of an AFL shot attempt.

Expected Score = (Goal Probability %) * 6 + (Behind Probability %)

There have been a few iterations, with varying successes. This is an ongoing project and problem to improve upon incrementally. But the aim of this project is to better understand the game of AFL using Expected notions rather an purely outcome based stats.

The Exepcted Scores will be used to create further metrics and deeper analysis to better understand the game.

Currently I am looking at building a multiclassification model to predict the probability of goal, behind and miss. The Expected Score is calculated from these.

## Usage

See notebooks folder for more comprehensive examples of usage.

### Model Building

#### Models
There are multiclassification models for the following model types:
- XGBoost
- Catboost
- MLP
- Logistic Regression

#### Features
The features for each model are mainly location based, with some information for previous actions to provide some context to the shot. 

#### Hyperparameter Tuning
Hyperparameters are selected using Optuna hyperparameter tuning process using cross validation within the training set. 

#### Model Fit
The final model is trained on the full training set with the best hyperparameters to be evaluated on the test set. The error metric used currently is log-loss since we want the model probabilities to be as accurate as possible rather than just the classification. We could consider using brier score as the error metric as well, this may remove the need for calibration step below.

#### Calibration
The outputs of each model will be class probabilities, however these probabilities might not align with the observed frequency of each label so we may need to calibrate some models.

#### Saving Predictions and Models
Predictions and the features used for the training and test data are saved down for use in the Model Evaluation. Previous model versions and predictions can be used for comparison to new models.

### Model Evaluation

Main model evaluation metrics looked at for multiclass classification here are logloss and brier loss score since we are interested in getting accurate probabilities.

Getting accurate average predictions are also a useful guide, but mainly interested in getting the most accurate calibrated probabilities.

For model interpretation and feature importance, there are SHAP Summary Plots and also Feature AvEs for every feature in the model.

Then finally we also have a calibration plot to check how calibrated the models are. When they predict a goal = 50%, do they actually see 50% of them be a goal?

### Expected Score

Once happy with model performance, we can calculate the Expected Score for each shot.

Expected Score = 6*Goal% + Behind%

### Analysis

#### Rolling Averages

** Out of Date **
![rolling-xs](notebooks/visualisations/rolling_expected_score/figures/20230718_afl_rolling_xS.png)

#### Shot Maps

** Out of Date **
![shot-map](notebooks/visualisations/expected_score_shot_map/figures/20230719_jeremy_cameron_shot_map.png)

#### Diamond Scatter Plots

** Out of Date **
![team-scatter](notebooks/visualisations/diamond_scatter_plot/figures/20230724_team_scatter.png)

![player-scatter](notebooks/visualisations/diamond_scatter_plot/figures/20230724_player_scatter.png)

#### Match Storytelling - Step / Lollipop Plots

** Out of Date **
![step-plot](notebooks/visualisations/expected_score_storytelling/figures/20230725_brisbane_sydney_step_plot.png)

![lollipop](notebooks/visualisations/expected_score_storytelling/figures/20230725_geelong_sydney_lollipop.png)

## Credits
Data sourced using a private R package. Credits to dgt23.

## CONTRIBUTING
I am currently working on this project so any bugs or suggestions are very welcome. Please contact me or create a pull request.

## License

[MIT](https://choosealicense.com/licenses/mit/)


