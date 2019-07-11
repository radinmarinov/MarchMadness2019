# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 20:56:37 2019

@author: Radin
"""
import pandas as pd
import numpy as np
"""
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score

end_of_season_stats_df              = pd.DataFrame.from_csv("C:/Users/Radin/Documents/MarchMadness2019/end_of_season_stats.csv")
tourney_matchup_df = pd.DataFrame.from_csv("C:/Users/Radin/Documents/MarchMadness2019/tourney_matchup_df.csv")
stats = list(tourney_matchup_df)[4:len(list(tourney_matchup_df))]
X_train, X_test, y_train, y_test = train_test_split(tourney_matchup_df[stats], tourney_matchup_df['Team1_won'])
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(2),  solver='sgd', activation='relu')
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
mod_accuracy  = accuracy_score(y_test, predictions)
mod_auc       = roc_auc_score(y_test, predictions)
mod_precision = precision_score(y_test, predictions)
mod_recall    = recall_score(y_test, predictions)
"""
end_of_season_stats_df              = pd.DataFrame.from_csv("C:/Users/Radin/Documents/MarchMadness2019/end_of_season_stats.csv")
#Tourney Simulator
seeds_df                            = pd.DataFrame.from_csv("C:/Users/Radin/Documents/MarchMadness2019/Stage2DataFiles/NCAATourneySeeds.csv")
slots_df                            = pd.DataFrame.from_csv("C:/Users/Radin/Documents/MarchMadness2019/Stage2DataFiles/NCAATourneySlots.csv")
seedroundslots_df                   = pd.DataFrame.from_csv("C:/Users/Radin/Documents/MarchMadness2019/Stage2DataFiles/NCAATourneySeedRoundSlots.csv")
tourney_df                          = pd.DataFrame.from_csv("C:/Users/Radin/Documents/MarchMadness2019/Stage2DataFiles/NCAATourneyCompactResults.csv")

seeds_df                            = seeds_df.reset_index()
seeds_df['Season']                  = seeds_df['Season'].apply(str)
seeds_df['Season']                  = seeds_df.Season.str[:4]
seeds_df['Season']                  = seeds_df['Season'].astype(int)
seeds_df                            = seeds_df[seeds_df.Season > 2002]

slots_df                            = slots_df.reset_index()
slots_df['Season']                  = slots_df['Season'].apply(str)
slots_df['Season']                  = slots_df.Season.str[:4]
slots_df['Season']                  = slots_df['Season'].astype(int)
slots_df                            = slots_df[slots_df.Season > 2002]


seeds_df              = seeds_df.rename(index = str, columns={'Seed' : 'StrongSeed'})
slots_df                  = pd.merge(slots_df, seeds_df, how='left', on= ['Season','StrongSeed'])
seeds_df              = seeds_df.rename(index = str, columns={'StrongSeed' : 'WeakSeed'})
slots_df                  = pd.merge(slots_df, seeds_df, how='left', on= ['Season','WeakSeed'])

slots_df['Winner'] = 0

conferences_df                      = pd.DataFrame.from_csv("C:/Users/Radin/Documents/MarchMadness2019/Stage2DataFiles/TeamConferences.csv")
conferences_df                      = conferences_df.reset_index()
conferences_df['Season']            = conferences_df['Season'].apply(str)
conferences_df['Season']            = conferences_df.Season.str[:4]
conferences_df['Season']            = conferences_df['Season'].astype(int)
conferences_df                      = conferences_df[conferences_df.Season > 2002]

def get_winner(team1, team2, year):
    
    team1_stats = end_of_season_stats_df[(end_of_season_stats_df.Season == year) & (end_of_season_stats_df.teamID == team1)]
    team2_stats = end_of_season_stats_df[(end_of_season_stats_df.Season == year) & (end_of_season_stats_df.teamID == team2)]
    for stat in list(end_of_season_stats_df):
        team1_stats = team1_stats.rename(index = str, columns={stat : 'Team1_' + stat})
        team2_stats = team2_stats.rename(index = str, columns={stat : 'Team2_' + stat})
    team1_seed = seeds_df.WeakSeed[(seeds_df.Season == year) & (seeds_df.TeamID == team1)].iloc[0]
    team1_seed = int(team1_seed[1:3])
    team2_seed = seeds_df.WeakSeed[(seeds_df.Season == year) & (seeds_df.TeamID == team2)].iloc[0]
    team2_seed = int(team2_seed[1:3])
    team1_stats = team1_stats.drop('Team1_Season', axis=1)
    team2_stats = team2_stats.drop('Team2_Season', axis=1)
    matchup1 = pd.concat([team1_stats, team2_stats], axis=1, sort=False)
    matchup1['Team1_seed'] = team1_seed
    matchup1['Team2_seed'] = team2_seed
    for stat in list(end_of_season_stats_df)[2:len(list(end_of_season_stats_df))]:
        matchup1['diff_' + stat] = matchup1['Team1_' + stat] - matchup1['Team2_' + stat]
    matchup1['diff_seed'] = team1_seed - team2_seed
    
    matchup1['Season'] = year
    conferences_df            = conferences_df.rename(index= str, columns={'TeamID':'Team1_teamID'})
    matchup1                  = pd.merge(matchup1, conferences_df, how= 'left', on= ['Season', 'Team1_teamID'])
    conferences_df            = conferences_df.rename(index= str, columns={'Team1_teamID':'Team2_teamID'})
    matchup1                  = pd.merge(matchup1, conferences_df, how= 'left', on= ['Season', 'Team2_teamID'])
    matchup1 = matchup1.drop('Season', axis=1)
    matchup1 = pd.get_dummies(matchup1)

    missing_cols = list(set(tourney_matchup_df) - set(matchup1 ))
    missing_cols.remove('Team1_won')
    for col in missing_cols:
        matchup1[col] = 0
    
    return team1

def season_simulator(year):
    season = slots_df[slots_df.Season == year]
    winners = {}
    for slot in season.Slot[~np.isnan(season[['TeamID_x', 'TeamID_y']]).any(1)]:
        team1 = season.TeamID_x[season.Slot == slot].iloc[0]
        team2 = season.TeamID_y[season.Slot == slot].iloc[0]
        winners[slot] = get_winner(team1, team2, year)
        season.Winner[season.Slot == slot]  = winners[slot]
    while season[['TeamID_x', 'TeamID_y']].isnull().values.any():   
        for slot in season.Slot:
            team1 = season.TeamID_x[season.Slot == slot].iloc[0]
            team2 = season.TeamID_y[season.Slot == slot].iloc[0]
            if np.isnan(team1):
                season.TeamID_x[season.Slot == slot] = winners[season.StrongSeed[season.Slot == slot].iloc[0]]
                team1 = season.TeamID_x[season.Slot == slot].iloc[0]
            if np.isnan(team2):                    
                season.TeamID_y[season.Slot == slot] = winners[season.WeakSeed[season.Slot == slot].iloc[0]]
                team2 = season.TeamID_y[season.Slot == slot].iloc[0]
            season.Winner[season.Slot == slot]  = get_winner(team1, team2, year)
            winners[slot] = season.Winner[season.Slot == slot].iloc[0]
    return season
                    
                    
                    
                    