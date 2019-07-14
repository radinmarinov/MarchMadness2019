# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 20:56:37 2019

@author: Radin
"""
import pandas as pd
import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

random.seed(1)
end_of_season_stats_df              = pd.DataFrame.from_csv("C:/Users/Radin/Documents/MarchMadness2019/end_of_season_stats.csv")
tourney_matchup_df = pd.DataFrame.from_csv("C:/Users/Radin/Documents/MarchMadness2019/tourney_matchup_df.csv")
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
teams_df                            = pd.DataFrame.from_csv("C:/Users/Radin/Documents/MarchMadness2019/Stage2DataFiles/Teams.csv")
teams_df = teams_df.reset_index()
def get_model(year):
    stats = list(tourney_matchup_df)[4:len(list(tourney_matchup_df))]
    X_train = tourney_matchup_df[tourney_matchup_df.Season < year][stats]
    y_train = tourney_matchup_df[tourney_matchup_df.Season < year]['Team1_won']
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    mlp = MLPClassifier(hidden_layer_sizes=(2,5,3),  solver='sgd', activation='identity', random_state=1)
    mlp.fit(X_train,y_train)
    return [mlp, scaler]

def create_matchup(team1, team2, year, conferences_df):
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
    matchup = pd.concat([team1_stats, team2_stats], axis=1, sort=False)
    matchup['Team1_seed'] = team1_seed
    matchup['Team2_seed'] = team2_seed
    for stat in list(end_of_season_stats_df)[2:len(list(end_of_season_stats_df))]:
        matchup['diff_' + stat] = matchup['Team1_' + stat] - matchup['Team2_' + stat]
    matchup['diff_seed'] = team1_seed - team2_seed
    
    matchup['Season'] = year
    conferences_df = conferences_df.rename(columns={ conferences_df.columns[1]: "Team1_teamID" })
    matchup                  = pd.merge(matchup, conferences_df, how= 'left', on= ['Season', 'Team1_teamID'])
    conferences_df = conferences_df.rename(columns={ conferences_df.columns[1]: "Team2_teamID" })
    matchup                  = pd.merge(matchup, conferences_df, how= 'left', on= ['Season', 'Team2_teamID'])
    matchup = matchup.drop('Season', axis=1)
    matchup = pd.get_dummies(matchup)
    missing_cols = list(set(tourney_matchup_df) - set(matchup ))
    missing_cols.remove('Team1_won')
    for col in missing_cols:
        matchup[col] = 0
        
    return matchup

def get_winner(team1, team2, year, model):
    matchup1 = create_matchup(team1, team2, year, conferences_df)
    matchup2 = create_matchup(team2, team1, year, conferences_df)
    
    stats = list(tourney_matchup_df)[4:len(list(tourney_matchup_df))]

    mlp= model[0]
    scaler = model[1]
    X_test1 = matchup1[stats]
    X_test1 = scaler.transform(X_test1)
    predictions1 = mlp.predict(X_test1)
    probabilities_mlp1 = mlp.predict_proba(X_test1)
    
    X_test2 = matchup2[stats]
    X_test2 = scaler.transform(X_test2)
    predictions2 = mlp.predict(X_test2)
    probabilities_mlp2 = mlp.predict_proba(X_test2)
    
    if predictions1 == predictions2:
        if probabilities_mlp1[0][1] > probabilities_mlp2[0][1]:
            return team1
        if probabilities_mlp2[0][1] > probabilities_mlp1[0][1]:
            return team2
        else:
            if matchup1.diff_seed.iloc[0] < 0:
                return team1
            else:
                return team2
            
    if predictions1 ==1:
        return team1
    else: 
        return team2
    
def season_simulator(year):
    season = slots_df[slots_df.Season == year]
    winners = {}
    model = get_model(year)
    for slot in season.Slot[~np.isnan(season[['TeamID_x', 'TeamID_y']]).any(1)]:
        team1 = season.TeamID_x[season.Slot == slot].iloc[0]
        team2 = season.TeamID_y[season.Slot == slot].iloc[0]
        winners[slot] = get_winner(team1, team2, year, model)
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
            season.Winner[season.Slot == slot]  = get_winner(team1, team2, year, model)
            winners[slot] = season.Winner[season.Slot == slot].iloc[0]
    return season
                    
                    
test_season = season_simulator(2019)
test_season = pd.merge(test_season, teams_df, how= 'left', left_on=['Winner'], right_on=['TeamID'])              


##################
#Kaggle Submission
##################
def get_probability_kaggle(team1, team2, year, model):
    matchup1 = create_matchup(team1, team2, year, conferences_df)
    matchup2 = create_matchup(team2, team1, year, conferences_df)
    
    stats = list(tourney_matchup_df)[4:len(list(tourney_matchup_df))]

    mlp= model[0]
    scaler = model[1]
    X_test1 = matchup1[stats]
    X_test1 = scaler.transform(X_test1)
    probabilities_mlp1 = mlp.predict_proba(X_test1)
    
    X_test2 = matchup2[stats]
    X_test2 = scaler.transform(X_test2)
    probabilities_mlp2 = mlp.predict_proba(X_test2)
    
    return (probabilities_mlp1[0][1] + probabilities_mlp2[0][0]) / 2

def kaggle_submission(year):
    season = slots_df[slots_df.Season == year]
    teams = list(season.TeamID_x)
    teams.extend(list(season.TeamID_y))
    teams = pd.unique(teams)
    teams = [x for x in teams if str(x) != 'nan']
    teams = sorted(teams)
    teams = np.asarray(teams)
    model = get_model(year)
    submission = pd.DataFrame({'id': [],'pred':[]})
    for team1 in teams:
        team2s = teams[teams>team1]
        for team2 in team2s:
            id_ = str(year) + '_'+ str(team1)[0:4] + '_' + str(team2)[0:4]
            pred = get_probability_kaggle(team1, team2, year, model)
            submission = submission.append(pd.DataFrame({'id': id_
                           ,'pred':pred}, index=[0]))
            print(id_)
    return submission

s = kaggle_submission(2019)
s.to_csv("C:/Users/Radin/Documents/MarchMadness2019/predictions.csv", index=False)         
