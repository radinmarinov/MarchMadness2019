# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 20:56:37 2019

@author: Radin
"""
import pandas as pd
"""
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score

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

slots_df['Winner'] = slots_df.TeamID_x
slots_df['Winner'] = None

for year in pd.unique(slots_df.Season):
    if year == 2003:
        season_2003 = slots_df[slots_df.Season == 2003]
        winners = {}
        winners['X16'] = season_2003.loc[season_2003.Slot=='X16', 'Winner'].iloc[0]
        season_2003.loc[season_2003.WeakSeed == 'X16', 'TeamID_y'] = winners['X16']
        while season_2003.TeamID_x.isnull().values.any():
            for slot in season_2003.Slot:
                if ~(season_2003.Winner[season_2003.Slot == slot].isnull().iloc[0]):
                    winners[slot] = season_2003.Winner[season_2003.Slot == slot].iloc[0]
                if season_2003.TeamID_x[season_2003.Slot == slot].isnull().iloc[0]:
                    season_2003.TeamID_x[season_2003.Slot == slot] = winners[season_2003.StrongSeed[season_2003.Slot == slot].iloc[0]]
                    season_2003.TeamID_y[season_2003.Slot == slot] = winners[season_2003.WeakSeed[season_2003.Slot == slot].iloc[0]]
                    season_2003.Winner[season_2003.Slot == slot]  = season_2003.TeamID_x[season_2003.Slot == slot].iloc[0] 
                    winners[slot] = season_2003.Winner[season_2003.Slot == slot].iloc[0]