# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:01:18 2019

@author: Radin
"""

import pandas as pd
import numpy as np

regular_season_df = pd.read_csv("C:/Users/Radin/Downloads/Data Science/DataFilesMM/RegularSeasonDetailedResults.csv")
regular_season_df['WPoss']  = regular_season_df['WFGA'] 
                                + regular_season_df['WTO'] 
                                + 0.44*regular_season_df['WFTA'] 
                                - regular_season_df['WOR']
regular_season_df['LPoss']  = regular_season_df['LFGA'] 
                                + regular_season_df['LTO'] 
                                + 0.44*regular_season_df['LFTA'] 
                                - regular_season_df['LOR']

end_of_season_stats = pd.DataFrame()
years = pd.unique(regular_season_df.Season)
for year in years:
    season = regular_season_df[regular_season_df.Season == year]
    teams = pd.unique(season.WTeamID.append(season.LTeamID))
    
    for team in teams:
        wins = season[season.WTeamID == team]
        losses = season[season.LTeamID == team]
        team_W = len(wins.index)
        team_L = len(losses.index)
        team_G = team_W + team_L
        team_Pace = (sum(wins.WPoss) + sum(losses.LPoss)) / team_G 
        team_Ortg = (sum(wins.WScore / wins.WPoss) + sum(losses.LScore / losses.LPoss)) / team_G
        team_Drtg = (sum(wins.LScore / wins.LPoss) + sum(losses.WScore / losses.WPoss)) / team_G
        team_FTr = (sum(wins.WFTA) + sum(losses.LFTA)) / (sum(wins.WFGA) + sum(losses.LFGA))
        team_3PAr = (sum(wins.WFGA3) + sum(losses.LFGA3)) / (sum(wins.WFGA) + sum(losses.LFGA))
        team_TSpct = (sum(wins.WScore) + sum(losses.LScore)) / (2 * (sum(wins.WFGA) + sum(losses.LFGA) + 0.44 *(sum(wins.WFTA) + sum(losses.LFTA))))
        team_TRBpct = (sum(wins.WOR) + sum(wins.WDR) + sum(losses.LOR) + sum(losses.LDR)) / (sum(wins.WOR) + sum(wins.WDR) + sum(losses.LOR) + sum(losses.LDR) + (sum(wins.LOR) + sum(wins.LDR) + sum(losses.WOR) + sum(losses.WDR)))
        team_ASTpct = (sum(wins.WAst / wins.WFGM) + sum(losses.LAst / losses.LFGM)) / team_G
        team_STLpct = (sum(wins.WStl) + sum(losses.LStl)) / (sum(wins.LPoss) + sum(losses.WPoss))
        team_BLKpct = (sum(wins.WBlk) + sum(losses.LBlk)) / (sum(wins.LFGA - wins.LFGA3) + sum(losses.WFGA - losses.WFGA3))
        team_eFGpct = (sum(wins.WFGM) + sum(0.5*wins.WFGM3) + sum(losses.LFGM) + sum(0.5*losses.LFGM3)) / (sum(wins.WFGA) + sum(losses.LFGA))
        team_TOVpct = (sum(wins.WTO) + sum(losses.LTO)) / (sum(wins.LPoss) + sum(losses.WPoss))
        team_ORBpct = (sum(wins.WOR) + sum(losses.LOR)) / (sum(wins.WOR) + sum(losses.LOR) + sum(wins.LDR) + sum(losses.WDR))
        teamFTperFG = (sum(wins.WFTM) + sum(losses.LFTM)) / (sum(wins.WFGA) + sum(losses.LFGA))
        
        oppo_Pace = (sum(wins.LPoss) + sum(losses.WPoss)) / team_G 
        oppo_Ortg = (sum(wins.LScore / wins.LPoss) + sum(losses.WScore / losses.WPoss)) / team_G
        oppo_Drtg = (sum(wins.WScore / wins.WPoss) + sum(losses.LScore / losses.LPoss)) / team_G
        oppo_FTr = (sum(wins.LFTA) + sum(losses.WFTA)) / (sum(wins.LFGA) + sum(losses.WFGA))
        oppo_3PAr = (sum(wins.LFGA3) + sum(losses.WFGA3)) / (sum(wins.LFGA) + sum(losses.WFGA))
        oppo_TSpct = (sum(wins.LScore) + sum(losses.WScore)) / (2 * (sum(wins.LFGA) + sum(losses.WFGA) + 0.44 *(sum(wins.LFTA) + sum(losses.WFTA))))
        oppo_TRBpct = (sum(wins.LOR) + sum(wins.LDR) + sum(losses.WOR) + sum(losses.WDR)) / (sum(wins.WOR) + sum(wins.WDR) + sum(losses.LOR) + sum(losses.LDR) + (sum(wins.LOR) + sum(wins.LDR) + sum(losses.WOR) + sum(losses.WDR)))
        oppo_ASTpct = (sum(wins.LAst / wins.LFGM) + sum(losses.WAst / losses.WFGM)) / team_G
        oppo_STLpct = (sum(wins.LStl) + sum(losses.WStl)) / (sum(wins.WPoss) + sum(losses.LPoss))
        oppo_BLKpct = (sum(wins.LBlk) + sum(losses.WBlk)) / (sum(wins.WFGA - wins.WFGA3) + sum(losses.LFGA - losses.LFGA3))
        oppo_eFGpct = (sum(wins.LFGM) + sum(0.5*wins.LFGM3) + sum(losses.WFGM) + sum(0.5*losses.WFGM3)) / (sum(wins.LFGA) + sum(losses.WFGA))
        oppo_TOVpct = (sum(wins.LTO) + sum(losses.WTO)) / (sum(wins.WPoss) + sum(losses.LPoss))
        oppo_ORBpct = (sum(wins.LOR) + sum(losses.WOR)) / (sum(wins.LOR) + sum(losses.WOR) + sum(wins.WDR) + sum(losses.LDR))
        oppoFTperFG = (sum(wins.LFTM) + sum(losses.WFTM)) / (sum(wins.LFGA) + sum(losses.WFGA))
