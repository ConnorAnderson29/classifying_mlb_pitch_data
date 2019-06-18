# Classifying MLB In-Game Pitches
## By Werlindo Mangrobang and Connor Anderson

We utilized MLB play-by-play stats from 2018 Seattle Mariner games
to create a classification model that can correctly classify pitches based on thier spin rate, break angle, break angle, pitch speed etc... 


# Getting Pitch-level Data
The goal is to obtain data related to a single pitch in an MLB game, given our decided parameters. For example, "the 2nd pitch of the 3rd at-bat of the bottom of the first inning" (to demonstrate the granularity).

---

## Libraries


```python
#DATA WRANGLING
import pandas as pd # Dataframes
from pandas.io.json import json_normalize # JSON wrangler
import statsapi # Python wrapper MLB data API
#DATA STORAGE
#from sqlalchemy import create_engine # SQL helper
import psycopg2 as psql #PostgreSQL DBs
#DATA MANIPULATION AND MODELLING
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn_pandas import DataFrameMapper, FunctionTransformer, gen_features, pipeline
from sklearn_pandas.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import pitch_functions
import xgboost as xgb
import os

```


```python
os.environ['KMP_DUPLICATE_LIB_OK']='True'
```

---

## Data Retrieval

In order to retrieval publicly available data from the Major League Baseball Stats API, we will use a module called `statsapi`.

### 1. Determine list of games


```python
schedule = statsapi.schedule(start_date="03/28/2018", end_date="09/30/2018", team=136)
full = json_normalize(schedule)
list_game_pks = full['game_id']

```


```python
gamepks_2018 = list(list_game_pks.unique())
len(gamepks_2018)
```




    162




```python
schedule = statsapi.schedule(start_date="08/01/2018", end_date="09/30/2018", team=136)
games_2018 = json_normalize(schedule)
august_2018 = games_2018['game_id']

```


```python
validation_list = list(august_2018.unique())
len(validation_list)
```




    55



### 2. Retrieve play-by-play data for game(s).

for index, row in play_ev.iterrows(): #Just using first 2 rows for testing

        # saw playEvents is a nested list, so json_normalize it
        play_events_df = json_normalize(row['playEvents'])
play_events_df.columns


```python
# Get one game from API
list_for_new_df = []
#gamepks = [566389]
for game in gamepks_2018:
    #print(game)
    curr_game = statsapi.get('game_playByPlay',{'gamePk':game})

    ### 3. Extract play-by-play data and store into dataframe.

    # Only care about the allPlays key 
    curr_plays = curr_game.get('allPlays')

    # Coerce all plays into a df
    curr_plays_df = json_normalize(curr_plays)

    ###################################
    # Build target table
    ###################################


    # Data from allPlays
    ap_sel_cols = ['about.atBatIndex', 'matchup.batSide.code', 'matchup.pitchHand.code', 'count.balls'
              ,'count.strikes', 'count.outs']

    # Data from playEvents
    plev_sel_cols = ['details.type.code', 'details.type.description', 
            'details.call.code', 'details.call.description', 
            'details.isBall', 'isPitch', 'details.isStrike'
            ,'pitchData.breaks.breakAngle'
            ,'pitchData.breaks.breakLength', 'pitchData.breaks.breakY'
            ,'pitchData.breaks.spinDirection', 'pitchData.breaks.spinRate'
            ,'pitchData.coordinates.aX'
            , 'pitchData.coordinates.aY','pitchData.coordinates.aZ', 'pitchData.coordinates.pX'
            , 'pitchData.coordinates.pZ', 'pitchData.coordinates.pfxX', 'pitchData.coordinates.pfxZ'
            , 'pitchData.coordinates.vX0', 'pitchData.coordinates.vY0', 'pitchData.coordinates.vZ0'
            , 'pitchData.coordinates.x', 'pitchData.coordinates.x0', 'pitchData.coordinates.y'
            , 'pitchData.coordinates.y0','pitchData.coordinates.z0', 'pitchData.endSpeed'
            , 'pitchData.startSpeed', 'pitchNumber', 'pitchData.zone'
           ]

    # Now go through each row. If there is nested list, json_normalize it
    #for index, row in test_df.head(2).iterrows(): #Just using first 2 rows for testing
    for index, row in curr_plays_df.iterrows(): #Just using first 2 rows for testing

        # saw playEvents is a nested list, so json_normalize it
        play_events_df = json_normalize(row['playEvents'])

        #     # look at runners
        #     runners_df = json_normalize(row['runners'])

        # Loop through THIS NESTED dataframe and NOW build the row for the new df    
        for plev_ind, plev_row in play_events_df.iterrows():

            # Instantiate new dict, which will be a single row in target df
            curr_dict = {}
            curr_dict['game_pk'] = game

            # Loop through each list, adding their respective values to curr_dict
            for col_ap in ap_sel_cols:
                if col_ap in curr_plays_df.columns:
                    curr_dict[col_ap] = row[col_ap]
                else:
                    curr_dict[col_ap] = np.nan
                #print(row['about.atBatIndex'])

            for col_plev in plev_sel_cols:
                if col_plev in play_events_df.columns:
                    curr_dict[col_plev] = plev_row[col_plev]
                else:
                    curr_dict[col_plev] = np.nan

            # collect row dictionary into list
            list_for_new_df.append(curr_dict)

```

We have gathered and organzied 49 Seattle Mariner games from the 2019 season. The only games not included are the games from March 20th to March 23, which were played in Japan before MLB's official opening day


```python
# Proof of concept on target dataframe
pitches_df = pd.DataFrame(list_for_new_df)
# print(len(pitches_df))
# gamepks_2019[:2]
# pitches_df[pitches_df['game_pk'].isin([566372, 566373])]
pitches_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50547 entries, 0 to 50546
    Data columns (total 38 columns):
    about.atBatIndex                  50547 non-null int64
    count.balls                       50547 non-null int64
    count.outs                        50547 non-null int64
    count.strikes                     50547 non-null int64
    details.call.code                 46295 non-null object
    details.call.description          46295 non-null object
    details.isBall                    46295 non-null object
    details.isStrike                  46295 non-null object
    details.type.code                 46131 non-null object
    details.type.description          46131 non-null object
    game_pk                           50547 non-null int64
    isPitch                           50547 non-null bool
    matchup.batSide.code              50547 non-null object
    matchup.pitchHand.code            50547 non-null object
    pitchData.breaks.breakAngle       46119 non-null float64
    pitchData.breaks.breakLength      46119 non-null float64
    pitchData.breaks.breakY           46119 non-null float64
    pitchData.breaks.spinDirection    46119 non-null float64
    pitchData.breaks.spinRate         45415 non-null float64
    pitchData.coordinates.aX          46119 non-null float64
    pitchData.coordinates.aY          46119 non-null float64
    pitchData.coordinates.aZ          46119 non-null float64
    pitchData.coordinates.pX          46119 non-null float64
    pitchData.coordinates.pZ          46119 non-null float64
    pitchData.coordinates.pfxX        46119 non-null float64
    pitchData.coordinates.pfxZ        46119 non-null float64
    pitchData.coordinates.vX0         46119 non-null float64
    pitchData.coordinates.vY0         46119 non-null float64
    pitchData.coordinates.vZ0         46119 non-null float64
    pitchData.coordinates.x           46295 non-null float64
    pitchData.coordinates.x0          46119 non-null float64
    pitchData.coordinates.y           46295 non-null float64
    pitchData.coordinates.y0          46119 non-null float64
    pitchData.coordinates.z0          46119 non-null float64
    pitchData.endSpeed                46119 non-null float64
    pitchData.startSpeed              46119 non-null float64
    pitchData.zone                    46119 non-null float64
    pitchNumber                       46295 non-null float64
    dtypes: bool(1), float64(24), int64(5), object(8)
    memory usage: 14.3+ MB



```python
temp_df = pitches_df.dropna()
temp = temp_df[temp_df['game_pk'].isin(august_2018)]
temp.shape
```




    (15517, 38)



# EDA

Dropping NA values in order to vizualize data


```python
pitches_df.shape
pitches_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>about.atBatIndex</th>
      <th>count.balls</th>
      <th>count.outs</th>
      <th>count.strikes</th>
      <th>details.call.code</th>
      <th>details.call.description</th>
      <th>details.isBall</th>
      <th>details.isStrike</th>
      <th>details.type.code</th>
      <th>details.type.description</th>
      <th>...</th>
      <th>pitchData.coordinates.vZ0</th>
      <th>pitchData.coordinates.x</th>
      <th>pitchData.coordinates.x0</th>
      <th>pitchData.coordinates.y</th>
      <th>pitchData.coordinates.y0</th>
      <th>pitchData.coordinates.z0</th>
      <th>pitchData.endSpeed</th>
      <th>pitchData.startSpeed</th>
      <th>pitchData.zone</th>
      <th>pitchNumber</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>B</td>
      <td>Ball - Called</td>
      <td>True</td>
      <td>False</td>
      <td>FF</td>
      <td>Four-Seam Fastball</td>
      <td>...</td>
      <td>-9.54</td>
      <td>146.72</td>
      <td>-2.46</td>
      <td>216.58</td>
      <td>50.0</td>
      <td>5.64</td>
      <td>82.3</td>
      <td>89.6</td>
      <td>13.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>X</td>
      <td>Hit Into Play - Out(s)</td>
      <td>False</td>
      <td>False</td>
      <td>SI</td>
      <td>Sinker</td>
      <td>...</td>
      <td>-7.10</td>
      <td>117.07</td>
      <td>-2.07</td>
      <td>200.69</td>
      <td>50.0</td>
      <td>5.80</td>
      <td>83.1</td>
      <td>90.6</td>
      <td>13.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>B</td>
      <td>Ball - Called</td>
      <td>True</td>
      <td>False</td>
      <td>SI</td>
      <td>Sinker</td>
      <td>...</td>
      <td>-4.91</td>
      <td>156.29</td>
      <td>-2.20</td>
      <td>174.75</td>
      <td>50.0</td>
      <td>5.90</td>
      <td>83.6</td>
      <td>91.4</td>
      <td>13.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>X</td>
      <td>Hit Into Play - Out(s)</td>
      <td>False</td>
      <td>False</td>
      <td>SI</td>
      <td>Sinker</td>
      <td>...</td>
      <td>-6.28</td>
      <td>132.08</td>
      <td>-2.14</td>
      <td>198.39</td>
      <td>50.0</td>
      <td>5.72</td>
      <td>82.8</td>
      <td>90.9</td>
      <td>7.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>S</td>
      <td>Strike - Swinging</td>
      <td>False</td>
      <td>True</td>
      <td>SI</td>
      <td>Sinker</td>
      <td>...</td>
      <td>-5.99</td>
      <td>113.29</td>
      <td>-2.07</td>
      <td>189.93</td>
      <td>50.0</td>
      <td>5.63</td>
      <td>83.1</td>
      <td>90.9</td>
      <td>8.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
pitches_df_clean = pitches_df.dropna(axis=0).copy()
pitches_df_clean.shape
```




    (45415, 38)




```python
pitches_df_clean = pitches_df_clean.loc[pitches_df_clean['pitchData.coordinates.pX'] < 25]

```


```python
g = sns.FacetGrid(pitches_df_clean, row='details.call.description', height=4, aspect=2
                 ,sharex=False, hue='details.call.description'
                 ,margin_titles=True)
g = g.map(sns.regplot, 'pitchData.coordinates.pX', 'pitchData.coordinates.pZ'
         ,scatter_kws={'alpha':0.3})
```


![png](output_23_0.png)


## PITCH COORDINATES VS PITCH TYPE


```python
g = sns.FacetGrid(pitches_df_clean, row='details.type.description', height=4, aspect=2
                  ,col='matchup.pitchHand.code'
                 ,sharex=False, hue='details.type.description'
                 ,margin_titles=True)
g = g.map(sns.regplot, 'pitchData.coordinates.pX', 'pitchData.coordinates.pZ'
         ,scatter_kws={'alpha':0.3})
```


![png](output_25_0.png)


## PITCH BREAK ANGLE


```python
g = sns.FacetGrid(pitches_df_clean, row='details.type.description', height=4, aspect=2
                 ,sharex=True, hue='details.type.description'
                 ,margin_titles=True)
g = g.map(sns.kdeplot, 'pitchData.breaks.breakAngle')
```


![png](output_27_0.png)


## PITCH BREAK LENGTH


```python
g = sns.FacetGrid(pitches_df_clean, row='details.type.description', height=4, aspect=2
                 ,sharex=True, hue='details.type.description'
                 ,margin_titles=True)
g = g.map(sns.kdeplot, 'pitchData.breaks.breakLength')
```

    /anaconda3/envs/learn-env/lib/python3.6/site-packages/statsmodels/nonparametric/kde.py:488: RuntimeWarning:
    
    invalid value encountered in true_divide
    
    /anaconda3/envs/learn-env/lib/python3.6/site-packages/statsmodels/nonparametric/kdetools.py:34: RuntimeWarning:
    
    invalid value encountered in double_scalars
    



![png](output_29_1.png)


## PITCH SPIN RATE VS SPEED


```python
g = sns.FacetGrid(pitches_df_clean, row='details.type.description', height=6
                 ,sharex=False, hue='details.type.description'
                 ,margin_titles=True)
g = g.map(sns.regplot, 'pitchData.startSpeed', 'pitchData.breaks.spinRate'
         ,scatter_kws={'alpha':0.3})
```


![png](output_31_0.png)


# Model 1: 2018 Seattle Mariners Pitch Data 
## ( Reg Season through July ~107 Games)


```python
new_pitches = pitches_df_clean.dropna().copy()
print(new_pitches.shape)
#new_pitches.head().T
pitches_df_clean.shape
```

    (45415, 38)





    (45415, 38)




```python
validation_df = new_pitches[new_pitches['game_pk'].isin(august_2018)]
validation_df.shape
```




    (15517, 38)




```python
modeling_df = new_pitches[~new_pitches['game_pk'].isin(august_2018)]
modeling_df.shape
```




    (29898, 38)




```python
target = modeling_df['details.type.code']
```


```python
target_val = validation_df['details.type.code']
```


```python
#predictors_val = validation_df.drop(['details.type.code'], axis=1).copy()
```


```python
#predictors = predictors[mapper_list].copy()
predictors_val = validation_df.drop(['details.type.code','details.call.description'
                               , 'details.type.description', 'game_pk'], axis=1)


```


```python
predictors_val.shape
```




    (15517, 34)




```python
#predictors = predictors[mapper_list].copy()
predictors = modeling_df.drop(['details.type.code','details.call.description'
                               , 'details.type.description', 'game_pk'], axis=1)


```


```python
#predictors = predictors[mapper_list].copy()
pred_temp = predictors.copy()
```


```python
old_names = pred_temp.columns
```


```python
new_names = dict((nm,0) for nm in old_names)
```


```python
old_names[:15]
```




    Index(['about.atBatIndex', 'count.balls', 'count.outs', 'count.strikes',
           'details.call.code', 'details.isBall', 'details.isStrike', 'isPitch',
           'matchup.batSide.code', 'matchup.pitchHand.code',
           'pitchData.breaks.breakAngle', 'pitchData.breaks.breakLength',
           'pitchData.breaks.breakY', 'pitchData.breaks.spinDirection',
           'pitchData.breaks.spinRate'],
          dtype='object')




```python
# Create new column names by hand
new_names['about.atBatIndex'] = 'at_bat_id'
new_names['count.balls'] = 'balls'
new_names['count.outs'] = 'outs'
new_names['count.strikes'] = 'strikes'
new_names['details.call.code'] = 'pitch_called'
new_names['details.isBall'] = 'is_ball'
new_names['details.isStrike'] = 'is_strike'
new_names['isPitch'] = 'is_pitch'
new_names['matchup.pitchHand.code'] = 'pitch_hand'
new_names['matchup.batSide.code'] = 'bat_side'
new_names['pitchData.breaks.breakAngle'] = 'break_angle'
new_names['pitchData.breaks.breakLength'] = 'break_length'
new_names['pitchData.breaks.breakY'] = 'break_Y'
new_names['pitchData.breaks.spinDirection'] = 'spin_direction'
new_names['pitchData.breaks.spinRate'] = 'spin_rate'
new_names['pitchData.coordinates.aX'] = 'coord_aX'
new_names['pitchData.coordinates.aY'] = 'coord_aY'
new_names['pitchData.coordinates.aZ'] = 'coord_aZ'
new_names['pitchData.coordinates.pX'] = 'coord_pX'
new_names['pitchData.coordinates.pZ'] = 'coord_pZ'
new_names['pitchData.coordinates.pfxX'] = 'coord_pfxX'
new_names['pitchData.coordinates.pfxZ'] = 'coord_pfxZ'
new_names['pitchData.coordinates.vX0'] = 'coord_vX0'
new_names['pitchData.coordinates.vY0'] = 'coord_vY0'
new_names['pitchData.coordinates.vZ0'] = 'coord_vZ0'
new_names['pitchData.coordinates.x'] = 'coord_x'
new_names['pitchData.coordinates.x0'] = 'coord_x0'
new_names['pitchData.coordinates.y'] = 'coord_y'
new_names['pitchData.coordinates.y0'] = 'coord_y0'
new_names['pitchData.coordinates.z0'] = 'coord_z0'
new_names['pitchData.startSpeed'] = 'speed_start'
new_names['pitchData.endSpeed'] = 'speed_end'
new_names['pitchData.zone'] = 'pitch_zone'
new_names['pitchNumber'] = 'pitch_num'
```


```python
new_names
```




    {'about.atBatIndex': 'at_bat_id',
     'count.balls': 'balls',
     'count.outs': 'outs',
     'count.strikes': 'strikes',
     'details.call.code': 'pitch_called',
     'details.isBall': 'is_ball',
     'details.isStrike': 'is_strike',
     'isPitch': 'is_pitch',
     'matchup.batSide.code': 'bat_side',
     'matchup.pitchHand.code': 'pitch_hand',
     'pitchData.breaks.breakAngle': 'break_angle',
     'pitchData.breaks.breakLength': 'break_length',
     'pitchData.breaks.breakY': 'break_Y',
     'pitchData.breaks.spinDirection': 'spin_direction',
     'pitchData.breaks.spinRate': 'spin_rate',
     'pitchData.coordinates.aX': 'coord_aX',
     'pitchData.coordinates.aY': 'coord_aY',
     'pitchData.coordinates.aZ': 'coord_aZ',
     'pitchData.coordinates.pX': 'coord_pX',
     'pitchData.coordinates.pZ': 'coord_pZ',
     'pitchData.coordinates.pfxX': 'coord_pfxX',
     'pitchData.coordinates.pfxZ': 'coord_pfxZ',
     'pitchData.coordinates.vX0': 'coord_vX0',
     'pitchData.coordinates.vY0': 'coord_vY0',
     'pitchData.coordinates.vZ0': 'coord_vZ0',
     'pitchData.coordinates.x': 'coord_x',
     'pitchData.coordinates.x0': 'coord_x0',
     'pitchData.coordinates.y': 'coord_y',
     'pitchData.coordinates.y0': 'coord_y0',
     'pitchData.coordinates.z0': 'coord_z0',
     'pitchData.endSpeed': 'speed_end',
     'pitchData.startSpeed': 'speed_start',
     'pitchData.zone': 'pitch_zone',
     'pitchNumber': 'pitch_num'}




```python
pred_temp.rename(columns=new_names, inplace=True)
```


```python
pred_temp.columns
```




    Index(['at_bat_id', 'balls', 'outs', 'strikes', 'pitch_called', 'is_ball',
           'is_strike', 'is_pitch', 'bat_side', 'pitch_hand', 'break_angle',
           'break_length', 'break_Y', 'spin_direction', 'spin_rate', 'coord_aX',
           'coord_aY', 'coord_aZ', 'coord_pX', 'coord_pZ', 'coord_pfxX',
           'coord_pfxZ', 'coord_vX0', 'coord_vY0', 'coord_vZ0', 'coord_x',
           'coord_x0', 'coord_y', 'coord_y0', 'coord_z0', 'speed_end',
           'speed_start', 'pitch_zone', 'pitch_num'],
          dtype='object')




```python
predictors = pred_temp.copy()
```


```python
predictors_val.rename(columns=new_names, inplace=True)
```


```python
predictors_val.columns
```




    Index(['at_bat_id', 'balls', 'outs', 'strikes', 'pitch_called', 'is_ball',
           'is_strike', 'is_pitch', 'bat_side', 'pitch_hand', 'break_angle',
           'break_length', 'break_Y', 'spin_direction', 'spin_rate', 'coord_aX',
           'coord_aY', 'coord_aZ', 'coord_pX', 'coord_pZ', 'coord_pfxX',
           'coord_pfxZ', 'coord_vX0', 'coord_vY0', 'coord_vZ0', 'coord_x',
           'coord_x0', 'coord_y', 'coord_y0', 'coord_z0', 'speed_end',
           'speed_start', 'pitch_zone', 'pitch_num'],
          dtype='object')




```python
predictors_val.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>at_bat_id</th>
      <th>balls</th>
      <th>outs</th>
      <th>strikes</th>
      <th>pitch_called</th>
      <th>is_ball</th>
      <th>is_strike</th>
      <th>is_pitch</th>
      <th>bat_side</th>
      <th>pitch_hand</th>
      <th>...</th>
      <th>coord_vZ0</th>
      <th>coord_x</th>
      <th>coord_x0</th>
      <th>coord_y</th>
      <th>coord_y0</th>
      <th>coord_z0</th>
      <th>speed_end</th>
      <th>speed_start</th>
      <th>pitch_zone</th>
      <th>pitch_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33256</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>B</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>L</td>
      <td>L</td>
      <td>...</td>
      <td>-6.75</td>
      <td>148.97</td>
      <td>2.13</td>
      <td>199.87</td>
      <td>50.0</td>
      <td>5.51</td>
      <td>78.0</td>
      <td>85.1</td>
      <td>13.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>33257</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>B</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>L</td>
      <td>L</td>
      <td>...</td>
      <td>-8.50</td>
      <td>142.65</td>
      <td>1.94</td>
      <td>226.04</td>
      <td>50.0</td>
      <td>5.49</td>
      <td>78.7</td>
      <td>85.7</td>
      <td>13.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>33258</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>S</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>L</td>
      <td>L</td>
      <td>...</td>
      <td>-2.46</td>
      <td>127.46</td>
      <td>2.17</td>
      <td>153.67</td>
      <td>50.0</td>
      <td>5.70</td>
      <td>77.8</td>
      <td>85.5</td>
      <td>11.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>33259</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>B</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>L</td>
      <td>L</td>
      <td>...</td>
      <td>-0.46</td>
      <td>105.42</td>
      <td>2.56</td>
      <td>150.66</td>
      <td>50.0</td>
      <td>5.62</td>
      <td>74.8</td>
      <td>80.9</td>
      <td>12.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>33260</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>S</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>L</td>
      <td>L</td>
      <td>...</td>
      <td>-3.49</td>
      <td>133.73</td>
      <td>2.30</td>
      <td>173.77</td>
      <td>50.0</td>
      <td>5.55</td>
      <td>77.2</td>
      <td>84.6</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>



We decided to create a pipeline to make our code more efficient. We will be able to feature engineer and then apply our data to the pipeline and it will give us our most productive model.


```python
numeric_features = list(predictors.select_dtypes(exclude='object'))
numeric_transformer = Pipeline(steps=[('keeper', None)])
```


```python
cat_features = list(predictors.select_dtypes(include='object'))
cat_transfomer = Pipeline(steps=[('onehot', OneHotEncoder())])
```


```python
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                              ('cat', cat_transfomer, cat_features)])
```


```python
classifiers = [DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100, max_depth=5), xgb.XGBClassifier(),
               GradientBoostingClassifier(n_estimators=100)]
```


```python
X_train, X_test, y_train, y_test = train_test_split(predictors, target, random_state=10)
```


```python
cv_list = []

for classifier in classifiers:
    clf1 = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', classifier)])
    clf1.fit(X_train, y_train)
    cv_scores = cross_val_score(clf1, X_train, y_train, cv=5)
    cv_list.append(cv_scores)
    one_hot_names = list(clf1.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names())
    final_feats = numeric_features + one_hot_names
    print(classifier)
    print('\n')
    print('Training Metrics')
    pitch_functions.calc_acc_and_f1_score(y_train, clf1.predict(X_train))
    print('\n')
    print('Testing Metrics')
    pitch_functions.calc_acc_and_f1_score(y_test, clf1.predict(X_test))
    print('\n')
    print('Average Cross Val Score, k=5')
    print('{:.3}'.format(np.mean(cv_scores)))
    
    ifeats = clf1.named_steps['classifier'].feature_importances_
    feature_importance = pd.DataFrame(ifeats, index=final_feats,
                                 columns = ['importance']).sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10,10))

    ax.barh(feature_importance.head(10).index, width=feature_importance.head(10).importance)
    plt.gca().invert_yaxis()
    plt.title('Feature Importance')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.show()

```

    /anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/model_selection/_split.py:652: Warning:
    
    The least populated class in y has only 3 members, which is too few. The minimum number of members in any class cannot be less than n_splits=5.
    


    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')
    
    
    Training Metrics
    Accuracy:1.000
    F1-Score: 1.000
    AUC: 1.000
    
    
    Testing Metrics
    Accuracy:0.853
    F1-Score: 0.853
    AUC: 0.846
    
    
    Average Cross Val Score, k=5
    0.84


    /anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning:
    
    F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
    



![png](output_60_3.png)


    /anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/model_selection/_split.py:652: Warning:
    
    The least populated class in y has only 3 members, which is too few. The minimum number of members in any class cannot be less than n_splits=5.
    


    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=5, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
    
    
    Training Metrics


    /anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning:
    
    F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
    


    Accuracy:0.778
    F1-Score: 0.757
    AUC: 0.720
    
    
    Testing Metrics
    Accuracy:0.791
    F1-Score: 0.772
    AUC: 0.724
    
    
    Average Cross Val Score, k=5
    0.772


    /anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning:
    
    F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
    



![png](output_60_9.png)


    /anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/model_selection/_split.py:652: Warning:
    
    The least populated class in y has only 3 members, which is too few. The minimum number of members in any class cannot be less than n_splits=5.
    


    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
           max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
           n_jobs=1, nthread=None, objective='multi:softprob', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=True, subsample=1)
    
    
    Training Metrics
    Accuracy:0.868
    F1-Score: 0.867
    AUC: 0.889
    
    
    Testing Metrics


    /anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning:
    
    F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
    


    Accuracy:0.865
    F1-Score: 0.863
    AUC: 0.804
    
    
    Average Cross Val Score, k=5
    0.849



![png](output_60_14.png)


    /anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/model_selection/_split.py:652: Warning:
    
    The least populated class in y has only 3 members, which is too few. The minimum number of members in any class cannot be less than n_splits=5.
    


    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, n_estimators=100,
                  n_iter_no_change=None, presort='auto', random_state=None,
                  subsample=1.0, tol=0.0001, validation_fraction=0.1,
                  verbose=0, warm_start=False)
    
    
    Training Metrics
    Accuracy:0.912
    F1-Score: 0.911
    AUC: 0.952
    
    
    Testing Metrics
    Accuracy:0.885
    F1-Score: 0.884
    AUC: 0.821
    
    
    Average Cross Val Score, k=5
    0.871


    /anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning:
    
    F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
    



![png](output_60_18.png)


# Plotting Pitch Prediction Accuracy 
## Using Validation Set - Aug/Sept 2018 - 55 games


```python
prediction1 = pd.DataFrame(clf1.predict(predictors_val))
prediction1.reset_index(drop=True, inplace=True)
```


```python
prediction1.shape
```




    (15517, 1)




```python
target_val = pd.DataFrame(target_val)
target_val.reset_index(drop=True, inplace=True)
```


```python
results_df = pd.concat([target_val, prediction1], axis=1, ignore_index=True)
results_df.columns=['actual','predicted']
results_df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actual</th>
      <th>predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FF</td>
      <td>FS</td>
    </tr>
  </tbody>
</table>
</div>




```python
import plotly_express as px

px.histogram(results_df, x='actual', y='actual'
             ,color='predicted', histfunc='count'
             ,title='Actual vs Predicted Pitches (Validation Set: August & September 2018)'
            )
```


<div>
        
        
            <div id="1b73b614-ed6d-4989-9cc3-2ddff29f31e8" class="plotly-graph-div" style="height:600px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("1b73b614-ed6d-4989-9cc3-2ddff29f31e8")) {
                    Plotly.newPlot(
                        '1b73b614-ed6d-4989-9cc3-2ddff29f31e8',
                        [{"alignmentgroup": "True", "histfunc": "count", "hoverlabel": {"namelength": 0}, "hovertemplate": "predicted=FS<br>actual=%{x}<br>count of actual=%{y}", "legendgroup": "predicted=FS", "marker": {"color": "#636efa"}, "name": "predicted=FS", "offsetgroup": "predicted=FS", "orientation": "v", "showlegend": true, "type": "histogram", "uid": "6a08aae8-db35-467a-a3d4-94aed38473af", "x": ["FF", "FS", "FF", "CH", "FS", "FS", "FS", "FT", "CH", "SI", "CH", "KC", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "KC", "SI", "CH", "FS", "FT", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FT", "FS", "FS", "FS", "FS", "FS", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SL", "CH", "SI", "FT", "FT", "FT", "FT", "CH", "CH"], "xaxis": "x", "y": ["FF", "FS", "FF", "CH", "FS", "FS", "FS", "FT", "CH", "SI", "CH", "KC", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "KC", "SI", "CH", "FS", "FT", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FT", "FS", "FS", "FS", "FS", "FS", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SL", "CH", "SI", "FT", "FT", "FT", "FT", "CH", "CH"], "yaxis": "y"}, {"alignmentgroup": "True", "histfunc": "count", "hoverlabel": {"namelength": 0}, "hovertemplate": "predicted=FT<br>actual=%{x}<br>count of actual=%{y}", "legendgroup": "predicted=FT", "marker": {"color": "#EF553B"}, "name": "predicted=FT", "offsetgroup": "predicted=FT", "orientation": "v", "showlegend": true, "type": "histogram", "uid": "6ba0ed79-1cef-41da-97aa-edb35ac09c10", "x": ["FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "SI", "CH", "CH", "CH", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "FT", "CH", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FF", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FF", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "CH", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "CH", "FF", "FF", "SI", "CH", "CH", "FT", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "FC", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "CH", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "CH", "FT", "CH", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "FT", "CH", "CH", "FT", "FT", "FT", "CH", "CH", "FT", "FT", "FT", "FT", "SI", "FF", "FF", "SI", "SI", "FT", "SI", "SI", "CH", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "CH", "FT", "CH", "CH", "FT", "CH", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "SI", "SI", "FT", "CH", "FT", "FT", "SI", "FT", "FT", "FT", "FT", "SI", "FT", "CH", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "FF", "FT", "FT", "CH", "CH", "FT", "CH", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FF", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "CH", "FT", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "SI", "SI", "FF", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "FF", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "CH", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FF", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "FT", "FT", "FT", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "CH", "FT", "CH", "CH", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "FT", "CH", "FT", "FT", "FT", "FT", "CH", "FT", "CH", "CH", "FT", "FT", "CH", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "FT", "FT", "CH", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT"], "xaxis": "x", "y": ["FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "SI", "CH", "CH", "CH", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "FT", "CH", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FF", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FF", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "CH", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "CH", "FF", "FF", "SI", "CH", "CH", "FT", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "FC", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "CH", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "CH", "FT", "CH", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "FT", "CH", "CH", "FT", "FT", "FT", "CH", "CH", "FT", "FT", "FT", "FT", "SI", "FF", "FF", "SI", "SI", "FT", "SI", "SI", "CH", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "CH", "FT", "CH", "CH", "FT", "CH", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "SI", "SI", "FT", "CH", "FT", "FT", "SI", "FT", "FT", "FT", "FT", "SI", "FT", "CH", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "FF", "FT", "FT", "CH", "CH", "FT", "CH", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FF", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "CH", "FT", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "SI", "SI", "FF", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "FF", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "CH", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FF", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "FT", "FT", "FT", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "CH", "FT", "CH", "CH", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "FT", "CH", "FT", "FT", "FT", "FT", "CH", "FT", "CH", "CH", "FT", "FT", "CH", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "FT", "FT", "CH", "FT", "FT", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT"], "yaxis": "y"}, {"alignmentgroup": "True", "histfunc": "count", "hoverlabel": {"namelength": 0}, "hovertemplate": "predicted=FF<br>actual=%{x}<br>count of actual=%{y}", "legendgroup": "predicted=FF", "marker": {"color": "#00cc96"}, "name": "predicted=FF", "offsetgroup": "predicted=FF", "orientation": "v", "showlegend": true, "type": "histogram", "uid": "dd128f49-3288-4e1e-8534-2ec33563780a", "x": ["FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FT", "FT", "FT", "FF", "SL", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "SI", "SI", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "SI", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "SI", "FT", "FF", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FC", "SI", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "SI", "FF", "FT", "FT", "FF", "FT", "FF", "FT", "FT", "FT", "FT", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FC", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FT", "FT", "FC", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FC", "FT", "FF", "FT", "CH", "FC", "FT", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "SL", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "SL", "FF", "FF", "FF", "SL", "SL", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "SL", "FF", "SL", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FC", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FT", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FT", "FC", "FC", "FF", "FF", "FT", "FF", "FC", "FF", "FF", "FC", "FC", "FC", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FC", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "FF", "SI", "SI", "FT", "FT", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FT", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FC", "FF", "FF", "FC", "FF", "FF", "FF", "FT", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FC", "FF", "FC", "FF", "FF", "FT", "FF", "FT", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FF", "FT", "FF", "FF", "FF", "FC", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FC", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FT", "FC", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "FF", "FF", "FF", "FC", "FF", "FC", "FC", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "SI", "FF", "FT", "FF", "FT", "FT", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "SI", "FF", "FT", "FT", "FF", "FT", "FT", "FF", "FT", "SI", "FF", "FF", "FT", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "PO", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "SL", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FT", "FF", "FF", "FT", "FF", "FT", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FC", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FC", "FF", "SI", "SI", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "PO", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FF", "FF", "FT", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "FC", "FF", "FC", "FF", "FF", "FF", "FC", "FF", "FC", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "FF", "FT", "FF", "FT", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "SI", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "SI", "SI", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "SL", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "FT", "SL", "FT", "FF", "FF", "FF", "FT", "FF", "FT", "FT", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF"], "xaxis": "x", "y": ["FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FT", "FT", "FT", "FF", "SL", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "SI", "SI", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "SI", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "SI", "FT", "FF", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FC", "SI", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "SI", "FF", "FT", "FT", "FF", "FT", "FF", "FT", "FT", "FT", "FT", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FC", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FT", "FT", "FC", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FC", "FT", "FF", "FT", "CH", "FC", "FT", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "SL", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "SL", "FF", "FF", "FF", "SL", "SL", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "SL", "FF", "SL", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FC", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FT", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FT", "FC", "FC", "FF", "FF", "FT", "FF", "FC", "FF", "FF", "FC", "FC", "FC", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FC", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "FF", "SI", "SI", "FT", "FT", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FT", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FC", "FF", "FF", "FC", "FF", "FF", "FF", "FT", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FC", "FF", "FC", "FF", "FF", "FT", "FF", "FT", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FF", "FT", "FF", "FF", "FF", "FC", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FC", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FT", "FC", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "FF", "FF", "FF", "FC", "FF", "FC", "FC", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "SI", "FF", "FT", "FF", "FT", "FT", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "SI", "FF", "FT", "FT", "FF", "FT", "FT", "FF", "FT", "SI", "FF", "FF", "FT", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "PO", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "SL", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FT", "FF", "FF", "FT", "FF", "FT", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FC", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FC", "FF", "SI", "SI", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "PO", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FF", "FF", "FT", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FC", "FC", "FF", "FC", "FF", "FF", "FF", "FC", "FF", "FC", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "FF", "FT", "FF", "FT", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "SI", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "SI", "SI", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "SL", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FT", "FT", "SL", "FT", "FF", "FF", "FF", "FT", "FF", "FT", "FT", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FT", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF"], "yaxis": "y"}, {"alignmentgroup": "True", "histfunc": "count", "hoverlabel": {"namelength": 0}, "hovertemplate": "predicted=FC<br>actual=%{x}<br>count of actual=%{y}", "legendgroup": "predicted=FC", "marker": {"color": "#ab63fa"}, "name": "predicted=FC", "offsetgroup": "predicted=FC", "orientation": "v", "showlegend": true, "type": "histogram", "uid": "434a0429-7937-4772-8c04-f9f1ff0c47ac", "x": ["FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "SL", "CH", "CH", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "SI", "SL", "SL", "SL", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FF", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "CH", "FC", "SL", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FF", "CU", "FF", "FF", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "CU", "FC", "FC", "SL", "FC", "FC", "FC", "SL", "SL", "FC", "FF", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FF", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "SL", "SL", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC"], "xaxis": "x", "y": ["FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "SL", "CH", "CH", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "SI", "SL", "SL", "SL", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SL", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FF", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "CH", "FC", "SL", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FF", "CU", "FF", "FF", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "CU", "FC", "FC", "SL", "FC", "FC", "FC", "SL", "SL", "FC", "FF", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FF", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FF", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "SL", "SL", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FF", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC"], "yaxis": "y"}, {"alignmentgroup": "True", "histfunc": "count", "hoverlabel": {"namelength": 0}, "hovertemplate": "predicted=CH<br>actual=%{x}<br>count of actual=%{y}", "legendgroup": "predicted=CH", "marker": {"color": "#19d3f3"}, "name": "predicted=CH", "offsetgroup": "predicted=CH", "orientation": "v", "showlegend": true, "type": "histogram", "uid": "24f92103-74d3-43c3-974b-c6ef947975ca", "x": ["FF", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "SL", "CH", "CH", "SL", "CH", "CH", "CH", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CU", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "FT", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "SI", "SL", "FS", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "CH", "CH", "CH", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "CH", "SL", "CH", "CH", "CH", "CH", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "SI", "SI", "CH", "CH", "SI", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "SI", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "FF", "FF", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FF", "CH", "FF", "FF", "FF", "CH", "CH", "FF", "FF", "FF", "FF", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CU", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "CH", "SL", "CH", "CH", "SI", "CH", "CH", "SI", "SI", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "CH", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "KC", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "SI", "CH", "FO", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "SL", "CH", "CH", "FO", "FO", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "FT", "FT", "CH", "CH", "CH", "CH", "FF", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "SI", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "SL", "SL", "CH", "CH", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "CH", "SI", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FF", "CH", "CH", "CH", "CU", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "FT", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH"], "xaxis": "x", "y": ["FF", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "SL", "CH", "CH", "SL", "CH", "CH", "CH", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CU", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "FT", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "SI", "SL", "FS", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "CH", "CH", "CH", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "CH", "SL", "CH", "CH", "CH", "CH", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "SI", "SI", "CH", "CH", "SI", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "SI", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "FF", "FF", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FF", "CH", "FF", "FF", "FF", "CH", "CH", "FF", "FF", "FF", "FF", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CU", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "CH", "SL", "CH", "CH", "SI", "CH", "CH", "SI", "SI", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "CH", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "KC", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "SI", "CH", "FO", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "FS", "CH", "CH", "CH", "FS", "FS", "FS", "FS", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SL", "SL", "CH", "CH", "FO", "FO", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "FT", "FT", "CH", "CH", "CH", "CH", "FF", "FS", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "SI", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "SL", "SL", "CH", "CH", "SL", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "CH", "SI", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FF", "CH", "CH", "CH", "CU", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "FT", "FT", "FT", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "SI", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH", "CH"], "yaxis": "y"}, {"alignmentgroup": "True", "histfunc": "count", "hoverlabel": {"namelength": 0}, "hovertemplate": "predicted=SL<br>actual=%{x}<br>count of actual=%{y}", "legendgroup": "predicted=SL", "marker": {"color": "#e763fa"}, "name": "predicted=SL", "offsetgroup": "predicted=SL", "orientation": "v", "showlegend": true, "type": "histogram", "uid": "d5ec43de-0656-4045-af38-f7c3400a8cdb", "x": ["SL", "FC", "FC", "FC", "SL", "FC", "SL", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "FS", "FS", "FS", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SI", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "CH", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "FC", "CU", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SI", "SI", "FC", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "CH", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SI", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FS", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "SL", "FC", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "FC", "SL", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "FC", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "FC", "SL", "SL", "SI", "FF", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "FC", "FC", "FC", "SL", "FC", "SL", "SL", "CH", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CH", "CH", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "KC", "SL", "SL", "FC", "SL", "FC", "FC", "CH", "CH", "CH", "SL", "CH", "SL", "CH", "FC", "CH", "CH", "CH", "CH", "CH", "FC", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "CU", "SL", "SL", "SL", "CU", "SL", "SL", "CU", "CU", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "CH", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FS", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "FC", "CH", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "FC", "FC", "FC", "FC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "FC", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "KC", "CU", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FF", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CH", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "FC", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FS", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "KC", "KC", "KC", "KC", "KC", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "KC", "KC", "KC", "KC", "KC", "SL", "FC", "SL", "SL", "SL", "FC", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "SL", "SL", "KC", "SL", "SI", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CH", "SL", "SL", "KC", "SL", "SL", "SL", "KC", "CU", "SL", "FC", "SL", "FC", "FC", "FC", "SL", "SI", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "FC", "FC", "SL", "FC", "SL", "SL", "SL", "SL", "FF", "SL", "CU", "CU", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CH", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "FC", "FC", "FC", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "KC", "KC", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "FC", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "SL", "CH", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CH", "FC", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "CU", "SL", "CU", "FC", "CU", "FC", "CU", "FC", "FC", "CU", "FC", "FC", "FC", "CU", "CU", "SL", "KC", "KC", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "KC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL"], "xaxis": "x", "y": ["SL", "FC", "FC", "FC", "SL", "FC", "SL", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "FS", "FS", "FS", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SI", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "CH", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "FC", "CU", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SI", "SI", "FC", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "CH", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SI", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FS", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "SL", "FC", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "FC", "SL", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "FC", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "FC", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "FC", "SL", "SL", "SI", "FF", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "FC", "FC", "FC", "SL", "FC", "SL", "SL", "CH", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CH", "CH", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "KC", "SL", "SL", "FC", "SL", "FC", "FC", "CH", "CH", "CH", "SL", "CH", "SL", "CH", "FC", "CH", "CH", "CH", "CH", "CH", "FC", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "CU", "SL", "SL", "SL", "CU", "SL", "SL", "CU", "CU", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "CH", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FS", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "FC", "CH", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "FC", "FC", "FC", "FC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "FC", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "KC", "CU", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FF", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CH", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "FC", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FS", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "KC", "KC", "KC", "KC", "KC", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "KC", "KC", "KC", "KC", "KC", "SL", "FC", "SL", "SL", "SL", "FC", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "SL", "FC", "FC", "FC", "SL", "SL", "KC", "SL", "SI", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CH", "SL", "SL", "KC", "SL", "SL", "SL", "KC", "CU", "SL", "FC", "SL", "FC", "FC", "FC", "SL", "SI", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "FC", "FC", "SL", "FC", "SL", "SL", "SL", "SL", "FF", "SL", "CU", "CU", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CH", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "FC", "FC", "FC", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "KC", "KC", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "SL", "SL", "SL", "SL", "FC", "FC", "SL", "SL", "FC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "SL", "FC", "SL", "SL", "SL", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "FC", "FC", "FC", "FC", "SL", "SL", "FC", "SL", "CH", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CH", "FC", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "SL", "CU", "SL", "CU", "FC", "CU", "FC", "CU", "FC", "FC", "CU", "FC", "FC", "FC", "CU", "CU", "SL", "KC", "KC", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "KC", "FC", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "KC", "SL", "SL", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL", "SL"], "yaxis": "y"}, {"alignmentgroup": "True", "histfunc": "count", "hoverlabel": {"namelength": 0}, "hovertemplate": "predicted=CU<br>actual=%{x}<br>count of actual=%{y}", "legendgroup": "predicted=CU", "marker": {"color": "#FECB52"}, "name": "predicted=CU", "offsetgroup": "predicted=CU", "orientation": "v", "showlegend": true, "type": "histogram", "uid": "53f879c5-3ac7-4d56-8ba5-275668b72238", "x": ["CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CH", "CH", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CH", "CH", "CH", "CH", "CH", "KC", "CU", "CH", "CH", "CH", "CH", "CH", "CU", "CH", "CH", "CH", "CU", "CH", "KC", "CH", "CU", "CH", "CH", "CH", "CH", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CH", "CH", "CU", "CU", "SL", "CU", "SL", "CU", "CU", "CU", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "SL", "KC", "SL", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "KC", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "CU", "CU", "SL", "SL", "CU", "CU", "SL", "CU", "CU", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "SL", "SL", "KC", "KC", "SL", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "CU", "KC", "KC", "CU", "CU", "CU", "KC", "KC", "KC", "CU", "CU", "CU", "KC", "KC", "CU", "KC", "KC", "CU", "KC", "KC", "CU", "CU", "CU", "CU", "CH", "CH", "CH", "CH", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "CU", "CU", "CU", "SL", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "SL", "FC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CH", "CH", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "SL", "SL", "CU", "CU", "CU", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "SL", "SL", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "KC", "KC", "KC", "KC", "KC", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "EP", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "KC", "EP", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CH", "CU", "CH", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SI", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "SL", "CU", "KC", "KC", "KC", "KC", "CU", "CU", "KC", "KC", "KC", "KC", "CU", "KC", "KC", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "SL", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "SL", "SL", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "CU", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "CU", "CU", "CU", "CU", "SL", "CU", "SL", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "SL", "SL", "CU", "SL", "CU", "SL", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "CU", "CU", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "CU", "CU", "SL", "KC", "KC", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "SL", "KC", "CU", "CU", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "KC", "KC", "CU", "CU", "KC", "KC", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "SL", "SL", "SL", "SL"], "xaxis": "x", "y": ["CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CH", "CH", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CH", "CH", "CH", "CH", "CH", "KC", "CU", "CH", "CH", "CH", "CH", "CH", "CU", "CH", "CH", "CH", "CU", "CH", "KC", "CH", "CU", "CH", "CH", "CH", "CH", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CH", "CH", "CU", "CU", "SL", "CU", "SL", "CU", "CU", "CU", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "SL", "KC", "SL", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "KC", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "SL", "SL", "SL", "SL", "CU", "CU", "CU", "SL", "SL", "CU", "CU", "SL", "CU", "CU", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "SL", "SL", "KC", "KC", "SL", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "CU", "KC", "KC", "CU", "CU", "CU", "KC", "KC", "KC", "CU", "CU", "CU", "KC", "KC", "CU", "KC", "KC", "CU", "KC", "KC", "CU", "CU", "CU", "CU", "CH", "CH", "CH", "CH", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "CU", "CU", "CU", "SL", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "SL", "FC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CH", "CH", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "SL", "SL", "CU", "CU", "CU", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "SL", "SL", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "KC", "KC", "KC", "KC", "KC", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "EP", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "KC", "EP", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CH", "CU", "CH", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SI", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "SL", "CU", "KC", "KC", "KC", "KC", "CU", "CU", "KC", "KC", "KC", "KC", "CU", "KC", "KC", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "SL", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "SL", "SL", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "CU", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "CU", "CU", "CU", "CU", "SL", "CU", "SL", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "SL", "SL", "CU", "SL", "CU", "SL", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "SL", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "CU", "CU", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "KC", "CU", "CU", "CU", "SL", "KC", "KC", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "SL", "KC", "CU", "CU", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "CU", "KC", "KC", "CU", "CU", "KC", "KC", "SL", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SL", "CU", "SL", "SL", "SL", "SL"], "yaxis": "y"}, {"alignmentgroup": "True", "histfunc": "count", "hoverlabel": {"namelength": 0}, "hovertemplate": "predicted=SI<br>actual=%{x}<br>count of actual=%{y}", "legendgroup": "predicted=SI", "marker": {"color": "#FFA15A"}, "name": "predicted=SI", "offsetgroup": "predicted=SI", "orientation": "v", "showlegend": true, "type": "histogram", "uid": "bf4dfa25-ef89-4bc3-b186-38d72f59b505", "x": ["FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FF", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "FF", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "SI", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FS", "FS", "FS", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "FF", "SI", "FF", "FF", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "CH", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SL", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SL", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "SI", "CH", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FS", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "FT", "FT", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "FC", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "CH", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "CH", "CH", "CH", "SI", "FT", "FT", "FT", "FT", "FT", "FS", "FT", "FT", "FT", "FT", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FF", "FT", "FT", "SI", "FF", "SI", "FF", "SI", "FF", "FF", "SI", "FF", "SI", "FF", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "CH", "SI", "CH", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "CH", "CH", "CH", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "SI", "SI", "FF", "CH", "CH", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "SI", "FF", "FF", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "FF", "SI", "FF", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "CH", "CH", "FF", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FC", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "CH", "SI", "SI", "CH", "CH", "SI", "SI", "SI", "SI", "SI", "CH", "CH", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "CH", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI"], "xaxis": "x", "y": ["FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FF", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "FF", "CH", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "SI", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FS", "FS", "FS", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "FF", "SI", "FF", "FF", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "CH", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SL", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SL", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "SI", "CH", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FS", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FF", "FT", "FT", "FT", "FT", "FF", "FF", "FF", "FF", "FF", "FT", "FT", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "SI", "SI", "FF", "FF", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "FT", "FT", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FF", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "FC", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "CH", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "FF", "FF", "CH", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "FF", "FF", "FF", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "CH", "CH", "CH", "SI", "FT", "FT", "FT", "FT", "FT", "FS", "FT", "FT", "FT", "FT", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FF", "FT", "FT", "SI", "FF", "SI", "FF", "SI", "FF", "FF", "SI", "FF", "SI", "FF", "FT", "FT", "FT", "FT", "FT", "CH", "CH", "CH", "CH", "SI", "CH", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "CH", "CH", "CH", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "CH", "CH", "CH", "CH", "CH", "SI", "SI", "SI", "SI", "FF", "CH", "CH", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "SI", "FF", "FF", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "FF", "SI", "FF", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "CH", "CH", "FF", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FC", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "CH", "SI", "SI", "CH", "CH", "SI", "SI", "SI", "SI", "SI", "CH", "CH", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "FF", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FT", "FT", "FT", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "CH", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "CH", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "SI", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "SI", "SI", "SI", "SI"], "yaxis": "y"}, {"alignmentgroup": "True", "histfunc": "count", "hoverlabel": {"namelength": 0}, "hovertemplate": "predicted=KC<br>actual=%{x}<br>count of actual=%{y}", "legendgroup": "predicted=KC", "marker": {"color": "#FF6692"}, "name": "predicted=KC", "offsetgroup": "predicted=KC", "orientation": "v", "showlegend": true, "type": "histogram", "uid": "8730854f-bcb8-49e1-a3e9-4b3adba12fa7", "x": ["SL", "CU", "CU", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "SL", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "CU", "KC", "KC", "SL", "SL", "SL", "KC", "KC", "SL", "SL", "SL", "SL", "KC", "KC", "CU", "KC", "KC", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SI", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SI", "SI", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "KC", "KC", "KC", "KC", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "SL", "KC", "KC", "KC", "CU", "SL", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "SL", "CU", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "KC", "CU", "KC", "KC", "SL", "KC", "SL", "KC", "KC", "CU", "SL", "SL", "KC", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "SI", "SL", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CH", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC"], "xaxis": "x", "y": ["SL", "CU", "CU", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "SL", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "CU", "KC", "KC", "SL", "SL", "SL", "KC", "KC", "SL", "SL", "SL", "SL", "KC", "KC", "CU", "KC", "KC", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "SL", "SL", "SL", "SL", "SL", "SL", "SI", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "SI", "SI", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "KC", "KC", "KC", "KC", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "SL", "KC", "KC", "KC", "CU", "SL", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "SL", "CU", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "CU", "KC", "CU", "KC", "KC", "SL", "KC", "SL", "KC", "KC", "CU", "SL", "SL", "KC", "KC", "KC", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "SI", "SL", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "SL", "SL", "SL", "KC", "SL", "SL", "SL", "CU", "CU", "CU", "CU", "CU", "CH", "CU", "CU", "CU", "CU", "CU", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC", "KC"], "yaxis": "y"}],
                        {"barmode": "relative", "grid": {"xaxes": ["x"], "xgap": 0.1, "xside": "bottom", "yaxes": ["y"], "ygap": 0.1, "yside": "left"}, "height": 600, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0508b8"], [0.0893854748603352, "#1910d8"], [0.1787709497206704, "#3c19f0"], [0.2681564245810056, "#6b1cfb"], [0.3575418994413408, "#981cfd"], [0.44692737430167595, "#bf1cfd"], [0.5363128491620112, "#dd2bfd"], [0.6256983240223464, "#f246fe"], [0.7150837988826816, "#fc67fd"], [0.8044692737430168, "#fe88fc"], [0.8938547486033519, "#fea5fd"], [0.9832402234636871, "#febefe"], [1.0, "#fec3fe"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0508b8"], [0.0893854748603352, "#1910d8"], [0.1787709497206704, "#3c19f0"], [0.2681564245810056, "#6b1cfb"], [0.3575418994413408, "#981cfd"], [0.44692737430167595, "#bf1cfd"], [0.5363128491620112, "#dd2bfd"], [0.6256983240223464, "#f246fe"], [0.7150837988826816, "#fc67fd"], [0.8044692737430168, "#fe88fc"], [0.8938547486033519, "#fea5fd"], [0.9832402234636871, "#febefe"], [1.0, "#fec3fe"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0508b8"], [0.0893854748603352, "#1910d8"], [0.1787709497206704, "#3c19f0"], [0.2681564245810056, "#6b1cfb"], [0.3575418994413408, "#981cfd"], [0.44692737430167595, "#bf1cfd"], [0.5363128491620112, "#dd2bfd"], [0.6256983240223464, "#f246fe"], [0.7150837988826816, "#fc67fd"], [0.8044692737430168, "#fe88fc"], [0.8938547486033519, "#fea5fd"], [0.9832402234636871, "#febefe"], [1.0, "#fec3fe"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0508b8"], [0.0893854748603352, "#1910d8"], [0.1787709497206704, "#3c19f0"], [0.2681564245810056, "#6b1cfb"], [0.3575418994413408, "#981cfd"], [0.44692737430167595, "#bf1cfd"], [0.5363128491620112, "#dd2bfd"], [0.6256983240223464, "#f246fe"], [0.7150837988826816, "#fc67fd"], [0.8044692737430168, "#fe88fc"], [0.8938547486033519, "#fea5fd"], [0.9832402234636871, "#febefe"], [1.0, "#fec3fe"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#506784", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0508b8"], [0.0893854748603352, "#1910d8"], [0.1787709497206704, "#3c19f0"], [0.2681564245810056, "#6b1cfb"], [0.3575418994413408, "#981cfd"], [0.44692737430167595, "#bf1cfd"], [0.5363128491620112, "#dd2bfd"], [0.6256983240223464, "#f246fe"], [0.7150837988826816, "#fc67fd"], [0.8044692737430168, "#fe88fc"], [0.8938547486033519, "#fea5fd"], [0.9832402234636871, "#febefe"], [1.0, "#fec3fe"]], "sequentialminus": [[0.0, "#0508b8"], [0.0893854748603352, "#1910d8"], [0.1787709497206704, "#3c19f0"], [0.2681564245810056, "#6b1cfb"], [0.3575418994413408, "#981cfd"], [0.44692737430167595, "#bf1cfd"], [0.5363128491620112, "#dd2bfd"], [0.6256983240223464, "#f246fe"], [0.7150837988826816, "#fc67fd"], [0.8044692737430168, "#fe88fc"], [0.8938547486033519, "#fea5fd"], [0.9832402234636871, "#febefe"], [1.0, "#fec3fe"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#19d3f3", "#e763fa", "#FECB52", "#FFA15A", "#FF6692", "#B6E880"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"fillcolor": "#506784", "line": {"width": 0}, "opacity": 0.4}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Actual vs Predicted Pitches (Validation Set: August & September 2018)"}, "xaxis": {"title": {"text": "actual"}}, "yaxis": {"title": {"text": "count of actual"}}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('1b73b614-ed6d-4989-9cc3-2ddff29f31e8');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


---

# 2019.05.30 - Re: Data Storage below:  We didn't actually end up storing our data on AWS, since it was relatively small enough to re-pull occasionally from API.

## Data Storage

Now that we have our data, let's store it in a PostgreSQL db on AWS so we don't have to keep rebuilding it.

### 1. Use SQLAlchemy to create PSQL engine:


```python
# dialect+driver://username:password@host:port/database

sql_alc_engine = create_engine('postgresql://dsaf:dsaf040119@flatiron-projects.\
cy8jwdkpepr0.us-west-2.rds.amazonaws.com/flatiron')
```

### 2. Use `pandas.to_sql` to write the `pitches_df` dataframe to the PostgreSQL database, using the SQLAlchemy engine.
    


```python
pitches_df.to_sql('pitches', sql_alc_engine)
```

### 3. Check that the table was created.


```python
# Setup PSQL connection
conn = psql.connect(
    database="flatiron",
    user="dsaf",
    password="dsaf040119",
    host="flatiron-projects.cy8jwdkpepr0.us-west-2.rds.amazonaws.com",
    port='5432'
)
```


```python
# Set up query
query = """
    SELECT * FROM pitches;
"""
```


```python
# Instantiate cursor
cur = conn.cursor()
```


```python
# Execute the query
cur.execute(query)
```


```python
# Check results
pitches_df_clone = pd.DataFrame(cur.fetchall())
pitches_df_clone.columns = [col.name for col in cur.description]
```


```python
pitches_df_clone.head()
```


```python
pitches_df.tail(7)
```


```python
pitches_df_clone.drop(['index'], axis=1, inplace=True)
```


```python
pitches_df_clone.tail(7)
```


```python
pitches_df.equals(pitches_df)
```


```python
pitches_df.info()
```


```python
pitches_df_clone.info()
```


```python
pitches_df_clone.shape
```

Ah, it seems that `NaN` got transformed to None in the migration to PSQL and come back as such.


```python
pitches_df.loc[pitches_df['details.call.code'].isna() ].shape
```


```python
# Let's try to find the Nones
pitches_df_clone.loc[pitches_df_clone['details.call.code'].isna() ].shape
```


```python
pitches_df['details.call.code'] == pitches_df_clone['details.call.code']
```


```python
pitches_df_clone['details.call.code'].value_counts()
```


```python
142+123+53
```


```python
import numpy as np
```


```python
pitches_df_clone.replace([None], np.nan, inplace=True)
```


```python
pitches_df_clone.tail(7)
```


```python
pitches_df_clone.info()
```


```python
pitches_df == pitches_df_clone
```

---

# Notes / To Dos / Future Work

~~### 1. Need to incorporate work to create list of desired games. Likely will require looping through list.~~

~~### 2. Should this data be written out to a database, e.g. SQL or NoSQL?~~

### 3. Other data to join? Team Characteristics? Player characteristics? RISP??
