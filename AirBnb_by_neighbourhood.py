import pandas as pd
import numpy as np
import re
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns

airbnb_data = pd.read_csv('AB_NYC_2019_cleaned.csv')


### By Neighborhood
# creates a dictionary where the neighbourhood name calls a list of ids of listings in that neighbourhood
neighborhood_dict = {}

for i, row in enumerate(airbnb_data.iterrows()):

    hood_name = airbnb_data.at[i, 'neighbourhood_group']
    id = airbnb_data.at[i, 'id']

    if hood_name not in neighborhood_dict:
        neighborhood_dict[hood_name] = [id]
    else:
        neighborhood_dict[hood_name].append(id)


### price by neighbohood
for hood in neighborhood_dict.keys():
    mean_price = airbnb_data[airbnb_data['id'].isin(neighborhood_dict[hood])]['price'].mean()
    print(hood, mean_price)
