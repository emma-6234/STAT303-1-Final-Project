import pandas as pd
import numpy as np
import re
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns

airbnb_data = pd.read_csv('AB_NYC_2019_cleaned.csv')


### By Room Type
roomtype_dict = {}

for i, row in enumerate(airbnb_data.iterrows()):

    room_type = airbnb_data.at[i, 'room_type']
    id = airbnb_data.at[i, 'id']

    if room_type not in roomtype_dict:
        roomtype_dict[room_type] = [id]
    else:
        roomtype_dict[room_type].append(id)


### price by room type
for type in roomtype_dict.keys():
    mean_price = airbnb_data[airbnb_data['id'].isin(roomtype_dict[type])]['price'].mean()
    print(type, mean_price)
