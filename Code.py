import pandas as pd
import numpy as np
import re
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns

airbnb_data = pd.read_csv('AB_NYC_2019_cleaned.csv')

import AirBnB_by_roomtype as roomtype_dict
import AirBnb_by_neighbourhood as neighborhood_dict
import AirBnB_name_word_occurances as name_words

### Average price for full house/apt in Manhattan
Manhattan_houses = airbnb_data[
    (airbnb_data['neighbourhood_group'] == 'Manhattan') &
    airbnb_data['id'].isin(neighborhood_dict.neighborhood_dict['Manhattan']) &
    airbnb_data['id'].isin(roomtype_dict.roomtype_dict['Entire home/apt'])
]

Mean_manhattan_house = Manhattan_houses['price'].mean()

print("Mean Manhattan Houses: ", Mean_manhattan_house)
