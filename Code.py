import pandas as pd
import numpy as np
import re

airbnb_data = pd.read_csv('AB_NYC_2019.csv')



# ### Creates an array of words that appear in AirBnB listing names and counts their occurances
# name_words = []
# word_count = np.empty((0, 2), dtype=object)

# # 'Good' characters
# letters = 'abcdefghijklmnopqrstuvwxyz'
# # Words that are common and not related to listing that we want to remove
# excluded_words = ['the', 'and', 'for', 'that', 'have', 'not', 'with', 'you', 'this', 'but', 'from', 
#                   'there', 'their', 'which', 'make', 'like', 'some', 'also', 'its']

# for name in airbnb_data['name']:

#     words = re.split(r'[/\s]+', str(name))

#     for word in words:

#         word = word.lower()
#         word = ''.join([letter for letter in word if letter in letters])

#         if len(word) > 2 and word not in excluded_words:   
#             if word not in name_words:
#                 name_words.append(word)
#                 word_count = np.vstack((word_count, [word, 1]))
#             else:
#                 index = name_words.index(word)
#                 word_count[index, 1] = int(word_count[index, 1]) + 1

# # Sort the array by word count in descending order
# word_count = word_count[word_count[:, 1].astype(int).argsort()[::-1]]



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


### By Room Type
roomtype_dict = {}

for i, row in enumerate(airbnb_data.iterrows()):

    room_type = airbnb_data.at[i, 'room_type']
    id = airbnb_data.at[i, 'id']

    if room_type not in roomtype_dict:
        roomtype_dict[room_type] = [id]
    else:
        roomtype_dict[room_type].append(id)



# ### price by neighbohood
# for hood in neighborhood_dict.keys():
#     mean_price = airbnb_data[airbnb_data['id'].isin(neighborhood_dict[hood])]['price'].mean()
#     print(hood, mean_price)


# ### price by room type
# for type in roomtype_dict.keys():
#     mean_price = airbnb_data[airbnb_data['id'].isin(roomtype_dict[type])]['price'].mean()
#     print(type, mean_price)


# ### Average price for full house/apt in Manhattan
# Manhattan_houses = airbnb_data[
#     (airbnb_data['neighbourhood_group'] == 'Manhattan') &
#     airbnb_data['id'].isin(neighborhood_dict['Manhattan']) &
#     airbnb_data['id'].isin(roomtype_dict[room_type])
# ]

# Mean_manhattan_house = Manhattan_houses['price'].mean()

# print("Mean Manhattan Houses: ", Mean_manhattan_house)