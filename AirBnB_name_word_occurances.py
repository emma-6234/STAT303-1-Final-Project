import pandas as pd
import numpy as np
import re
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns

airbnb_data = pd.read_csv('AB_NYC_2019_cleaned.csv')



### Creates an array of words that appear in AirBnB listing names and counts their occurances
name_words = []
word_id_dict = {}
word_count = np.empty((0, 2), dtype=object)

# 'Good' characters
letters = 'abcdefghijklmnopqrstuvwxyz'
# Words that are common and not related to listing that we want to remove
excluded_words = ['the', 'and', 'for', 'that', 'have', 'not', 'with', 'you', 'this', 'but', 'from', 
                  'there', 'their', 'which', 'make', 'like', 'some', 'also', 'its']


for name in airbnb_data['name']:

    words = re.split(r'[/\s]+', str(name))

    for word in words:

        word = word.lower()
        word = ''.join([letter for letter in word if letter in letters])

        if len(word) > 2 and word not in excluded_words:   

            if word not in name_words:
                name_words.append(word)
                word_count = np.vstack((word_count, [word, 1]))
            else:
                index = name_words.index(word)
                word_count[index, 1] = int(word_count[index, 1]) + 1


# Sort the array by word count in descending order and set index and column name
word_count = pd.DataFrame(word_count[word_count[:, 1].astype(int).argsort()[::-1]])
word_count.rename({0: 'word', 1: 'count'}, axis=1, inplace=True)

# print(word_count.head(10))


i = 0
for listing in airbnb_data.iterrows():

    words = re.split(r'[/\s]+', str(listing[1]['name']))

    for word in words:

        word = word.lower()
        word = ''.join([letter for letter in word if letter in letters])

        if len(word) > 2 and word not in excluded_words:   

            if word not in word_id_dict.keys():
                word_id_dict[word] = [listing[1]['id']]
            
            else:
                word_id_dict[word].append(listing[1]['id'])

word_id_dict = dict(sorted(word_id_dict.items(), key=lambda item: len(item[1]), reverse=True))


# compare to price
# find average price for each word
price_word_array = np.empty((0, 2), dtype=object)

i=0
for word in word_id_dict.keys():
    mean_price = airbnb_data[airbnb_data['id'].isin(word_id_dict[word])]['price'].mean()


    price_word_array = np.vstack((price_word_array, [word, mean_price]))

price_word_df = pd.DataFrame(price_word_array)

price_word_df.rename({0: 'word', 1: 'mean_price'}, axis=1, inplace=True)
price_word_df = price_word_df.sort_values(by=['mean_price'], ascending=False)


# compare words to reviews per month (as correlation to popularity)



