import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

current_directory = os.getcwd()
target_directory = r'C:\Users\emmal\OneDrive\Desktop\Fall 2023\STAT 303-1\Final Project'

if current_directory != target_directory:
    os.chdir(target_directory)

    
airbnb_data = pd.read_csv('AB_NYC_2019_cleaned.csv')
print(airbnb_data.columns)


sns.set_theme(style="darkgrid")


### ----------------------------------- Availability to Reviews/ ----------------------------------- ###
# Select the two columns for which you want to calculate the correlation
selected_columns = ['availability_365', 'reviews_per_month']
correlation_matrix = airbnb_data[selected_columns].corr()

# Get the correlation value between the two columns


correlation_value = correlation_matrix.loc['availability_365', 'reviews_per_month']

print(f"Correlation between 'availability_365' and 'reviews_per_month': {correlation_value}")



### ----------------------------------- Listing Name Data ----------------------------------- ###
# Constants
MIN_WORD_LENGTH = 3
EXCLUDED_WORDS = {'the', 'and', 'for', 'that', 'have', 'not', 'with', 'you', 'this', 'but', 'from', 'there', 'their', 'which', 'make', 'like', 'some', 'also', 'its', 'near'}

# Helper function to clean and filter words
def clean_word(word):
    word = word.lower()
    word = ''.join([letter for letter in word if letter.isalpha()])
    return word

word_id_dict = {}
word_price_data = {}

# Extract and clean words in names
words = airbnb_data['name'].str.lower().str.findall(r'\b\w{3,}\b')
airbnb_data['cleaned_words'] = words.apply(lambda word_list: [clean_word(word) for word in word_list])
airbnb_data['cleaned_words'] = airbnb_data['cleaned_words'].apply(lambda word_list: [word for word in word_list if word not in EXCLUDED_WORDS and len(word) > MIN_WORD_LENGTH])



for index, row in airbnb_data.iterrows():
    price = row['price']
    
    for word in row['cleaned_words']:
        word_id_dict.setdefault(word, []).append(row['id'])
        word_price_data.setdefault(word, []).append(price)


# Filter by count and price
word_stats = [(word, np.mean(word_price_data[word]), len(word_id_dict[word])) for word in word_id_dict if (len(word_id_dict[word]) > 10 and np.mean(word_price_data[word]) < 1000)]
word_stats_df = pd.DataFrame(word_stats, columns=['word', 'mean_price', 'count'])
# Sort words by mean_price in descending order
word_stats_df = word_stats_df.sort_values(by='mean_price', ascending=False)


# Bin by price
num_bins = 20
cut_intv = np.linspace(0, 1000, num_bins+1)
print(cut_intv)

word_stats_df['price_bin'] = pd.cut(word_stats_df['mean_price'], bins=num_bins, labels=cut_intv[1:].astype(int))

airbnb_data['word_count'] = airbnb_data['cleaned_words'].apply(lambda row: len(row))

## num words in a listing name vs. price
num_word_price_corr = airbnb_data['cleaned_words'].apply(len).corr(airbnb_data['price'])

print(f"Correlation between number of words in the listing name and price: {round(num_word_price_corr, 5)}")
plt.figure(figsize=(8, 6))
sns.boxplot(x=airbnb_data['word_count'], y=airbnb_data['price'], showfliers=True)
plt.xlabel('Number of Words in Listing Name', labelpad=10)
plt.ylabel('Price (USD/day)', labelpad=10)
plt.suptitle('Number of Words in Listing Name vs. Price', fontsize=16, weight='bold')
plt.title('(with outliers)', pad=10)
plt.savefig('visualizations/num_words_price_with_outliers.png')
plt.show()


plt.figure(figsize=(9, 6.75))
sns.boxplot(x=airbnb_data['word_count'], y=airbnb_data['price'], showfliers=True)
plt.xlabel('Number of Words in Listing Name', labelpad=10)
plt.ylabel('Price (USD/day)', labelpad=10)
plt.suptitle('Number of Words in Listing Name vs. Price', fontsize=16, weight='bold')
plt.title('(cropped)', pad=10)
plt.xlim(-0.5, 10.5)
plt.ylim(0, 1000)
plt.tight_layout()
plt.savefig('visualizations/num_words_price_cropped.png')
plt.show()




### ---Word Length--- ###
# Calculate the average word length for each listing
airbnb_data['average_word_length'] = airbnb_data['cleaned_words'].apply(lambda word_list: np.mean([len(word) for word in word_list]))

# Scatter plot of word length vs. price
plt.figure(figsize=(8, 5))
sns.scatterplot(x=airbnb_data['average_word_length'], y=airbnb_data['price'], alpha = 0.2, linewidth=0)
plt.xlabel('Average Word Length in Listing Name')
plt.ylabel('Price (USD/day)')
plt.xlim(MIN_WORD_LENGTH, 20)
plt.title('Average Word Length vs. Price', fontsize=16, weight='bold', pad=15)
plt.tight_layout()
plt.savefig('visualizations/avg_word_length_price.png')
plt.show()



# # Plot the number of words by their average price
# plt.figure(figsize=(8, 6))
# sns.barplot(x='price_bin', y='count', data=word_stats_df)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.xlabel('Price Range', labelpad=10)
# plt.ylabel('Word Count', labelpad=12)
# plt.title('Count for Words avg Price', fontsize=16, weight='bold', pad=15)
# plt.tight_layout()
# plt.show()
# plt.savefig('visualizations/word_count_price_range.png')



# plt.figure(figsize=(8, 6))
# sns.boxplot(x=airbnb_data['word_count'], y=airbnb_data['price'], showfliers=False)
# plt.xlabel('Number of Words in Listing Name', labelpad=10)
# plt.ylabel('Price (USD/day)', labelpad=10)
# plt.suptitle('Number of Words in Listing Name vs. Price', fontsize=16, weight='bold')
# plt.title('(no outliers)', pad=10)
# plt.savefig('visualizations/num_words_price_no_outliers.png')
# plt.show()


# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=airbnb_data['average_word_length'], y=airbnb_data['price'], alpha = 0.2, linewidth=0)
# plt.xlabel('Average Word Length in Listing Name')
# plt.ylabel('Price (USD/day)')
# plt.xlim(MIN_WORD_LENGTH, 20)
# plt.ylim(0, 1000)
# plt.suptitle('Average Word Length vs. Price', fontsize=16, weight='bold')
# plt.title('(cropped)')
# plt.savefig('visualizations/avg_word_length_price_cropped.png')
# plt.show()




# from collections import Counter

# # Combine all listing names into a single list of words
# all_words = [word for word_list in airbnb_data['cleaned_words'] for word in word_list]
# neighborhoods = airbnb_data['neighbourhood_group'].unique()

# # Count word frequency
# word_frequency = Counter(all_words)
# total_words = len(all_words)

# # Get the most common words and their frequencies
# most_common_words = word_frequency.most_common(20)  

# # Plot word frequency
# plt.figure(figsize=(8, 6))
# sns.barplot(x=[word[0] for word in most_common_words], y=[word_frequency[word[0]] for word in most_common_words])
# plt.xticks(rotation=55, fontsize=10, ha='right')
# plt.xlabel('Word', labelpad=15)
# plt.ylabel('Frequency', labelpad=15)
# plt.title('Most Common Words', fontsize=16, weight='bold', pad=15)
# plt.tight_layout()
# plt.savefig('visualizations/most_common_words.png')
# plt.show()


# # Step 2: Calculate the top 50% most common words in the dataframe
# sorted_dataframe_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
# cumulative_frequency = 0
# top_words_dataframe = []

# for word, frequency in sorted_dataframe_words:
#     cumulative_frequency += frequency
#     top_words_dataframe.append(word)

#     if cumulative_frequency >= 0.5 * total_words:
#         break

# exclude_words = ['room', 'bedroom', 'private', 'apartment']


# # Function to calculate and normalize word frequencies for a neighborhood
# def calculate_neighborhood_word_frequency(neighborhood_data):
#     neighborhood_words = [word for word_list in neighborhood_data['cleaned_words'] for word in word_list]
#     # Exclude specified words
#     neighborhood_words = [word for word in neighborhood_words if word not in exclude_words]
#     word_freq_neighborhood = Counter(neighborhood_words)
#     total_word_count_neighborhood = len(neighborhood_words)
#     normalized_word_freq = {word: word_freq_neighborhood[word] / total_word_count_neighborhood for word in word_freq_neighborhood}
#     return normalized_word_freq

# neighborhood_word_frequencies = {}

# for neighborhood in airbnb_data['neighbourhood_group'].unique():
#     neighborhood_data = airbnb_data[airbnb_data['neighbourhood_group'] == neighborhood]
#     neighborhood_word_frequencies[neighborhood] = calculate_neighborhood_word_frequency(neighborhood_data)

# difference_neighborhoods = {}

# # Assuming you have total_words, word_frequency, and neighborhood_word_frequencies defined
# for neighborhood, neighborhood_word_freq in neighborhood_word_frequencies.items():
#     difference = {word: neighborhood_word_freq[word] - word_frequency.get(word, 0) / total_words for word in neighborhood_word_freq}
#     difference_neighborhoods[neighborhood] = difference

# # Number of top words to display
# num_top_words = 10

# # Function to create a bar plot for a neighborhood
# def create_neighborhood_bar_plot(hood, ax):
#     neighborhood_data = airbnb_data[airbnb_data['neighbourhood_group'] == hood]
#     neighborhood_word_frequencies = calculate_neighborhood_word_frequency(neighborhood_data)
#     top_hood_words = sorted(neighborhood_word_frequencies.items(), key=lambda x: x[1], reverse=True)[:num_top_words]

#     # Create a bar chart for the words
#     sns.barplot(x=[word[0] for word in top_hood_words], y=[word[1] for word in top_hood_words], ax=ax)
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
#     ax.set_ylabel('Normalized Frequency', fontsize=8)
#     ax.set_ylim(0, 0.065)
#     ax.tick_params(axis='both', which='major', labelsize=8)
#     ax.set_title(f'{hood} Frequency', weight='bold', fontsize=12)

# # Number of columns in the plot grid
# cols = 3

# # Number of rows in the plot grid
# rows = (len(neighborhoods) + cols - 1) // cols

# # Create subplots
# fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

# # Iterate through neighborhoods and create subplots
# for i, hood in enumerate(neighborhoods):
#     ax = axes[i // cols, i % cols]
#     create_neighborhood_bar_plot(hood, ax)

# # Adjust layout
# plt.tight_layout()
# plt.savefig('visualizations/neighborhood_word_frequency.png')
# plt.show()

# # Function to create a bar plot of frequency difference for each neighborhood
# def create_difference_bar_plot(hood, ax):
#     top_words_diff = sorted(difference_neighborhoods[hood].items(), key=lambda x: x[1], reverse=True)[:num_top_words]
    
#     # Create a bar chart for the words
#     sns.barplot(x=[word[0] for word in top_words_diff], y=[word[1] for word in top_words_diff], ax=ax)
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)    
#     ax.set_ylabel('Frequency Difference', fontsize=8)
#     ax.set_ylim(0, 0.065)
#     ax.tick_params(axis='both', which='major', labelsize=8)
#     ax.set_title(f'{hood} Frequency Difference', weight='bold', fontsize=12)

# # Create subplots for frequency difference
# fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

# # Iterate through neighborhoods and create subplots for frequency difference
# for i, neighborhood in enumerate(neighborhoods):
#     ax = axes[i // cols, i % cols]
#     create_difference_bar_plot(neighborhood, ax)

# # Adjust layout
# plt.tight_layout()
# plt.savefig('visualizations/neighborhood_word_frequency_difference.png')
# plt.show()





# # Function to create a bar plot with global frequency markers
# def create_global_frequency_bar_plot(hood, ax):
#     neighborhood_data = airbnb_data[airbnb_data['neighbourhood_group'] == hood]
#     neighborhood_word_frequencies = calculate_neighborhood_word_frequency(neighborhood_data)
#     top_hood_words = sorted(neighborhood_word_frequencies.items(), key=lambda x: x[1], reverse=True)[:num_top_words]

#     # Create a bar chart for the words
#     bars = sns.barplot(x=[word[0] for word in top_hood_words], y=[word[1] for word in top_hood_words], ax=ax)
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
#     ax.set_ylabel('Normalized Frequency', fontsize=8)
#     ax.set_ylim(0, 0.065)
#     ax.tick_params(axis='both', which='major', labelsize=8)
#     ax.set_title(f'{hood} Frequency', weight='bold', fontsize=12)

#     global_frequencies = [word_frequency[top_hood_words[j][0]] / total_words for j in range(len(top_hood_words))]
#     ax.plot(bars.get_xticks(), global_frequencies, '_', markersize=12, color = 'red')


# # Number of columns in the plot grid
# cols = 3

# # Number of rows in the plot grid
# rows = (len(neighborhoods) + cols - 1) // cols

# # Create subplots for neighborhood frequency with global frequency markers
# fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

# # Iterate through neighborhoods and create subplots
# for i, hood in enumerate(neighborhoods):
#     ax = axes[i // cols, i % cols]
#     create_global_frequency_bar_plot(hood, ax)

# # Adjust layout
# plt.tight_layout()
# plt.savefig('visualizations/neighborhood_word_global_frequency.png')
# plt.show()