import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

current_directory = os.getcwd()
target_directory = r'C:\Users\emmal\OneDrive\Desktop\Fall 2023\STAT 303-1\Final Project'

if current_directory != target_directory:
    os.chdir(target_directory)

    
airbnb_data = pd.read_csv('AB_NYC_2019_cleaned.csv')
print(airbnb_data.columns)

sns.set_theme(style="darkgrid")


### --- Neighbourhood Group --- ###
# Calculate the percentage for each category
group_counts = airbnb_data['neighbourhood_group'].value_counts(normalize=True) * 100
labels = [label.split('=')[-1].strip() for label in group_counts.index]


## roomtype by neighbourhood
cross_tab_room_hood = pd.crosstab(airbnb_data['room_type'], airbnb_data['neighbourhood_group'])
print(cross_tab_room_hood)

plt.figure(figsize=(8, 6))
sns.heatmap(cross_tab_room_hood, annot=True, cmap='crest', cbar_kws={'label': 'Count'})
plt.title('Room Type vs. Neighborhood Group', fontsize=14, pad=15, weight='bold')
plt.ylabel('Room Type', labelpad=15, fontsize=13)
plt.xlabel('Neighbourhood Group', labelpad=15, fontsize=13)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.savefig('visualizations/heatmap_roomtype_neighbourhood.png')
plt.show()

# Plot normalized data
sns.set_theme(style="darkgrid")
plt.figure(figsize=(8, 6))
sns.barplot(x=group_counts.index, y=group_counts.values, palette='tab10')
plt.xlabel('Neighbourhood Group', labelpad=10)
plt.ylabel('Percentage', labelpad=10)
plt.title('Normalized Count of Neighbourhood Groups', fontsize=16, weight='bold', pad=15)
plt.savefig('visualizations/neighbourhood_group_count_normalized.png')
plt.show()

sns.catplot(data=airbnb_data, x='neighbourhood_group', y="price", kind="bar", hue='neighbourhood_group', palette='tab10', legend=False)
plt.title('Neighbourhood Group vs. Price', weight='bold', fontsize=12)
plt.xlabel('Neighbourhood Group', labelpad=10)
plt.ylabel('Price')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('visualizations/neighbourhood_group_price.png')
plt.show()

print(pd.concat(
    [(airbnb_data["neighbourhood_group"].value_counts(normalize=True)),
     (airbnb_data["neighbourhood_group"].value_counts())],
    axis=1, keys=["percentage", "freq"]
))


### --- Neighbourhood --- ###

# Group by 'neighbourhood' and calculate the average price
hood_grouped = airbnb_data.groupby('neighbourhood')['price'].mean().reset_index()
hood_grouped.columns = ['neighbourhood', 'avg_price']
hood_sorted = hood_grouped.sort_values(by='avg_price', ascending=False)

# Take the top 25 rows
top_hoods = hood_sorted.head(25)
top_hoods = top_hoods.merge(airbnb_data[['neighbourhood','neighbourhood_group']], left_on='neighbourhood', right_on='neighbourhood', how='left')
top_hoods.set_index('neighbourhood', inplace=True)

print(top_hoods)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_hoods, x="avg_price", y=top_hoods.index, hue='neighbourhood_group', palette='tab10')
plt.suptitle('Neighbourhood vs. Price - Top 25', weight='bold', fontsize=16)
plt.xlabel('Average Price (USD/day)', fontsize = 10, labelpad=10)
plt.ylabel('Neighbourhood', labelpad=10)
plt.xticks(fontsize=10)
plt.yticks(rotation=15, fontsize=8)
plt.legend(title='Neighbourhood Group', loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('visualizations/neighbourhood_price.png')
plt.show()



# neighborhood_counts = airbnb_data["neighbourhood"].value_counts()
# neighborhood_percentage = airbnb_data["neighbourhood"].value_counts(normalize=True)
# neighborhood_data = pd.concat([neighborhood_percentage, neighborhood_counts], axis=1, keys=["percentage", "freq"])

# # Merge 'neighbourhood_group' column from 'airbnb_data'
# neighborhood_data = neighborhood_data.merge(airbnb_data[['neighbourhood','neighbourhood_group']], left_index=True, right_on='neighbourhood', how='inner')
# neighborhood_data.drop_duplicates('neighbourhood', inplace=True)
# neighborhood_data.set_index('neighbourhood', inplace=True)

# top_neighborhoods = neighborhood_data.head(25)

# # Plot barplot
# plt.figure(figsize=(12, 6))
# sns.barplot(y=top_neighborhoods.index, x=top_neighborhoods["freq"], hue=top_neighborhoods['neighbourhood_group'], palette='tab10', dodge=False)
# plt.xlabel('Frequency', labelpad=15)
# plt.ylabel('Neighborhood')
# plt.yticks(rotation=15, fontsize=8)
# plt.title('Top 25 Neighborhoods by Frequency', fontsize=16, weight='bold', pad=15)
# plt.legend(title='Neighbourhood Group', loc='upper right', bbox_to_anchor=(1.25, 1))
# plt.tight_layout()
# plt.savefig('visualizations/top_25_neighborhoods_count.png')
# plt.show()

# print(pd.concat(
#     [(airbnb_data["neighbourhood"].value_counts(normalize=True)),
#      (airbnb_data["neighbourhood"].value_counts())],
#     axis=1, keys=["percentage", "freq"]).head(15)
#     )
