import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os

current_directory = os.getcwd()
target_directory = r'C:\Users\emmal\OneDrive\Desktop\Fall 2023\STAT 303-1\Final Project'

if current_directory != target_directory:
    os.chdir(target_directory)


airbnb_data = pd.read_csv('AB_NYC_2019_cleaned.csv')
print(airbnb_data.columns)


## ---- ##
sns.set_theme(style="darkgrid")
## ---- ##



### ----------------------------------- Host Data ----------------------------------- #### 
airbnb_data.rename(columns={'id':'listing_id'}, inplace=True)

# Group data by host_id and calculate host listing count and average price
host_listing_df = airbnb_data.groupby('host_id')['calculated_host_listings_count'].max().reset_index()
host_listing_df.rename(columns={'calculated_host_listings_count': 'listing_count'}, inplace=True)

host_listing_df['average_price'] = host_listing_df['host_id'].apply(
    lambda host_id: airbnb_data[airbnb_data['host_id'] == host_id]['price'].mean())

host_listing_df = host_listing_df.sort_values(by='listing_count', ascending=False)

# Merge host_listing_df with airbnb_data to add 'host_name'
host_listing_df = pd.merge(host_listing_df, airbnb_data[['host_id', 'host_name']], on='host_id', how='left')

# Filter multi-listing hosts
multi_listing_hosts = host_listing_df[host_listing_df['listing_count'] > 1]['host_id'].unique()
multi_listing_df = airbnb_data[airbnb_data['host_id'].isin(multi_listing_hosts)]

# # --- Top Five Hosts Pie Chart --- #

# # Extract information about the top five hosts
# top_five_hosts = host_listing_df['host_id'].value_counts().reset_index().head(5)
# top_five_hosts = pd.merge(top_five_hosts, airbnb_data[['host_id', 'host_name']], on='host_id', how='left').drop_duplicates()

# # Convert host_id to string for labels
# labels_host = [(f"{host['host_name']} \n(id: {str(host['host_id'])})") for _, host in top_five_hosts.iterrows()]

# # Create a pie chart for the top five hosts
# plt.figure(figsize=(8, 6), facecolor=[0.9, 0.9, 0.9])
# plt.pie(top_five_hosts['count'], explode=(0.2, 0, 0, 0, 0), autopct='%2.2f%%', 
#         shadow=True, wedgeprops = {'linewidth' : 0}, center=(0, -250))

# plt.title('Top Five Hosts by Listing Count', pad=20, weight='bold', fontsize=16)
# plt.legend(labels_host, loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=10)
# plt.tight_layout()
# plt.savefig('visualizations/top_five_hosts_distribution.png')
# plt.show()


# --- Host Count Percentiles --- #

# Calculate percentiles for hosts with single or fewer listings and hosts with more than one listing
total_listings = host_listing_df['listing_count'].sum()
total_hosts = len(host_listing_df)

print(total_listings)
print(type(total_listings))

sum_listings_host_single_or_less = host_listing_df[host_listing_df['listing_count'] <= 1]['listing_count'].sum()
sum_listings_host_more_than_one = host_listing_df[host_listing_df['listing_count'] > 1]['listing_count'].sum()

perc_list_single_or_less = sum_listings_host_single_or_less / total_listings
perc_list_multi_listing = sum_listings_host_more_than_one / total_listings

# Print and visualize host count percentiles
print(f'''
Sum of listings for hosts with <=1 or fewer listings: {sum_listings_host_single_or_less} ({round(perc_list_single_or_less, 3)}%)
Sum of listings for hosts with >1 listing: {sum_listings_host_more_than_one} ({round(perc_list_multi_listing, 3)}%)''')


# Percentiles for hosts with single or fewer listings and hosts with more than one listing
perc_hosts_single_or_less = len(host_listing_df[host_listing_df['listing_count'] <= 1]) / total_hosts
perc_hosts_multi_listing = len(host_listing_df[host_listing_df['listing_count'] > 1]) / total_hosts
print(f'''
Sum of hosts with <=1 or fewer listings: {sum_listings_host_single_or_less} ({round(perc_list_single_or_less, 3)}%)
Sum of hosts with >1 listing: {sum_listings_host_more_than_one} ({round(perc_list_multi_listing, 3)}%)''')



# Assign bins using cut
def is_multi(row):
    if row['listing_count'] > 1:
        return 'Multi-Listing'
    else:
        return 'Single or Less'

# Apply the function to create the 'bin_label' column
host_listing_df['bin_label'] = host_listing_df.apply(is_multi, axis=1)

# Function to filter DataFrame based on listing count and operator
def get_df_listing_count(df, listing_count, operator='=='):
    operators = {
        '==': lambda x, y: x == y, 
        '>=': lambda x, y: x >= y, '>': lambda x, y: x > y,
        '<=': lambda x, y: x <= y, '<': lambda x, y: x < y
    }
    
    if operator in operators:
        return df[operators[operator](df['listing_count'], listing_count)]
    else:
        raise ValueError("Invalid operator. Supported operators are '==', '>=', '>', and '<'.")

# Function to calculate the percentage of hosts in a DataFrame
def get_perc(df):
    return len(df) / total_hosts

# Calculate host count percentiles
percentiles_host_num_listings = []

# Iterate through unique listing counts
for i in host_listing_df['listing_count'].unique():
    num_listings = i
    temp_df = get_df_listing_count(host_listing_df, num_listings, '>')
    quantity_of_hosts = len(temp_df)
    perc = get_perc(temp_df)

    temp_dict = {'listing_count': num_listings, 'quantity_of_hosts': quantity_of_hosts, 'perc_of_hosts': perc}

    percentiles_host_num_listings.append(temp_dict)

percentiles_host_num_listings = pd.DataFrame(percentiles_host_num_listings)

# Display the DataFrame
print(percentiles_host_num_listings)


# ----------------------------------- Bar Plot of Percentiles by Listing Count ----------------------------------- #

# Create a bar plot of host count percentiles
plt.figure(figsize=(12, 6))
sns.barplot(data=percentiles_host_num_listings, x='listing_count', y='perc_of_hosts')
plt.title('Percentage of Hosts by Listing Count', pad=15, weight='bold', fontsize=16)
plt.xlabel('Listing Count')
plt.ylabel('Percentage of Hosts\n')
plt.xticks(rotation=40, fontsize=8)
plt.yticks(fontsize=10)
plt.savefig('visualizations/perc_by_host_listing_count.png')
plt.show()


# ----------------------------------- Histogram of Host Listing Counts ----------------------------------- #

# # Histogram of host listing counts (30 bins)
# plt.figure(figsize=(10, 6))
# sns.histplot(host_listing_df['listing_count'], bins=30, kde=True)
# plt.title('Distribution of Host Listing Counts', pad=15, weight='bold', fontsize=16)
# plt.xlabel('Number of Listings')
# plt.ylabel('Frequency (in thousands)\n')
# plt.margins(x=0.02, y=0.02)    
# plt.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.savefig('visualizations/histogram_of_host_listing_count.png')
# plt.show()

# # Histogram of host listing counts (50 cap)
# plt.figure(figsize=(10, 6))
# sns.histplot(host_listing_df['listing_count'], bins=range(1, 51), kde=True)
# plt.suptitle('Distribution of Host Listing Counts', weight='bold', fontsize=16)
# plt.title('(50 cap)')
# plt.xlabel('Number of Listings')
# plt.ylabel('Frequency (in thousands)\n')
# plt.margins(x=0.01, y=0.05)    
# plt.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
# plt.xlim(0, 50)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.tight_layout()
# plt.savefig('visualizations/histogram_of_host_listing_50_cap.png')
# plt.show()

# # Histogram of host listing counts (20 cap)
# plt.figure(figsize=(10, 7))
# sns.histplot(host_listing_df['listing_count'], bins=range(1, 20), kde=True)
# plt.title('Distribution of Host Listing Counts', weight='bold', fontsize=16)
# plt.title('(20 cap)')
# plt.xlabel('Number of Listings')
# plt.ylabel('Frequency (in thousands)')
# plt.margins(x=0, y=0.05)    
# plt.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
# plt.xlim(0, 20)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.tight_layout()
# plt.savefig('visualizations/histogram_of_host_listing_20_cap.png')
# plt.show()


# # Bar plot of host count percentiles 
# plt.figure(figsize=(6, 5))
# plt.bar(['Single or Less', 'Multi-Listing'], [perc_list_single_or_less, perc_list_multi_listing])
# plt.title('Portion of Listings by Host Type', pad=10, weight='bold', fontsize=16)
# plt.xlabel("Host's Listing Count")
# plt.ylabel('Percentage of Total Listings')
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.savefig('visualizations/barplot_perc_listings_host_type.png')
# plt.show()

# # Bar plot of host count percentiles
# plt.figure(figsize=(6, 5))
# plt.bar(['Single or Less', 'Multi-Listing'], [perc_hosts_single_or_less, perc_hosts_multi_listing])
# plt.title('Portion of Hosts by Host Type', pad=20, weight='bold', fontsize=16)
# plt.xlabel("Host's Listing Count")
# plt.ylabel('Percentage of Total Hosts')
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.savefig('visualizations/barplot_perc_host_type.png')
# plt.show()





# ----------------------------------- Map of Mid-Range Hosts by Neighbourhood Group ----------------------------------- #
# Load the NYC map image, set the palette, and define the extent
img = plt.imread("Neighbourhoods_New_York_City_Map.png")
my_palette = sns.color_palette('Set1') 
sns.set_style("whitegrid", {'axes.grid' : False})

x1, x2, y1, y2 = -74.260, -73.688, 40.490, 40.916

# ### --- Cropped Map --- ###
# # Create a scatter plot to overlay Airbnb listings on the map
# plt.figure(figsize=(10, 12.5), dpi=100)
# plt.imshow(img, extent=[x1, x2, y1, y2], cmap='gray')
# sns.scatterplot(x=airbnb_data['longitude'], y=airbnb_data['latitude'], hue=airbnb_data['neighbourhood_group'],
#                 palette=my_palette, s=5, alpha=0.1, linewidth=0.1, legend=False)
# plt.xlim(x1, x2)
# plt.ylim(y1, y2)
# plt.title("Listings by Neighbourhood Group", pad=20, weight='bold', fontsize=16)
# plt.xlabel("Longitude", labelpad=10, fontsize=10)
# plt.ylabel("Latitude", labelpad=10, fontsize=10)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.tight_layout()
# plt.savefig('visualizations/nyc_airbnb_map_neighbourhood_group_cropped.png')
# plt.show()



# # ----------------------------------- Map of Mid-Range Hosts by Neighbourhood ----------------------------------- #

# # Create a scatter plot to overlay Airbnb listings on the map
# plt.figure(figsize=(12, 18), dpi=100)
# plt.imshow(img, extent=[x1, x2, y1, y2], cmap='gray')
# sns.scatterplot(x=airbnb_data['longitude'], y=airbnb_data['latitude'], hue=airbnb_data['neighbourhood'],
#                 palette=my_palette, s=5, alpha=0.1, linewidth=0.1, legend=False)
# plt.xlim(x1, x2)
# plt.ylim(y1, y2)
# plt.title("Listings by Neighbourhood", pad=20, weight='bold', fontsize=16)
# plt.xlabel("Longitude", labelpad=10, fontsize=10)
# plt.ylabel("Latitude", labelpad=10, fontsize=10)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.tight_layout()
# plt.savefig('visualizations/nyc_airbnb_map_neighbourhood.png')
# plt.show()


# ----------------------------------- Map of Mid-Range Hosts by Listing Count ----------------------------------- #

# Define the 98.5th percentile for listing count
count_95_perc = host_listing_df['listing_count'].quantile(0.95)

# Filter hosts with more than 3 listings and less than the 95th percentile
mid_sub_list = host_listing_df[
    (host_listing_df['listing_count'] > 3) &
    (host_listing_df['listing_count'] < count_95_perc)]['host_id'].tolist()

# Create DataFrame for mid-range hosts
mid_sub_df = airbnb_data[airbnb_data['host_id'].isin(mid_sub_list)]

# Specify the number of unique host_ids
num_colors = len(mid_sub_df['host_id'].unique())
my_palette = sns.hls_palette(n_colors=num_colors, s=1)

num_mid_hosts = len(mid_sub_df['host_id'].unique())


# Create a scatter plot to overlay Airbnb listings on the map
plt.figure(figsize=(9, 6), dpi=100)
plt.imshow(img, extent=[x1, x2, y1, y2], cmap='gray')
sns.scatterplot(x=mid_sub_df['longitude'], y=mid_sub_df['latitude'], hue=mid_sub_df['host_id'],
                palette=my_palette, s=5, alpha=0.3, linewidth=0, legend=False)
plt.xlim(x1, x2)
plt.ylim(y1, y2)
plt.title("Listings Color-Coded by Host Listings Count (Mid-Range)", pad=20, weight='bold', fontsize=16)

plt.text(x2, y1-0.045, f'{len(mid_sub_df)} listings\n{num_mid_hosts} hosts',
         bbox=dict(facecolor=[0.85, 0.85, 0.85], edgecolor='black', linewidth=0.5),
         fontsize=8, color='black', ha='right')

plt.xlabel("Longitude", labelpad=10, fontsize=10)
plt.ylabel("Latitude", labelpad=10, fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
# plt.tight_layout()
plt.savefig('visualizations/nyc_airbnb_map_mid.png')
plt.show()

# ----------------------------------- Map of Top 1.5% of Hosts by Listing Count -----------------------------------

# Filter top 1.5% of hosts by listing count
top_sub_list = host_listing_df[host_listing_df['listing_count'] > count_95_perc]['host_id'].unique()
top_sub_df = airbnb_data[airbnb_data['host_id'].isin(top_sub_list)]

num_top_hosts = len(top_sub_df['host_id'].unique())


# Create a scatter plot to overlay Airbnb listings on the map
plt.figure(figsize=(9, 6), dpi=100)
plt.imshow(img, extent=[x1, x2, y1, y2], cmap='gray')
sns.scatterplot(x=top_sub_df['longitude'], y=top_sub_df['latitude'], hue=top_sub_df['host_id'],
                palette='gnuplot2', s=3, alpha=0.6, linewidth=0.1, legend=False)
plt.xlim(x1, x2)
plt.ylim(y1, y2)
plt.title("Listings Color-Coded by Host (Top 1.5%)", pad=20, weight='bold', fontsize=16)

plt.text(x2, y1-0.045, f'{len(top_sub_df)} listings\n{num_top_hosts} hosts',
         bbox=dict(facecolor=[0.85, 0.85, 0.85], edgecolor='black', linewidth=0.5),
         fontsize=8, color='black', ha='right')

plt.xlabel("Longitude", labelpad=10, fontsize=10)
plt.ylabel("Latitude", labelpad=10, fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
# plt.tight_layout()
plt.savefig('visualizations/nyc_airbnb_map_top.png')
plt.show()

# # ----------------------------------- Map of Hosts with <=3 Listing -----------------------------------

# # Filter hosts with only 1 listing
# low_sub_list = host_listing_df[host_listing_df['listing_count'] <=3]['host_id'].tolist()
# low_sub_df = airbnb_data[airbnb_data['host_id'].isin(low_sub_list)]
# num_low_hosts = len(low_sub_df['host_id'].unique())


# # Create a scatter plot to overlay Airbnb listings on the map
# plt.figure(figsize=(8, 10), dpi=100)
# plt.imshow(img, extent=[x1, x2, y1, y2], cmap='gray')
# sns.scatterplot(x=low_sub_df['longitude'], y=low_sub_df['latitude'], hue=low_sub_df['neighbourhood_group'], s=5, alpha=0.2, linewidth=0.1, legend=False)
# plt.xlim(x1, x2)
# plt.ylim(y1, y2)
# plt.suptitle('Color-Coded by Neighbourhood Group', fontsize=16, weight='bold')
# plt.title("Color-Coded by Neighbourhood (host has <=3 listings)", pad=15, fontsize=10)

# plt.text(x2, y1-0.045, f'{len(low_sub_df)} listings\n{num_low_hosts} hosts',
#          bbox=dict(facecolor=[0.85, 0.85, 0.85], edgecolor='black', linewidth=0.5),
#          fontsize=8, color='black', ha='right')

# plt.xlabel("Longitude", labelpad=10, fontsize=10)
# plt.ylabel("Latitude", labelpad=10, fontsize=10)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.tight_layout()
# plt.savefig('visualizations/nyc_airbnb_map_low.png', dpi=300, bbox_inches='tight')
# plt.show()



# ----------------------------------- Map with Color by Price (only hosts w/ multilisting) -----------------------------------

# Map with color quantiles
plt.figure(figsize=(9,6), dpi=100)
plt.imshow(img, extent=[x1, x2, y1, y2], cmap='gray')
sns.scatterplot(x=multi_listing_df['longitude'], y=multi_listing_df['latitude'],
                hue=multi_listing_df[(multi_listing_df['price'] < 500)]['price'],
                palette=sns.color_palette("magma", as_cmap=True),
                alpha=0.4, s=3, linewidth=0.05)
plt.xlim(x1, x2)
plt.ylim(y1, y2)
plt.suptitle("Listings by Price", weight='bold', fontsize=16)
plt.title("(multi-lising hosts only, cap $500)", pad=15, fontsize=10)
plt.xlabel("Longitude", labelpad=10, fontsize=10)
plt.ylabel("Latitude", labelpad=10, fontsize=10)
plt.legend(markerscale=5, fontsize=8, title='Price (USD/day)', title_fontsize=10, loc='lower right', bbox_to_anchor=(1.2, 0))
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
# plt.tight_layout()
plt.savefig("visualizations/nyc_airbnb_map_price.png", dpi=300, bbox_inches='tight')
plt.show()



# # ----------------------------------- Host Statistics and Visualizations -----------------------------------

## ----- ##
sns.set_theme(style="darkgrid")
## ----- ##

# Create DataFrame for relevant host data
filter_host_df = host_listing_df[['host_id', 'listing_count', 'average_price']]

# Find host with the most listings
max_listing_host_id = filter_host_df.iloc[0]['host_id']
max_listing_host = airbnb_data[airbnb_data['host_id'] == max_listing_host_id].iloc[0]

# Find host with the highest average price
max_avg_price_host_id = filter_host_df[filter_host_df['average_price'] == filter_host_df['average_price'].max()].iloc[0]['host_id']
max_avg_price_host = airbnb_data[airbnb_data['host_id'] == max_avg_price_host_id].iloc[0]

# Print relevant host information
print(f"\nThe host with the most listings is {max_listing_host['host_name']} with {int(filter_host_df.iloc[0]['listing_count'])} listings")
print(f"\nThe highest average price is ${round(filter_host_df['average_price'].max(), 2)} hosted by {max_avg_price_host['host_name']} (host id {max_avg_price_host_id})")
print(f"\nAverage number of listings per host: {round(filter_host_df['listing_count'].mean(), 3)}")
print(f"\nCorrelation between the number of listings a host has and the average price:\n {filter_host_df.corr()}")

# ----------------------------------- Scatterplot for Number of Listings vs. Average Price -----------------------------------

# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='listing_count', y='average_price', data=filter_host_df, alpha=0.1, linewidth=0)
# plt.xlabel('Number of Listings')
# plt.ylabel('Average Price (USD/day)')
# plt.title('Average Price vs. Number of Listings', pad=20, weight='bold', fontsize=16)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.savefig('visualizations/avg_price_num_listings_scatter_all.png')
# plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='listing_count', y='average_price', data=filter_host_df, alpha=0.1, linewidth=0)
plt.xlabel('Number of Listings')
plt.ylabel('Average Price (USD/day)')
plt.suptitle('Average Price vs. Number of Listings', weight='bold', fontsize=16)
plt.title('(price < $2000 & listings <50)', pad=15, fontsize=10)
plt.xlim(0, 50)
plt.ylim(0, 2000)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('visualizations/avg_price_num_listings_scatter_sub.png')
plt.show()




# # ----------------------------------- Boxplot for Listing Count and Average Price (>5) -----------------------------------

# # Filter hosts with more than 5 listings
# sub_host_df_5_more = filter_host_df[(filter_host_df['listing_count'] > 5) & (filter_host_df['average_price'] < 1500)]

# plt.figure(figsize=(10, 6))
# sns.boxplot(x='listing_count', y='average_price', data=sub_host_df_5_more, showfliers=True)
# plt.xlabel('Number of Listings')
# plt.ylabel('Average Price (USD/day)')
# plt.suptitle('Average Price vs. Number of Listings', weight='bold', fontsize=16)
# plt.title('(listings > 5 & price < 1500)', pad=15)
# plt.xticks(rotation=70, fontsize=10)
# plt.yticks(fontsize=10)
# plt.savefig('visualizations/avg_price_num_listings_box_over_5.png')
# plt.show()



# # ----------------------------------- Boxplot for Listing Count and Average Price (<5) -----------------------------------

# # Filter hosts with less than 5 listings
# sub_host_df_5_less = filter_host_df[filter_host_df['listing_count'] < 5]

# # With outliers
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='listing_count', y='average_price', data=sub_host_df_5_less, showfliers=True)
# plt.xlabel('Number of Listings')
# plt.ylabel('Average Price (USD/day)')
# plt.suptitle('Average Price vs. Number of Listings', weight='bold', fontsize=16)
# plt.title('(listings < 5)', pad=15)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.savefig('visualizations/avg_price_num_listings_box_under_5_with_outliers.png')
# plt.show()

# # Without outliers
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='listing_count', y='average_price', data=sub_host_df_5_less, showfliers=False)
# plt.xlabel('Number of Listings', labelpad=10)
# plt.ylabel('Average Price (USD/day)')
# plt.suptitle('Average Price vs. Number of Listings',  weight='bold', fontsize=16)
# plt.title('(listings < 5 & No Outliers)', pad=15)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.savefig('visualizations/avg_price_num_listings_box_under_5_no_outliers.png')
# plt.show()


# # Also < $3000
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='listing_count', y='average_price', data=sub_host_df_5_less[sub_host_df_5_less['average_price']<3000], showfliers=True)
# plt.xlabel('Number of Listings')
# plt.ylabel('Average Price (USD/day)', labelpad=10)
# plt.suptitle('Average Price vs. Number of Listings', weight='bold', fontsize=16)
# plt.title('(listings < 5 & price < 3000)', pad=15)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.savefig('visualizations/avg_price_num_listings_box_under_5_with_outliers.png')
# plt.show()





# ----------------------------------- Barplot for Listing Count and Average Price -----------------------------------

print(filter_host_df['listing_count'].unique())
print(len((filter_host_df['listing_count']).unique()))
print(round(len((filter_host_df['listing_count']).unique())/2))

k = 2


unique_counts = filter_host_df['listing_count'].unique()

# Calculate the number of bins
num_bins = math.ceil(len(unique_counts) / k)

q, i = pd.qcut(unique_counts, num_bins, retbins=True)

cut_intv= [round(x) for x in i.tolist()]



count_bin = pd.cut(filter_host_df['listing_count'], bins=cut_intv, retbins=True)
filter_host_df['count_bin'] = count_bin[0]

new_labels = []

for label in range(len(cut_intv)-1):
    new_labels.append(f'{cut_intv[label]}-{cut_intv[label+1]}')
    
filter_host_df['count_bin'] = filter_host_df['count_bin'].cat.rename_categories(new_labels)


plt.figure(figsize=(10, 6))
sns.barplot(x='count_bin', y='average_price', data=filter_host_df)
plt.xlabel('Number of Listings')
plt.ylabel('Average Price (USD/per)', labelpad=10)
plt.title('Average Price vs. Number of Listings', pad=20, weight='bold', fontsize=16)
plt.xticks(rotation=45, fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig('visualizations/avg_price_listing_count_bar.png')
plt.show()


# # ----------------------------------- Histogram for Listing Count -----------------------------------

# plt.figure(figsize=(10, 6))
# sns.distplot(filter_host_df['listing_count'])
# plt.xlabel('Number of Listings')
# plt.ylabel('Frequency', labelpad=10)
# plt.title('Distribution of Number of Listings', pad=20, weight='bold', fontsize=16)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.savefig('visualizations/avg_price_listing_count_hist.png')
# plt.show()


 
# # ----------------------------------- Distance Between Hosts Listings ----------------------------------- #

# # Assuming you have a haversine function to calculate distances
# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371  # Earth radius in kilometers
#     dlat = np.radians(lat2 - lat1)
#     dlon = np.radians(lon2 - lon1)
#     a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#     distance = R * c
#     return distance

# # Create a new column 'distance_to_other_listings' to store the distance between each pair of listings for the same host
# multi_listing_df['distance_to_other_listings'] = multi_listing_df.apply(lambda row: multi_listing_df.apply(lambda inner_row: haversine(row['latitude'], row['longitude'], inner_row['latitude'], inner_row['longitude']), axis=1).tolist(), axis=1)

# # Initialize an empty dictionary to store flattened lists for each host
# flat_distances_dict = {}

# for host, host_df in multi_listing_df.groupby('host_id'):
#     listing_dists = [item for sublist in host_df['distance_to_other_listings'] for item in sublist if item != 0.0]
#     flat_distances_dict[host] = list(set(listing_dists))

# # Update 'distance_to_other_listings' in host_listing_df using the dictionary
# host_listing_df['distance_to_other_listings'] = host_listing_df['host_id'].map(flat_distances_dict)
# host_listing_df.drop_duplicates(subset=['host_id'], inplace=True)

# host_listing_df['avg_dist'] = host_listing_df['distance_to_other_listings'].apply(lambda x: np.mean(x) if x else np.nan)

# # Print the maximum and minimum average distances
# max_avg_dist = host_listing_df['avg_dist'].max()
# min_avg_dist = host_listing_df['avg_dist'].min()

# print(f'max - {max_avg_dist:.2f} km')
# print(f'min - {min_avg_dist:.2f} km')

# # Histogram of average distances between host listings
# plt.figure(figsize=(10, 6))
# sns.histplot(host_listing_df['avg_dist'], bins=20, kde=True)
# plt.title('Distribution of Average Distances Between Listings', pad=20, weight='bold', fontsize=16)
# plt.xlabel('Average Distance (km)')
# plt.ylabel('Frequency')
# plt.xticks(rotation=70, fontsize=10)
# plt.yticks(fontsize=10)
# plt.savefig('visualizations/avg_dist_between_host_listings.png')
# plt.show()

# # Scatter Plot: Average Distance vs. Number of Listings and Average Price
# plt.figure(figsize=(12, 6))

# # Scatter plot: Average Distance vs. Number of Listings
# plt.subplot(1, 2, 1)
# sns.scatterplot(data=host_listing_df, x='listing_count', y='avg_dist', alpha=0.5)
# plt.title('Average Distance vs. Number of Listings', pad=20, weight='bold', fontsize=16)
# plt.xlabel('Number of Listings')
# plt.ylabel('Average Distance (km)')
# plt.xticks(rotation=70, fontsize=10)
# plt.yticks(fontsize=10)

# # Scatter plot: Average Distance vs. Average Price
# plt.subplot(1, 2, 2)
# sns.scatterplot(data=host_listing_df, x='average_price', y='avg_dist', alpha=0.5)
# plt.title('Average Distance vs. Average Price', pad=20, weight='bold', fontsize=16)
# plt.xlabel('Average Price (USD/day)')
# plt.ylabel('Average Distance (km)')
# plt.xticks(rotation=70, fontsize=10)
# plt.yticks(fontsize=10)

# plt.tight_layout()
# plt.savefig('visualizations/avg_dist_price_and_num_listings.png')
# plt.show()