import pandas as pd  # pandas for data frame operations

# initial work with the cleansed dataset after merging and aggregating
highway = pd.read_csv('county_and_nhts_inp_to_clustering.csv', sep = ',')

# examine the shape of the DataFrame
print(highway.shape)

# look at the list of column names, note that y is the response
list(highway.columns.values)


# look at the beginning of the DataFrame
highway.head()

# create dataframe of the columns we're looking for
df = pd.DataFrame(highway, columns=['usps','pop10','hu10','aland','awater','aland_sqmi',
    'awater_sqmi','est_pmiles2007_11','est_ptrp2007_11','est_vmiles2007_11',
    'est_vtrp2007_11','median_hh_inc2007_11','mean_hh_veh2007_11','mean_hh_mem2007_11',
    'pct_owner2007_11','mean_hh_worker2007_11','pct_lchd2007_11','pct_lhd12007_11',
    'pct_lhd22007_11','pct_lhd42007_11'])

# Summarize all the columns in the dataframe
df.describe()

# Summarize all the columns in the dataframe, filtering by IL counties only
df[df['usps'] == 'IL'].describe()

# When we want to describe all the counties within a specific cluster, we'll
# need to add the cluster column in the .DataFrame() step on line #17 and
# repeat line #27 to result the cluster type/number.
#
# Example: df[df['cluster'] == '2'].describe()