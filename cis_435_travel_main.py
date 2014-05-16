import pandas as pd  # pandas for data frame operations

# Load NHTS detail file
nhts = pd.read_csv('NHTS_2009_transfer_US.txt', '\t', dtype = {'tractid':str});

# Load the data to individual dataframes and remove null rows
df_initial = pd.DataFrame(nhts).dropna();

# Derive countyid by extracting first 5 chars of tractid
df_initial['countyid'] = df_initial['tractid'].map(lambda x: str(x)[0:5]);

# Select only relevant features from nhts data frame
df_nhts = df_initial[['tractid', 'countyid', 'cluster', 'urban_group', 'est_pmiles2007_11', 'est_ptrp2007_11', 'est_vmiles2007_11', 'est_vtrp2007_11', 'median_hh_inc2007_11', 'mean_hh_veh2007_11', 'mean_hh_mem2007_11', 'pct_owner2007_11', 'mean_hh_worker2007_11', 'pct_lchd2007_11', 'pct_lhd12007_11', 'pct_lhd22007_11', 'pct_lhd42007_11']]

# Aggregate the nhts dataframe on the countyid and calculate mean() for numeric features
# This will roll-up the detail nhts file and create one row for each county
df_nhts_agg = df_nhts.groupby(by='countyid').mean()

# Save the aggregated nhts file to text
df_nhts_agg.to_csv('output.txt')

# Read the new files
nhts_agg = pd.read_csv('output.txt', ',', dtype = {'countyid':str});
counties = pd.read_csv('Gaz_counties_national_V2.txt', '\t', dtype = {'countyid':str})

# Convert the file objects to dataframes
df_nhts_agg = pd.DataFrame(nhts_agg)
df_counties = pd.DataFrame(counties)

# Join nhts agg dataframe and counties dataframe on countyid feature
# This creates a merged dataframe that has data from both couties and nhts files
df_merged_tmp  = pd.merge(df_nhts_agg, df_counties, on='countyid', how='inner')

# Create the final dataframe that will only have relevant features and will
# be the input datastructure for building the model
df_final = df_merged_tmp[['countyid', 'pop10', 'hu10', 'aland', 'awater', 'aland_sqmi', 'awater_sqmi', 'statepop', 'statearea', '%pop', '%area', 'est_pmiles2007_11', 'est_ptrp2007_11', 'est_vmiles2007_11', 'est_vtrp2007_11', 'median_hh_inc2007_11', 'mean_hh_veh2007_11', 'mean_hh_mem2007_11', 'pct_owner2007_11', 'mean_hh_worker2007_11', 'pct_lchd2007_11', 'pct_lhd12007_11', 'pct_lhd22007_11', 'pct_lhd42007_11']]


