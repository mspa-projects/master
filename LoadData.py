# Modules for loading csv files to data frames
# Following files are used here
# NHTS file has tractid (11 digit)
# Counties file has countyid (5 digit)
# Tracts file has tractid (11 digit)
import pandas as pd  # pandas for data frame operations

# Load counties, tracts and nhts files in csv format
# Set the data type of countyid and ansicode 
# Pandas will default the countyid and ansicode to number and drop leading 0s
# To prevent that, set the data type to 'str' while reading csv
counties = pd.read_csv('datasets/Gaz_counties_national/Gaz_counties_national_formatted.csv', dtype = {'countyid':str, 'ansicode':str});
tracts = pd.read_csv('datasets/Gaz_counties_national/Gaz_tracts_national_formatted.csv', dtype = {'tractid':str});
nhts = pd.read_csv('datasets/nhts/NHTS_2009_transfer_US.csv', dtype = {'tractid':str});

# Load the data to individual dataframes
df_counties = pd.DataFrame(counties);
df_tracts = pd.DataFrame(tracts);
df_nhts = pd.DataFrame(nhts);

# Join nhts and tracts data files using tractid
df_tmp = pd.merge(df_tracts, df_nhts, on='tractid', how='inner').dropna();

# Derive countyid by extract first 5 chars of tractid
df_tmp['countyid'] = df_tmp['tractid'].map(lambda x: str(x)[0:5]);

# Use the countyid derived above to join again counties data file
df_merged  = pd.merge(df_tmp, df_counties, on='countyid', how='inner').dropna();


