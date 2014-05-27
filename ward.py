# -*- coding: utf-8 -*-
'''
This module implements the Hierarchical (Ward) Model of clustering
using NHTS and Couties dataset.

Inputs:
    trainingData: A dataframe or file name that has nhts-couties data
    type: indicator to classifiy the type of trainingData. Use 0 to indicate 
        that trainingData is a data frame and 1 to indicate it is a filename
    sep: If trainingData is a file, provide the field separator
    clusterComp: provide the number of cluster components; defaults to 6
    pcaFlag: boolean value to specify if pca should be run on the trainingData
    pcaComp: number of pca components

Output:
    cluster dataframe: a dataframe of the cluster in which McLean county was found
    score: silhouette coefficient for Hierarchical (Ward) model

'''

import numpy as np
from sklearn.cluster import Ward
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import pandas as pd

# Main module that acts as the API to the wrapper module
# This module builds the Hierarchical (Ward) model, fits to the training data
# and returns the silhouette score
def buildWardModel(
		trainingData,
		type=0, # 0 implies 'trainingData' is a dataframe and 1 implies it is file name
		sep=',',
		clusterComp=6,  
		pcaFlag = False, 
		pcaComp=6
    ) :
    
    # trainingData is a data frame if type = 0
    # trainingData is a filename if type = 1
    if (type == 0):
        df_nhtsagg = trainingData
    else :
        df_nhtsagg = pd.DataFrame(pd.read_csv(trainingData, sep))
    
    # Extract and save relevant features from trainingData and store in local
    # numpy arrays    
    # Refer to the data dictionary for details of what each feature means
    pop10 = np.array(df_nhtsagg['pop10'])
    hu10 = np.array(df_nhtsagg['hu10'])
    aland = np.array(df_nhtsagg['aland'])
    awater = np.array(df_nhtsagg['awater'])
    aland_sqmi = np.array(df_nhtsagg['aland_sqmi'])
    awater_sqmi = np.array(df_nhtsagg['awater_sqmi'])
    statepop = np.array(df_nhtsagg['statepop'])
    statearea = np.array(df_nhtsagg['statearea'])
    perc_pop = np.array(df_nhtsagg['perc_pop'])
    perc_area = np.array(df_nhtsagg['perc_area'])
    est_pmiles2007_11 = np.array(df_nhtsagg['est_pmiles2007_11'])
    est_ptrp2007_11 = np.array(df_nhtsagg['est_ptrp2007_11'])
    est_vmiles2007_11 = np.array(df_nhtsagg['est_vmiles2007_11'])
    est_vtrp2007_11 = np.array(df_nhtsagg['est_vtrp2007_11'])
    median_hh_inc2007_11 = np.array(df_nhtsagg['median_hh_inc2007_11'])
    mean_hh_veh2007_11 = np.array(df_nhtsagg['mean_hh_veh2007_11'])
    mean_hh_mem2007_11 = np.array(df_nhtsagg['mean_hh_mem2007_11'])
    pct_owner2007_11 = np.array(df_nhtsagg['pct_owner2007_11'])
    mean_hh_worker2007_11 = np.array(df_nhtsagg['mean_hh_worker2007_11'])
    pct_lchd2007_11 = np.array(df_nhtsagg['pct_lchd2007_11'])
    pct_lhd12007_11 = np.array(df_nhtsagg['pct_lhd12007_11'])
    pct_lhd22007_11 = np.array(df_nhtsagg['pct_lhd22007_11'])
    pct_lhd42007_11 = np.array(df_nhtsagg['pct_lhd42007_11'])
    
    # Extract and transpose the input features
    X_train = np.array([
  		np.array(pop10),
#  		np.array(hu10),
#  		np.array(aland),
#  		np.array(awater),
  		np.array(aland_sqmi)
#  		np.array(awater_sqmi),
#		np.array(statepop),
#		np.array(statearea),
#  		np.array(perc_pop),
#  		np.array(perc_area),
#  		np.array(est_pmiles2007_11),
#  		np.array(est_ptrp2007_11),
#  		np.array(est_vmiles2007_11),
#  		np.array(est_vtrp2007_11),
#  		np.array(median_hh_inc2007_11),
#  		np.array(mean_hh_veh2007_11),
#  		np.array(mean_hh_mem2007_11),
#  		np.array(pct_owner2007_11),
#  		np.array(mean_hh_worker2007_11),
#  		np.array(pct_lchd2007_11),
#  		np.array(pct_lhd12007_11),
#  		np.array(pct_lhd22007_11),
#  		np.array(pct_lhd42007_11)
   	]).T
   
    # Perform standard scaling on training data, X, to ensure all features have
    # one unit of variance and no single feature dominates the clustering process
    X = StandardScaler().fit_transform(X_train)

    # Execute the gaussian mixture model using following parameters
    # n_clusters: number of components/clusters to be identified
    clf = Ward(n_clusters=clusterComp)
    clf.fit(X)
    
    # Predict the cluster labels using the training data
    # As this is an unsupervised clustering methodology, we'll not employ train-test
    # validation procedure for building the clusters
    # Instead, we'll use the same dataset that was used for fitting the model
    # for classifying it and identifying the labels/clusters
    y = clf.labels_
    
    # Add the cluster labels to input data frame
    df_nhtsagg['label'] = y.tolist()
     
    # For McLean county (17113) return a data frame that has all counties
    # that belong to the same cluster as McLean county, IL
    df_countyCluster = getCountyDf(df_nhtsagg, '17113')
    
    # Output the results of gaussian mixture model and the labels identified
    # to a local file for further analysis/debugging, if needed
    df_nhtsagg.to_csv('ward_output.csv', ',')

    # If pca is enabled, perform PCA on trainingData
    if (pcaFlag) :
        pca = PCA(n_components=pcaComp)
        pca.fit(X)
        X = pca.transform(X)

    # Calculate the efficiency of this clustering model using silhouette coefficient
    # if score is close to 1, then the clusters are dense and well separated - this is desired
    # if score is close to 0, then there are a number of overlapping clusters
    # if socre is close to -1, then there is incorrect clustering - undesired
    # euclidean distance is the chosen metric to measure cluster efficiency
    score = silhouette_score(X, y, metric='euclidean')
    
    # Return appropriate variables back to the calling API for further processing
    return df_countyCluster, score


# This module returns a data frame that has data only for the cluster to which
# countyid belongs to
def getCountyDf(df_nhtsagg, cid) :
    
    # For the given countyid (cid), find the cluster (label) it belongs to
    # Return a copy of the data frame of all counties that belong to the same label
    return df_nhtsagg[(df_nhtsagg.label == ((df_nhtsagg[(df_nhtsagg.countyid == cid)]['label']).values[0]))].copy()

    
def main():
    df_countyCluster, score = buildWardModel(
                                'county_and_nhts_inp_to_clustering.csv', 
                                type=1,
                                sep=','
                              )
    
if __name__ == "__main__":
  main()
  