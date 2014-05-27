print(__doc__)

from time import time
import numpy as np
import pylab as pl
import pandas as pd

from sklearn import metrics
from sklearn.cluster import KMeans
#from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D
import sys

# use the full data set after development is complete with the smaller data set
# bank = pd.read_csv('bank-full.csv', sep = ';')  # start with smaller data set

# initial work with the smaller data set
county = pd.read_csv('county_and_nhts_inp_to_clustering.csv')  # start with smaller data set
# drop observations with missing data, if any
county = county.dropna()

# examine the shape of the DataFrame
#print(county.shape)

# look at the list of column names, note that y is the response
#list(county.columns.values)

# look at the beginning of the DataFrame
#county.head()

countyid = county['countyid']
pop10 = county['pop10']
land = county['aland_sqmi']
y = county['funding']

X = np.array([np.array(countyid), np.array(pop10), np.array(land)]).T
n_digits = 20

km = KMeans(init='k-means++', n_clusters=n_digits, max_iter = 100, n_init=10)
km.fit(X)

#labels1 = list(county.columns.values)
labels = km.labels_
cluster_centers = km.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print(79 * '_')

sample_size = 500

print("Clustering sparse data with %s" % km)
def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_, metric='euclidean', sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=X)
bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=X)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(X)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based", data=X)
print(79 * '_')

sys.stdout = open('clusters_KMean.csv', 'w')
for k in range(n_clusters_):
    my_members = labels == k
    print "cluster {0}: {1}".format(k, X[my_members, 0])
