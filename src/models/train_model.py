'''
To run this file correctly, run build_features.py in
the features directory.
'''
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from pickle import dump

## LOAD THE DATA
df_customers = pd.read_pickle('../../data/processed/01_built_features.pkl')

## SCALE THE DATA
scaler = StandardScaler()
scaled_customers = scaler.fit_transform(df_customers)

scaled_customers = pd.DataFrame(scaled_customers, 
                                columns=df_customers.columns)

## CLUSTERING

cluster_nums = [2,3,4,5,6,7]
scores = []
sum_of_squared_distances = []

for cluster_num in cluster_nums:
    kmeans = KMeans(cluster_num, random_state=0)
    kmeans.fit(scaled_customers)
    sum_of_squared_distances.append(kmeans.inertia_)
    clusters = kmeans.predict(scaled_customers)
    silhouette = silhouette_score(scaled_customers, clusters)
    scores.append(silhouette)

## PLOT RESULTS

# Silhouette score
sns.set_style('whitegrid')
plt.ylabel('Silhouette Score')
plt.xlabel('Clusters')
sns.lineplot(x=cluster_nums,y=scores)
plt.title('Silhouette Score by n clusters')
plt.savefig('../visualization/silhouette.png')

# 
plt.plot(cluster_nums,sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')
plt.savefig('../visualization/sum_of_square.png')

# SHOW SEGMENTS

kmeans_2groups = KMeans(2, random_state=0)
kmeans_3groups = KMeans(3, random_state=0)

kmeans_2groups.fit(scaled_customers)
kmeans_3groups.fit(scaled_customers)

# Analysing 2 customer groups
plt.figure(figsize=(15,3.5))
plt.title('Two clusters analysis')
fig = sns.heatmap(scaler.inverse_transform(kmeans_2groups.cluster_centers_),
            annot=True,
            yticklabels=['Cluster 1','Cluster 2'],
            xticklabels=scaled_customers.columns)
plt.savefig('../visualization/two_clusters.png')

# Analysing 3 customer groups
plt.figure(figsize=(15,3.5))
plt.title('Two clusters analysis')
sns.heatmap(scaler.inverse_transform(kmeans_3groups.cluster_centers_),
            annot=True,
            yticklabels=['Cluster 1','Cluster 2', 'Cluster 3'],
            xticklabels=scaled_customers.columns)
plt.savefig('../visualization/three_clusters.png')

# 3 clusters seems better, so we will save the three_clusters model

# SAVE 3 CLUSTERS MODEL
dump(kmeans_3groups, open('../../models/kmeans_3_segments.pickle', "wb"))