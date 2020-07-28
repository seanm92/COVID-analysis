from dtw import *
import pandas as pd
import numpy as np
import itertools
from sklearn.cluster import AgglomerativeClustering 
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

working_path = r'C:\Users\97254\Desktop\Covid_19'
def dist(a,b):
    return np.linalg.norm(a-b)

covid = pd.read_csv(working_path + r'\covid_19_data.csv')
infected_countries = pd.read_csv(working_path + r'\infected_countries.csv')
# --- this file was not provided by kaggle but rather manually collected from wikipedia --- #

covid = pd.merge(left=covid, right=infected_countries)
covid['prop_confirmned'] = covid['Confirmed']/covid['Population'].str.replace(',', '').astype(int)
covid['prop_deaths'] = covid['Deaths']/covid['Population'].str.replace(',', '').astype(int)
covid['prop_recovered'] = covid['Recovered']/covid['Population'].str.replace(',', '').astype(int)
covid = covid.groupby(['Country/Region', 'ObservationDate'],as_index = False).sum()

contries = covid['Country/Region'].unique()
countries_to_inds = {a: i for i, a in enumerate(contries)}

combs = list(itertools.combinations(contries, 2))
Dist_matrix = np.zeros((len(contries),len(contries)))
for a,b in combs:
    ind1 = countries_to_inds[a]
    ind2 = countries_to_inds[b]
    M1 = np.asarray(covid[(covid['Country/Region'] == a)]['prop_deaths'])
    M2 = np.asarray(covid[(covid['Country/Region'] == b)]['prop_deaths'])
    Dist_matrix[ind1, ind2] = dtw(M1, M2, dist=dist)[0]


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def get_distances(X, model, mode='l2'):
    distances = []
    weights = []
    children = model.children_
    dims = (X.shape[1], 1)
    distCache = {}
    weightCache = {}
    for childs in children:
        c1 = X[childs[0]].reshape(dims)
        c2 = X[childs[1]].reshape(dims)
        c1Dist = 0
        c1W = 1
        c2Dist = 0
        c2W = 1
        if childs[0] in distCache.keys():
            c1Dist = distCache[childs[0]]
            c1W = weightCache[childs[0]]
        if childs[1] in distCache.keys():
            c2Dist = distCache[childs[1]]
            c2W = weightCache[childs[1]]
        d = np.linalg.norm(c1-c2)
        cc = ((c1W*c1)+(c2W*c2))/(c1W+c2W)

        X = np.vstack((X, cc.T))

        newChild_id = X.shape[0]-1

        # How to deal with a higher level cluster merge with lower distance:
        if mode == 'l2':  # Increase the higher level cluster size suing an l2 norm
            added_dist = (c1Dist**2+c2Dist**2)**0.5 
            dNew = (d**2 + added_dist**2)**0.5
        elif mode == 'max':  # If the previrous clusters had higher distance, use that one
            dNew = max(d, c1Dist, c2Dist)
        elif mode == 'actual':  # Plot the actual distance.
            dNew = d


        wNew = (c1W + c2W)
        distCache[newChild_id] = dNew
        weightCache[newChild_id] = wNew

        distances.append(dNew)
        weights.append(wNew)
    return distances, weights


# ---  plot dendogram --- #
model = AgglomerativeClustering()
model.fit(Dist_matrix)
distance, weight = get_distances(Dist_matrix, model)
linkage_matrix = np.column_stack([model.children_, distance, weight]).astype(float)
plt.figure(figsize=(20, 10))
dendrogram(linkage_matrix)
plt.show()

n_clusters = 8
# --- chosen by the dendogram --- #
hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
clusters = hc.fit_predict(Dist_matrix)
First_cluster = [country for ind, country in enumerate(contries) if clusters[ind] == 0]
Second_cluster = [country for ind, country in enumerate(contries) if clusters[ind] == 1]
Third_cluster = [country for ind, country in enumerate(contries) if clusters[ind] == 2]
Fourth_cluster = [country for ind, country in enumerate(contries) if clusters[ind] == 3]
Fifth_cluster = [country for ind, country in enumerate(contries) if clusters[ind] == 4]
Sixth_cluster = [country for ind, country in enumerate(contries) if clusters[ind] == 5]
Seventh_cluster = [country for ind, country in enumerate(contries) if clusters[ind] == 6]
eigth_cluster = [country for ind, country in enumerate(contries) if clusters[ind] == 7]

covid['Date'] = pd.to_datetime(covid['ObservationDate'])
covid = covid.sort_values('Date')
dates = covid['ObservationDate'].unique()


def means_stds_over_time_covid(cluster_contries, dates, covid_data, column):
    subset_covid_data = covid_data[covid_data['Country/Region'].isin(cluster_contries)]
    cluster_means = np.array([np.mean(subset_covid_data[subset_covid_data['ObservationDate'] == date][column]) for date in dates])
    cluster_stds = np.array([np.std(subset_covid_data[subset_covid_data['ObservationDate'] == date][column]) for date in dates])
    path_deviation = 2 * cluster_stds
    under_line = (cluster_means-path_deviation)
    over_line = (cluster_means+path_deviation)
    return cluster_means,  under_line, over_line


cluster_means1, under_line1, over_line1 = means_stds_over_time_covid(First_cluster, dates, covid, 'prop_deaths')
cluster_means2, under_line2, over_line2 = means_stds_over_time_covid(Second_cluster, dates, covid, 'prop_deaths')
cluster_means3, under_line3, over_line3 = means_stds_over_time_covid(Third_cluster, dates, covid, 'prop_deaths')
cluster_means4, under_line4, over_line4 = means_stds_over_time_covid(Fourth_cluster, dates,covid, 'prop_deaths')
cluster_means5, under_line5, over_line5 = means_stds_over_time_covid(Fifth_cluster, dates, covid, 'prop_deaths')
cluster_means6, under_line6, over_line6 = means_stds_over_time_covid(Sixth_cluster, dates, covid,'prop_deaths')
cluster_means7, under_line7, over_line7 = means_stds_over_time_covid(Seventh_cluster, dates, covid, 'prop_deaths')
cluster_means8, under_line8, over_line8 = means_stds_over_time_covid(eigth_cluster, dates, covid, 'prop_deaths')

# --- for visualization purpose I didn't use the 7-8 clusters, and discard Stds --- #

dates_range = list(range(len(dates)))
plt.plot(dates_range, cluster_means1)
plt.plot(dates_range, cluster_means2)
plt.plot(dates_range, cluster_means3)
plt.plot(dates_range, cluster_means4)
plt.plot(dates_range, cluster_means5)
plt.plot(dates_range, cluster_means6)
plt.legend(['Austria, Denmark, Germany...', 'Belgium, France', 'US, UK, Sweden...',
    'Spain', 'Israel, Chile,Finland...', 'Italy'], loc='upper left')
plt.xlabel('Date')
plt.ylabel('Deaths / Population')
plt.title('Clustering by Mortality Rate')



