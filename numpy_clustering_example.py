import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from numpy_clustering_functions import *


"""
Main script to test the functions in numpy_clustering_example.py

The script will load the data sets and labels from the files and then perform the following steps:
1. Perform leader clustering on the data set with different distance thresholds
2. Perform k-means clustering on the data set with different number of clusters
3. Compute inter-cluster distances for the different clustering results
4. Compute the intra-cluster distances for the different clustering results
6. Select the best clustering result based on the performance characteristics
7. Use the selected clustering to assign labels to the test data set

!!!!
  It is not necessary to change anything in this script
!!!!

"""

if __name__ == "__main__":
    # define the data file
    data_set_1_file = "../../../../data/module_4/cluster_data_set_2d.csv"
    data_labels_1_file = "../../../../data/module_4/cluster_data_set_2d_labels.csv"
    test_data_set_1_file = "../../../../data/module_4/cluster_test_data_set_2d.csv"
    test_data_labels_1_file = "../../../../data/module_4/cluster_test_data_set_2d_labels.csv"
    if (not os.path.exists(data_set_1_file) or not os.path.exists(data_labels_1_file) or
            not os.path.exists(test_data_set_1_file) or not os.path.exists(test_data_labels_1_file)):
        print(f"Data files not found. Are you working in the correct directory?")
        exit(1)

    # load the data sets and corresponding labels into numpy arrays
    # the variables to store the data are already defined,
    # make sure to assign the loaded data to these variables as they are used in the remainder of the script
    # be aware that class labels are not necessarily 0-based
    data_set_1 = np.loadtxt(data_set_1_file, delimiter=',', skiprows=1)
    data_labels_1 = np.loadtxt(data_labels_1_file, delimiter=',', skiprows=1, )
    test_data_set_1 = np.loadtxt(test_data_set_1_file, delimiter=',', skiprows=1)
    test_data_labels_1 = np.loadtxt(test_data_labels_1_file, delimiter=',', skiprows=1)

    if data_set_1 is None or data_labels_1 is None:
        print("Data sets not loaded. Skipping leader clustering clustering steps.")
        exit(101)

    # Perform leader clustering
    cluster_results = []
    try:
        for threshold in np.array([0.8, 1.4, 2.0, 3.0, 4.0]):
            print(f"Clustering with threshold {threshold}")
            c1, l1 = leader_clustering(data_set_1, threshold)
            print(f"Clustering with threshold {threshold} completed")
            print(f"Number of clusters: {len(np.unique(l1))}")
            r = {'type': 'leader', 'parameter': threshold, 'clusters': c1.copy(), 'labels': l1.copy()}
            cluster_results.append(r)
    except Exception as e:
        print("Unable to perform leader clustering", e)
        exit(101)

    # perform k-means clustering
    try:
        for n_clusters in np.array([2, 3, 4, 5, 6]):
            print(f"Clustering with {n_clusters} clusters")
            c1, l1 = kmeans_clustering(data_set_1, n_clusters)
            print(f"Clustering with {n_clusters} clusters completed")
            print(f"Number of clusters: {len(np.unique(l1))}")
            r = {'type': 'k-means', 'parameter': n_clusters, 'clusters': c1.copy(), 'labels': l1.copy()}
            cluster_results.append(r)
    except Exception as e:
        print("Unable to perform k-means clustering", e)
        exit(101)

    # compute for each clustering the performance measures and store those as additional fields in the results dictionary
    #  'intra': result from intra_cluster_distance function
    #  'inter': result from inter_cluster_distance function
    #
    for c in cluster_results:
        c['intra'] = intra_cluster_distance(data_set_1, c['labels'], c['clusters'])
        c['inter'] = inter_cluster_distance(data_set_1, c['labels'], c['clusters'])

    #
    # Check the performance scores
    #
    performance_ready = True if len(cluster_results) == 10 else False
    for c in cluster_results:
        if 'intra' not in c.keys() or 'inter' not in c.keys():
            print(f"The clustering result {c['type']}/{c['parameter']} does not contain the expected fields.")
            performance_ready = False
        else:
            print(f"The clustering result {c['type']}/{c['parameter']} - intra = {c['intra']} - inter = {c['inter']}")

    if not performance_ready:
        print("Not all clustering results contain the expected fields. Please check the implementation.")
        exit(1)

    # Select one of the clustering results based on the performance measures
    # and store this in the `selected_cluster` variable
    #
    # Use the centers from this cluster to assign cluster labels to the test data set
    # and to store the results and parameters in the `test_cluster_result` dictionary.
    #   - compute the confusion matrix for the assigned labels of the test data set
    best_idx = 6
    selected_cluster = cluster_results[best_idx]
    test_cluster_result = {
        'type': selected_cluster['type'],
        'parameter': selected_cluster['parameter'],
        'clusters': selected_cluster['clusters'].copy()
    }
    test_cluster_result['labels'] = assign_data_to_cluster(test_data_set_1, selected_cluster['clusters'])
    test_cluster_result['intra'] = intra_cluster_distance(test_data_set_1, test_cluster_result['labels'], selected_cluster['clusters'])
    test_cluster_result['inter'] = inter_cluster_distance(test_data_set_1, test_cluster_result['labels'], selected_cluster['clusters'])


    #
    # FINAL results
    # visualisation of the data and clustering results
    #
    if selected_cluster is None or test_cluster_result is None:
        print("No selected clustering result found. Skipping visualisation steps.")
    else:
        print(f"Selected clustering result: {test_cluster_result}")

        df_centers = pd.DataFrame(selected_cluster['clusters'], columns=['x','y'], index=[0,1,2])
        df_train = pd.concat([pd.DataFrame(data_set_1, columns=['x','y']), pd.DataFrame(data_labels_1, columns=['cluster'])], axis=1)
        df_clustered = pd.concat([pd.DataFrame(data_set_1, columns=['x','y']), pd.DataFrame(selected_cluster['labels'], columns=['cluster'])], axis=1)
        sns.scatterplot(data=df_train, x='x', y='y', hue='cluster' , palette="tab10", alpha=0.4)
        sns.scatterplot(data=df_clustered, x='x', y='y', hue='cluster' , palette="tab10", marker='x')
        sns.scatterplot(data=df_centers, x='x', y='y', c="k", marker='^')
        plt.title(f"Clustering data set with {selected_cluster['type']}/{selected_cluster['parameter']}")
        plt.show(block=True)

        df_test_clustered = pd.concat([pd.DataFrame(test_data_set_1, columns=['x','y']), pd.DataFrame(test_cluster_result['labels'], columns=['cluster'])], axis=1)
        df_test = pd.concat([pd.DataFrame(test_data_set_1, columns=['x','y']), pd.DataFrame(test_data_labels_1, columns=['cluster'])], axis=1)
        sns.scatterplot(data=df_test, x='x', y='y', hue='cluster' , palette="tab10", alpha=0.4)
        sns.scatterplot(data=df_test_clustered, x='x', y='y', hue='cluster' , palette="tab10", marker='x')
        sns.scatterplot(data=df_centers, x='x', y='y', c="k", marker='^')
        plt.title(f"Clustering test set with {selected_cluster['type']}/{selected_cluster['parameter']}")
        plt.show(block=True)
