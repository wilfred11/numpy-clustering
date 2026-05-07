import numpy as np
import numpy.ma as ma

"""
This module contains functions for performing two different clustering algorithms on multivariate data sets. 

The different functions to be implemented are:
    - generate_normal_random_group 
    - generate_uniform_random_group
    - min_max_normalize
    - compute_distance_to_centers
    - compute_cluster_centers
    - clustering_initialisation_by_selection
    - clustering_initialisation_by_assignment
    - assign_data_to_cluster
    - intra_cluster_distance
    - inter_cluster_distance
    - leader_clustering
    - kmeans_clustering

For each function listed below, a docstring is provided explaining its purpose and usage, with type hints 
for the expected input and output. Each function can be implemented with Numpy arrays and methods.
Add your code in the indicated places after each function definition.

The main clustering functions and performance metrics implemented in this file are used in the script pdd_assignment_4.py.
In that script, you will find the main code that calls the clustering functions and computes the performance metrics 
on a given dataset.
"""

# random number generator
rng = np.random.default_rng(seed=20250819)


def generate_normal_random_group(center: np.ndarray, scale: np.ndarray, size: int) -> np.ndarray | None:
    """
    Generate a group of multivariate normal random vectors.

    This function generates a specified number of random vectors sampled
    from a multivariate normal distribution defined by the provided
    center (mean) and scale (standard deviation) parameters.

    :param center: The mean values for each dimension of the distribution, the shape is `(n_dim,)` or `(n_dim, 1)`
    :type center: np.ndarray
    :param scale: The standard deviations for each dimension of the distribution, the shape is `(n_dim,)` or `(n_dim, 1)`
    :type scale: np.ndarray
    :param size: The number of random vectors to generate
    :type size: int
    :return: A 2D array of shape `(size, n_dim)` where `n_dim` is the length of
        the `center`, containing the generated random vectors.
    :raises ValueError: If the `center` and `scale` arrays do not have the same shape.
    """
    # YOUR CODE STARTS HERE
    if np.equal(scale.shape, center.shape):
        x = np.random.normal(loc=center, scale=scale, size=(size, len(center)))
        return x
    else:
        raise ValueError('The provided center and scale arrays do not have the same shape.')
    # # YOUR CODE ENDS HERE


def generate_uniform_random_group(center: np.ndarray, scale: np.ndarray, size: int) -> np.ndarray | None:
    """
    Generates a group of uniformly distributed random points within a specified range defined
    by a center point and scale for each dimension.

    Each dimension's range is computed as [center - scale, center + scale].
    The size parameter determines the number of random points generated.

    :param center: Center point of the uniform distribution. It determines the midpoint
        of the range for each dimension (shape: (n_dim,)).
    :type center: np.ndarray
    :param scale: Scale for each dimension, which specifies the range of values
        around the center for each dimension (shape: (n_dim,)).
    :type scale: np.ndarray
    :param size: Number of random points to generate, defining the output size
        of the random group.
    :type size: int
    :return: A numpy array of shape (size, n_dim) containing the generated
        random points drawn from the uniform distribution.
    :rtype: np.ndarray
    :raises ValueError: If the center and scale arrays do not have the same shape.
    """
    # YOUR CODE STARTS HERE
    if np.equal(scale.shape, center.shape):
        x = np.random.uniform(low=center - scale, high=center + scale, size=(size, len(center)))
        return x
    else:
        raise ValueError('The provided center and scale arrays do not have the same shape.')
    # # YOUR CODE ENDS HERE


def min_max_normalize(data_set: np.ndarray) -> np.ndarray | None:
    """
    Performs Min-Max normalization on the input dataset to scale features to a range between 0 and 1.

    The transformation is applied independently to each feature (column) using the formula:
        x_scaled = (x - min) / (max - min)

    This function is essential for clustering algorithms, as it prevents features with larger numerical ranges
    from dominating the distance calculations.

    :param data_set: A 2-dimensional numpy array where each row represents a data point,
     and each column represents a feature.
    :type data_set: ndarray
    :return: A numpy array of the same shape as `data_set`, with values scaled between 0 and 1.
    :rtype: ndarray
    :raises ValueError: If the input array is empty, or if a feature has zero variance (min equals max).
    """
    # YOUR CODE STARTS HERE
    if data_set.size != 0:
        if np.min(data_set) != np.max(data_set):
            x_scaled = (data_set - np.min(data_set)) / np.max(data_set) - np.min(data_set)
            return x_scaled
    raise ValueError
    ## YOUR CODE ENDS HERE


def compute_distance_to_centers(data_points: np.ndarray, cluster_centers: np.ndarray) -> np.ndarray | None:
    """
    Computes the Euclidean distance from each data point to the provided cluster centers.

    The Euclidean distance between two points x and is computed as the square root of the sum of the
    squared differences of their coordinates [x1,x2,...,xn] and [y1,y2,...,yn]:

        distance = sqrt((x1 - y1)^2 + (x2 - y2)^2 + ... + (xn - yn)^2)

    This function calculates the scalar distance values between an array of center points
    and each data point in the given array. The computation is performed
    using the Euclidean distance formula in a vectorized manner.

    :param data_points: A 2-dimensional numpy array where each row represents the coordinates of a data point.
    :type  data_points: ndarray
    :param cluster_centers: A 2-dimensional numpy array representing the coordinates of the center points.
    :type  cluster_centers: ndarray
    :return: A 2-dimensional numpy array containing the Euclidean distances
        from each data point to the center points. The shape of the array is of the form [nbr_data_points, nbr_centers].
    :rtype: np.ndarray
    :raises ValueError: If the number of features in the center and data set differ
    """
    # YOUR CODE STARTS HERE
    if cluster_centers.shape[1] == cluster_centers.shape[1]:
        d=np.repeat(data_points, cluster_centers.shape[0], axis=0).reshape(data_points.shape[0],cluster_centers.shape[0],2)-cluster_centers
        distances=np.sqrt(d**2)
        return distances
    else:
        raise ValueError('The number of features in the center and data set elements differ')
    # # YOUR CODE ENDS HERE



def assign_data_to_cluster(data_set: np.ndarray, cluster_centers: np.ndarray, center_labels: np.ndarray = None) -> np.ndarray | None:
    """
    Assigns each data point in the data set to a cluster based on the provided cluster centers.

    This function calculates the closest cluster center for each data point in the
    data set. If weights are provided, a weighted distance computation is used;
    otherwise, a default distance calculation is applied. The function then
    assigns the data points to the clusters corresponding to the nearest centers.

    :param data_set: 2D array where each row represents a data point and each column represents a feature.
    :type data_set: np.ndarray
    :param cluster_centers: A NumPy array containing the coordinates of cluster
        centers, where each row corresponds to a cluster center in
        multidimensional space.
    :type cluster_centers: np.ndarray
    :param center_labels: cluster labels for each center, if not provided, the labels are from 0 to nbr_cluster-1
    :type center_labels: np.ndarray, optional
    :return: A NumPy array of integers where each entry represents the cluster ID
        to which the corresponding data point in the data set is assigned.
    :rtype: np.ndarray
    :raises ValueError: If the number of features in the center and data set differ
    """
    if data_set.shape[1] == cluster_centers.shape[1]:
        dataset_d=compute_distance_to_centers(data_set, cluster_centers)
        summed=np.sum(dataset_d,axis=2)
        minned=np.argmin(summed,axis=1)
        return minned
    else:
        raise ValueError('The number of features in the center and data set elements differ')
    # # YOUR CODE ENDS HERE


def compute_cluster_centers(data_set: np.ndarray, cluster_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Computes the cluster centers for the given dataset based on cluster assignments.

    This function calculates the mean position of points assigned to each cluster to
    determine the center of that cluster. The number of clusters is inferred from the
    different values in the `cluster_numbers` array.

    :param data_set: Input dataset, typically a 2D array where rows correspond to data points
        and columns correspond to features.
    :type data_set: np.ndarray
    :param cluster_ids: A 1D NumPy array where each element corresponds to a cluster
        assignment for the respective data point in data_set.
    :type cluster_ids: np.ndarray
    :return: A tuple containing the cluster centers and the respective
             cluster labels associated with each center (in order).
    :rtype: tuple[np.ndarray, np.ndarray]
    :raises ValueError: If the number of cluster assignments does not match the number of data points.
    """
    cluster_ids=cluster_ids[np.newaxis,:]
    data= np.concatenate((cluster_ids.T,data_set),axis=1)
    data = data[data[:, 0].argsort()]
    cluster_arrays = np.array_split(data[:, :], np.unique(data[:, 0], return_index=True)[1][1:])
    cluster_array_means =[]
    for clustered_points in cluster_arrays:
        cluster_array_means.append(np.mean(clustered_points[0:, 1:], axis=0))
    cluster_array_means=np.array(cluster_array_means)
    return cluster_array_means, np.unique(data[:, 0])



def clustering_initialisation_by_selection(data_set: np.ndarray, nbr_cluster: int) -> np.ndarray | None:
    """
    Selects and initializes cluster centroids from a given dataset randomly. The function checks
    that the number of clusters is not greater than the number of data points in the dataset.
    It then randomly selects unique data points from the dataset to serve as the initial centroids
    for clustering.

    :param data_set: Input dataset, typically a 2D array where rows correspond to data points
        and columns correspond to features.
    :type data_set: np.ndarray
    :param nbr_cluster: Number of clusters to initialize centroids for.
    :type nbr_cluster: int
    :return: A NumPy array containing randomly selected initial cluster centroids.
    :rtype: np.ndarray
    :raises ValueError: If the number of clusters is larger than the number of data points
    """
    if nbr_cluster <= data_set.shape[0]:
        index = np.random.choice(data_set.shape[0], nbr_cluster, replace=False)
        return data_set[index]
    else:
        raise ValueError("The number of clusters should not be larger than the number of data points in the dataset")


def clustering_initialisation_by_assignment(data_set: np.ndarray, nbr_cluster: int) -> np.ndarray | None:
    """
    Assigns each data point in the given dataset to an initial cluster. This function is used
    to initialize clusters randomly by assigning a cluster ID to each data point. The number
    of clusters must not exceed the number of data points in the dataset.

    :param data_set: Input dataset, typically a 2D array where rows correspond to data points
        and columns correspond to features.
    :type data_set: np.ndarray
    :param nbr_cluster: Number of distinct clusters to assign data points to. Value must be
        less than or equal to the number of data points in the dataset.
    :type nbr_cluster: int
    :return: A 1D numpy array with random cluster IDs assigned to data points. The length of
        the array matches the number of data points in the input dataset.
        Cluster IDs are from 1 till the requested number of clusters.
    :rtype: np.ndarray
    :raises ValueError: If the number of clusters is larger than the number of data points in
        the dataset.
    """
    if nbr_cluster <= data_set.shape[0]:
        ids=np.random.choice(np.arange(0, nbr_cluster), len(data_set), replace=True)
        return ids
    else:
        raise ValueError("The number of clusters should not be larger than the number of data points in the dataset")


def intra_cluster_distance(data_set: np.ndarray, cluster_ids: np.ndarray, cluster_centers: np.ndarray, cluster_labels: np.ndarray = None) -> np.float64 | None:
    """
    Calculate the average distance of each data point within a cluster to the centroid of the cluster.
    This function computes the intra-cluster distance for each cluster. It evaluates how closely data
    points in each cluster are grouped together. The smaller the intra-cluster distance, the more
    tightly packed the cluster is around its centroid.

    :param data_set: 2D array where each row represents a data point and each column represents a feature.
    :type data_set: np.ndarray
    :param cluster_ids: 1D array where each entry corresponds to the cluster ID assignment of the data point.
    :type cluster_ids: np.ndarray
    :param cluster_centers: 2D array where each row represents a cluster center
    :type cluster_centers: np.ndarray
    :param cluster_labels: 1D array where each entry corresponds to the cluster ID of the corresponding cluster center.
    :type cluster_labels: np.ndarray
    :return: A float representing the average intra-cluster distance over all clusters.
    :rtype: np.float64
    :raises ValueError: If the dimensions of data points, cluster labels and cluster centers are not compatible.
    """
    dataset_d = compute_distance_to_centers(data_set, cluster_centers)
    cluster_ids = cluster_ids[:, np.newaxis, np.newaxis]
    taken_data = np.take_along_axis(dataset_d[:, :], cluster_ids, axis=1)
    taken_data = np.squeeze(taken_data, axis=1)
    cluster_ids = np.squeeze(cluster_ids, axis=1)
    data = np.concatenate((cluster_ids, taken_data), axis=1)
    data = data[data[:, 0].argsort()]
    cluster_intra_distances_to_center = np.array_split(data[:, :], np.unique(data[:, 0], return_index=True)[1][1:])
    intra_cluster_distances_mean_list = []
    weights = []
    for clustered_distances in cluster_intra_distances_to_center:
        intra_cluster_distances_mean_list.append(np.mean(clustered_distances[0:, 1:], axis=0))
        weights.append(clustered_distances.shape[0])
    intra_cluster_distances_means = np.array(intra_cluster_distances_mean_list)
    weights = np.array(weights)[np.newaxis, :]
    m= np.sum(weights.T*intra_cluster_distances_means, axis=0)/np.sum(weights)
    return m

def inter_cluster_distance(data_set: np.ndarray, cluster_ids: np.ndarray, cluster_centers: np.ndarray) -> np.float64 | None:
    """
    Calculate the average distance of each data point to the next nearest cluster.

    This function computes the inter-cluster distance for each data point. It evaluates how distant the clusters are.
    The bigger the inter-cluster distance, the farther the data points are from each other.

    :param data_set: A 2D numpy array containing data points where each row
        represents a data sample and each column represents feature values.
    :type data_set: np.ndarray
    :param cluster_ids: A 1D numpy array specifying the cluster ID for each
        corresponding data point in the dataset.
    :type cluster_ids: np.ndarray
    :param cluster_centers: A 2D numpy array where each row represents the center
        coordinates of a cluster.
    :type cluster_centers: np.ndarray
    :return: A float representing the average inter-cluster distance over all data points.
    :rtype: np.float64
    """
    dataset_d = compute_distance_to_centers(data_set, cluster_centers)
    s= np.sum(dataset_d, axis=2)
    m=np.argpartition(s,1, axis=1)[:dataset_d.shape[0]]
    weights=np.sum(m, axis=0)
    mx = ma.masked_array(s, mask=np.logical_not(m))
    mean_dist_ = np.mean(mx,axis=0)
    weighted_d=np.sum((weights*mean_dist_))/np.sum(weights)
    return weighted_d

def leader_clustering(data_set: np.ndarray, threshold: np.float64) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Clusters a given dataset into groups using the leader clustering algorithm. The algorithm
    passes through each data point in the dataset once.
    The algorithm starts with the first data point as the initial cluster center.
    For each point it calculates the distance to the active set of cluster centers. It then
    assigns each data point to the closest cluster center if the distance is smaller than the threshold distance.
    Otherwise, it creates a new cluster center from the selected datapoint
    and assigns the data point to this new cluster center.

    The resulting cluster centroids are returned together with the cluster assignments for each data point.
    The cluster assignment corresponds to the index of the cluster center in the `cluster_centers` array.

    :param data_set: 2D array where each row represents a data point and each column represents a feature.
    :type data_set: np.ndarray
    :param threshold: A floating-point value representing the threshold distance for clustering.
    :type threshold: np.float64
    :return: A tuple containing the final cluster centers and the respective
             cluster assignments for data points in the input data set.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    cluster_centers=[]
    cluster_centers.append(data_set[0])
    cluster_centers_np = np.array(cluster_centers)
    go_on=True
    while go_on:
        distances=compute_distance_to_centers(data_set, cluster_centers_np)
        mean_of_distances_per_datapoint = np.mean(distances, axis=2)
        cluster_ids = assign_data_to_cluster(data_set, cluster_centers_np)
        data = np.concatenate((cluster_ids.T[:,np.newaxis],np.arange(0, data_set.shape[0]).T[:,np.newaxis], mean_of_distances_per_datapoint), axis=1)
        data = data[data[:, 0].argsort()]
        match_cluster_ind=data[:,0].astype(np.int32)+2
        match_cluster_ind=match_cluster_ind[:,np.newaxis]
        clustered_distances=np.take_along_axis(data, match_cluster_ind)
        dataset_indices=data[:, 1]
        max_index_clustered_distances = np.argmax(clustered_distances, 0)
        dataset_index_max_distance=dataset_indices[max_index_clustered_distances]
        dataset_index_max_distance=dataset_index_max_distance.astype(np.int32)
        max_distance = np.max(clustered_distances, 0)
        if max_distance[0]>=threshold:
            s=data_set[dataset_index_max_distance[0]]
            s=s[np.newaxis,:]
            cluster_centers_np = np.concatenate((cluster_centers_np, s), axis=0)
        else:
            go_on=False

    cluster_ids = assign_data_to_cluster(data_set, cluster_centers_np)
    return cluster_centers_np, cluster_ids

def kmeans_clustering(data_set: np.ndarray, nbr_clusters: int = 3, max_iterations: int = 100) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Clusters the given data set using the k-means algorithm.
    The algorithm starts with randomly selected data points as the initial cluster centers.
    Then it iteratively assigns each data point to the closest cluster
    and updates cluster centers based on the new assignments.
    This process is repeated until convergence or the maximum number of iterations is reached.

    Convergence is reached when the cluster assignment of all data points
    between two consecutive iterations is the same.

    The resulting cluster centroids are returned together with the cluster assignments for each data point.
    The cluster assignment corresponds to the index of the cluster center in the `cluster_centers` array.

    :param data_set: 2D array where each row represents a data point and each column represents a feature.
    :type data_set: np.ndarray
    :param nbr_clusters: The number of clusters to divide the data set into.
    :type nbr_clusters: int, optional
    :param max_iterations: The maximum number of iterations to perform.
    :type max_iterations: int, optional
    :return: A tuple containing the final cluster centers and the respective
             cluster assignments for data points in the input data set.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    cluster_centers = clustering_initialisation_by_selection(data_set, nbr_clusters)
    cluster_ids=assign_data_to_cluster(data_set, cluster_centers)
    previous_cluster_ids=cluster_ids.copy()
    for i in range(max_iterations):
        cluster_centers,_=compute_cluster_centers(data_set, cluster_ids)
        cluster_ids = assign_data_to_cluster(data_set, cluster_centers)
        if np.array_equal(cluster_ids, previous_cluster_ids):
            break
        else:
            previous_cluster_ids=cluster_ids.copy()
    return cluster_centers, cluster_ids


