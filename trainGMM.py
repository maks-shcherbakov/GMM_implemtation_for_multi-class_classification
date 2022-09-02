from plot_gaussians import *

def trainGMM(k, data):

    """
        :param data          : 2D array of our dataset
        :param k            : number of GMMs
    """
    # Write your code here
    # Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
    # Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
    # Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"

    # Train a GMM with k components, on the chosen phoneme

    # as dataset X, we will use only the samples of the chosen phoneme
    X = data.copy()
    # get number of samples
    N = X.shape[0]
    # get dimensionality of our dataset
    D = X.shape[1]

    # common practice : GMM weights initially set as 1/k
    p = np.ones(k) / k
    # GMM means are picked randomly from data samples
    random_indices = np.floor(N * np.random.rand(k))
    random_indices = random_indices.astype(int)
    mu = X[random_indices, :]  # shape kxD
    # covariance matrices
    s = np.zeros((k, D, D))  # shape kxDxD
    # number of iterations for the EM algorithm
    n_iter = 100

    # initialize covariances
    for i in range(k):
        cov_matrix = np.cov(X.transpose())
        # initially set to fraction of data covariance
        s[i, :, :] = cov_matrix / k

    return p,mu,s,n_iter,N,D