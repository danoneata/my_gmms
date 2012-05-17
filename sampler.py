from numpy.random import dirichlet


def gibbs_sampler(xx, nr_iterations=100):
    """ Gibbs sampler for the finite Bayesian Gaussian mixture model.

    Inputs
    ------
    xx: array [nr_points, nr_features]
        Data matrix.

    nr_iterations: int, default 100
        Number of iterations the Gibbs sampler is run.

    Output
    ------
    pi: array [nr_clusters]
        Mixing weights vector.

    zz: array [nr_points]
        Cluster assignments for each point.

    mu: array [nr_clusters, nr_features]
        Mean vectors for each of cluster.

    sigma: array [nr_clusters, nr_features]
        Variances for each cluster. At the moment, assuming diagonal matrices.

    """
    # TODO Use burn-out period.
    N, D = xx.shape  # nr_points, nr_features
    for ii in xrange(nr_iterations):
        # TODO Thin the samples to avoid correlation.
        # Sample mixing weights.
        pi = sample_pi(alpha, zz)
        for nn in xrange(N):
            # Sample latent z_n.
            zz[nn] = sample_z
        for kk in xrange(K):
            # Sample Gaussian parameters, mean and variance.
            mu[kk], sigma[kk] = sample_mu_sigma()
        sample = [pi, zz, mu, sigma]
        yield sample


def sample_pi(alpha, zz):
    """ Sample mixing weights from the posterior distribution p(\pi|Z). 

    Inputs
    ------
    alpha: array [nr_clusters]
        The parameters of the prior of the mixing weights Dirichlet(\pi|alpha).

    zz: array [nr_points]
        The observer assignments of each of the N points.

    Output
    ------
    pi: array [nr_clusters]
        Sample for mixing weights.

    """
    K = len(alpha)
    # Count how many times appears each state.
    counts = np.zeros(K)
    for state in zz:
        counts[state] += 1
    alpha_new = np.array(alpha) + counts
    pi = dirichlet(alpha_new)
    return pi
    

def sample_z():
    pass


def sample_mu_sigma():
    pass
