import numpy as np
from scipy import sparse
from scipy import optimize
from . import utils


def compute_mean_variance(counts, lengths, use_zero_counts=True,
                          f=2./3,
                          bias=None):
    """
    Estimates the mean and variance at fixed genomic distances

    Parameters
    ----------
    counts : (n, n) ndarray or sparse array
        the raw contact count matrix

    lengths : (l,) ndarray
        the lengths of each chromosome

    user_zero_counts : boolean, default: True
        Whether to use 0 counts or not in the estimation of the mean and
        variance.

    f : float, default : 2./3
        The proportion of genomic distances to compute the mean / variance
        for.

    bias : (n, ) ndarray
        The bias vector. Not that the estimation of the mean and variance
        directly on the contact count is biased and needs to be corrected.

    Returns
    -------
    (gdis, mean, variances, num_data_points)
        Returns a tuple containing four (n, ) ndarrays. The first contains the
        genomic distance of the estimation. The second contains the estimation
        of the mean. The third contains the unbiased estimation of the
        variance. The fourth contains the number of data points used in the
        estimation. Note that the fourth vector can be deduced from the first.
    """
    if sparse.issparse(counts) and not sparse.isspmatrix_coo(counts):
        counts = counts.tocoo()
    gdis = utils.get_genomic_distances(lengths, counts)
    mean, var, num_data_points = [], [], []
    min_size = lengths.max() * f

    if bias is None:
        bias = np.ones((counts.shape[0], 1))

    if sparse.isspmatrix_coo(counts):
        mean, var, num_data_points = _compute_unbiased_mean_variance_sparse(
            counts, gdis, use_zero_counts, min_size, lengths, bias)
    else:
        gdis = np.triu(gdis)
        bias = bias.reshape(-1, 1)
        for i in range(1, int(min_size)):
            # FIXME this should deal natively with biases NAN
            counts_ = counts[gdis == i]
            b = (bias.T * bias)[gdis == i]
            b = b[np.invert(np.isnan(counts_))]
            counts_ = counts_[np.invert(np.isnan(counts_))]
            if not use_zero_counts:
                counts_ = counts_[counts_ != 0]
            mean_ = (counts_ / b).mean()
            mean.append(mean_)
            num_data_points.append(len(counts_))

            v = 1. / (len(counts_) - 1) * ((counts_ / b - mean_)**2).sum()
            var.append(v / (1. / b).mean())

    # Returns a tuple of
    return (np.arange(1, int(min_size)),
            np.array(mean), np.array(var),
            np.array(num_data_points))


def _get_indices_genomic_distances(lengths, d):
    row, col = [], []
    begin, end = 0, 0
    for l in lengths:
        end += l
        row += range(begin, end - d)
        col += range(begin + d, end)
        begin = end
    return row, col


def _compute_unbiased_mean_variance_sparse(counts, gdis,
                                           use_zero_counts, min_size,
                                           lengths, bias):
    bias = bias.flatten()
    var, mean, num_data_points = [], [], []
    for i in range(1, int(min_size)):
        counts_ = counts.data[gdis == i]
        bias_ = bias[counts.row[gdis == i]] * bias[counts.col[gdis == i]]

        if not use_zero_counts:
            assert np.all(counts.data != 0)

            # Compute the mean on normalized data
            mean_ = np.nanmean(counts_ / bias_)

            var_ = (
                1. / (len(counts_)-1) * np.nansum(
                    (counts_ / bias_ - mean_)**2))
            var_ /= np.nanmean(1. / bias_)
            mean.append(mean_)
            var.append(var_)
            num_data_points.append(len(counts_))
        else:
            # FIXME deal with the matrix is symmetric or upper sup.
            # Except the code only works for upper sup matrix

            t = lengths - i
            # TODO check that this is correct...
            mean_ = np.nansum(counts_ / bias_) / (t[t > 0].sum())
            mean.append(mean_)

            number_ = (t[t > 0]).sum() - len(counts_)

            v = (np.nansum((counts_ / bias_ - mean_) ** 2) +
                 np.nansum((mean_ * np.ones(number_)) ** 2))
            v /= (t[t > 0].sum() - 1)

            # Correct for the bias with all the biases of that genomic
            # distance.
            row, col = _get_indices_genomic_distances(lengths, i)
            v /= np.nanmean(1. * (bias[row] * bias[col]))

            var.append(v)
            num_data_points.append(t[t > 0].sum())
    return mean, var, num_data_points


def _estimate_exponential_pol(mean, variance, degree=2, sample_weights=None,
                              loss="linear"):
    """
    """
    # Remove all points with mean == 0 or variance == 0
    mask = ~((mean == 0) | (variance == 0))

    # Stabilization
    variance += 1e-8 * np.random.rand(variance.shape[0])
    dispersion = mean[mask] ** 2 / (variance[mask] - mean[mask])
    mean = mean[mask]
    if sample_weights is not None:
        sample_weights = sample_weights[mask]
    else:
        sample_weights = np.ones(mean.shape)

    def eval_f(coef):
        f = sum([coef[i] * np.log(mean) ** (i)
                 for i in np.arange(degree+1)])
        return sample_weights * (np.exp(f) - dispersion)

    optimization_results = optimize.least_squares(
        eval_f, np.zeros(degree+1), loss=loss)
    coefficients = optimization_results["x"]
    return coefficients


class ExponentialDispersion(object):
    """
    """

    def __init__(self, degree=2):
        """
        """
        self.degree = degree

    def fit(self, mean, variance, sample_weights=None):
        """
        Fits the Dispersion Parameter
        """
        self.coef_ = _estimate_exponential_pol(
            mean, variance,
            degree=self.degree,
            sample_weights=sample_weights)
        return self

    def predict(self, mean, bias=None):
        log_dispersion = np.zeros(mean.shape)
        for i in np.arange(self.degree + 1):
            log_dispersion += self.coef_[i] * np.log(mean) ** (i)

        # Remove degenerarcies
        dispersion = np.exp(log_dispersion)
        if bias is not None:
            dispersion *= bias
        return dispersion

    def derivate(self, mean, bias=None):
        """
        Derivates the function at the points mean
        """
        if self.degree == 0:
            return np.zeros(mean.shape)
        mean_shape = mean.shape
        mean = mean.flatten()
        dispersion = self.predict(mean)
        derivatives = (
            self.coef_[:, np.newaxis] / mean *
            np.arange(self.degree+1)[:, np.newaxis] *
            np.log(mean) ** (np.arange(self.degree+1) - 1)[:, np.newaxis]) *\
            dispersion
        derivatives = np.sum(derivatives, axis=0)
        if bias is not None:
            derivatives *= bias.flatten()
        return derivatives.reshape(mean_shape)
