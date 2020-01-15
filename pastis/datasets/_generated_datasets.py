# -*- coding: utf-8 -*-
import warnings
import numpy as np
from iced.utils import get_intra_mask
from sklearn.utils import check_random_state


__all__ = ["generate_dataset_from_distances"]

_DISTRIBUTIONS = ["Poisson", "NB", "NegativeBinomial", "Intensity"]


def generate_dataset_from_distances(dis, alpha=-3, beta=1,
                                    alpha_inter=None, lengths=None,
                                    distribution="NegativeBinomial",
                                    random_state=None,
                                    dispersion=7):
    """
    Generate dataset from distance matrix

    Parameters
    ----------
    dis : (n, n) ndarray

    alpha : float, optional, default: -3
        count-to-distance parameter

    beta : float, optional, default: 1
        coverage or scaling factor

    alpha_inter : float, optional, default: None
        count-to-distance parameter for inter-chromosomal count.
        When provided, lengths also needs to be provided

    lengths : ndarray (L, ), optional, default: None
        Vector of lengths of chromosomes.

    distribution : string, optional, default: "NegativeBinomial"
        The distribution used to draw contact counts from. Can be "Poisson",
        "NB", "NegativeBinomial", or "Intensity".
        If "Intensity" is provided, returns the intensity of the random
        process instead of a random distribution.

    random_state : int, optional, default: None
        Determines random number generation. Use an int to make the randomness
        deterministic.

    dispersion : float, optional, default: 7.
        Dispersion parameter for the Negative Binomial distribution.
        Will be ignored for the Poisson distribution.

    Returns
    -------
    ndarray (n, n)
    """
    if distribution not in ["Poisson", "NB", "NegativeBinomial"]:
        raise ValueError(
            "Unknown distribution %s. Possibile distributions are %s" %
            (distribution, ", ".join(_DISTRIBUTIONS)))
    if lengths is not None and dis.shape[0] != lengths.sum():
        raise ValueError("lengths and dis are of incompatible shapes")

    if alpha_inter is not None and lengths is None:
        raise ValueError(
            "When alpha_inter is provided, lengths also needs to be provided")

    random_state = check_random_state(random_state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        intensity = beta * dis ** alpha

    if alpha_inter is not None:
        inter_mask = ~get_intra_mask(lengths)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            intensity[inter_mask] = beta * dis[inter_mask] ** alpha

    intensity[np.isinf(intensity)] = 0
    if distribution in ["NB", "NegativeBinomial"]:
        if hasattr(dispersion, "predict"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = beta * dispersion.predict(dis ** alpha)
        else:
            d = beta * dispersion

        p = intensity / (intensity + d)
        counts = random_state.negative_binomial(
            d, 1 - p)
    elif distribution == "Poisson":
        counts = random_state.poisson(intensity)
    elif distribution == "Intensity":
        counts = intensity

    counts = (counts + counts.T)

    return counts
