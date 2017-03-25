import numpy as np
import pymc3
import scipy.stats

from darkopt.learning_curve import skeletons

default_map_options = {}

default_sample_options = {
    'draws': 1000,
}


def _single(x, y, curve, map_only, map_options, sample_options):
    if map_options is None:
        map_options = default_map_options
    if sample_options is None:
        sample_options = default_sample_options

    curve = skeletons.get_curve(curve)
    name, n_params, func = curve

    model_single = pymc3.Model()
    with model_single:
        params = pymc3.Flat(name, shape=n_params)
        mu = func(x, params)
        sd = pymc3.Uniform('sd', lower=1e-9, upper=1e-1)
        pymc3.Normal('y_obs', mu=mu, sd=sd, observed=y)

        map_estimate = pymc3.find_MAP(**map_options)
        if map_only:
            return map_estimate

        trace = pymc3.sample(start=map_estimate,
                             step=pymc3.Metropolis(), **sample_options)  #
        return trace


def map_single(x, y, curve, map_options):
    return _single(x, y, curve, True, map_options, None)


def sample_single(x, y, curve, map_options, sample_options):
    return _single(x, y, curve, False, map_options, sample_options)


def sample_ensemble(x, y, curves, map_options, sample_options):
    if map_options is None:
        map_options = default_map_options
    if sample_options is None:
        sample_options = default_sample_options

    curves = skeletons.get_curve_set(curves)
    map_estimates = {
        curve[0]: map_single(x, y, curve, map_options)
        for curve in curves
    }
    print(curves)
    print(map_estimates)
    start = {
        name: map_estimate[name]
        for name, map_estimate in map_estimates.items()
    }
    start['weights_unnormalized_interval_'] = np.zeros(len(curves))
    start['sd_interval_'] = 0

    model_ensemble = pymc3.Model()
    with model_ensemble:
        mu_single = []
        for name, n_params, func in curves:
            params = pymc3.Flat(name, shape=n_params)
            mu_single.append(func(x, params))
        weights_unnormalized = pymc3.Uniform(
            'weights_unnnormalized', lower=0, upper=1, shape=len(curves))
        weights_normalized = pymc3.Deterministic(
            'weights_normalized', weights_unnormalized / weights_unnormalized.sum())
        mu_ensemble = weights_normalized.dot(mu_single)
        sd = pymc3.Uniform('sd', lower=1e-9, upper=1e-1)
        pymc3.Deterministic('sd', sd)
        pymc3.Normal('y_obs', mu=mu_ensemble, observed=y, sd=sd)

    with model_ensemble:
        trace = pymc3.sample(
            start=start, step=pymc3.Metropolis(), **sample_options)

    return trace


def predict_single(x, curve, param):
    name, _, func = skeletons.get_curve(curve)
    return func(x, param[name])


def predict_ensemble(x, curves, param):
    curves = skeletons.get_curve_set(curves)
    ps = [predict_single(x, curve, param) for curve in curves]
    return param['weights_normalized'].dot(ps)


def _predict_proba_less_than(y, mu, param):
    sd = param['sd']
    cdf = scipy.stats.norm.cdf(y, loc=mu, scale=sd)
    return cdf


def predict_proba_less_than_single(x, y, curve, param):
    mu = predict_single(x, curve, param)
    return _predict_proba_less_than(y, mu, param)


def predict_proba_less_than_ensemble(x, y, curve, param):
    mu = predict_ensemble(x, curve, param)
    return _predict_proba_less_than(y, mu, param)
