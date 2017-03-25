import numpy as np


curves_skeletons = {
    'vapore_pressure':
        (3, lambda x, p: p[0] * np.exp(p[1] / (1 + x) + p[2] * np.log1p(x))),
    'weibull':
        (3, lambda x, p: p[0] - p[1] * np.exp(-p[2] * x)),
}


curve_sets = {
    'all': curves_skeletons.keys()
}


def get_curve(curve):
    if isinstance(curve, str):
        return (curve, *curves_skeletons[curve])
    else:
        return curve


def get_curve_set(curves):
    if isinstance(curves, str):
        curves = curve_sets[curves]
    return [get_curve(curve) for curve in curves]
