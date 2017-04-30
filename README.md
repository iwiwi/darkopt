# Darkopt

[![Documentation Status](https://readthedocs.org/projects/darkopt/badge/?version=latest)](http://darkopt.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


*Darkopt* is a Python library that implements the *gray-box approach* for automatic hyper-parameter optimization of machine learning algorithms.

Existing hyper-parameter optimization libraries rely on the black-box approach, i.e., they assume almost nothing about ML algorithms and use only trial outcomes. In contrast, Darkopt achieves significantly better performance by employing the novel gray-box approach. By focusing on iterative ML algorithms, it exploits learning curves to predict the outcomes, and terminate hopeless trials earlier. It is applicable for a wide range of ML algorithms including GBDT (e.g., XGBoost) and deep learning (e.g., Chainer).


## Modules

Darkopt's main components are:

* **learning_curve** ---learning curve predictors using PyMC3,
* **integration** --- callbacks for famous ML libraries (e.g., XGBoost and Chainer) to introduce pruning, and
* **optimize** --- basic hyper-parameter search engines.


## Use Cases

We suppose the following two use cases:

1. Using Darkopt's built-in basic optimizers, or
2. Combining Darkopt's learning curve predictors with Bayesian optimization libraries (e.g., hyperopt).

The former is easier, but the latter may perform better.


## License

[MIT License](LICENSE)


## References

1. T. Domhan, et al. **Speeding Up Automatic Hyperparameter Optimization of Deep Neural Networks by Extrapolation of Learning Curves.** In *IJCAI 2015.*
2. A. Klein, et al. **Learning Curve Prediction with Bayesian Neural Networks.** In *ICLR' 17.*
