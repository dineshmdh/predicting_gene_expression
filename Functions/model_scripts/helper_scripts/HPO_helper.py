import numpy as np
import hyperopt
from functools import partial
from hyperopt import hp

'''Define some basic HPO functions and variables'''
LAMDA_BASE = 3  # used to define lamda in parameter space search
LAMDA_LOWER = 1  # lower limit in 10e-{} space to select lamda
LAMDA_UPPER = 7


def uniform_int(name, lower, upper):
    # `quniform` returns: round(uniform(low, high) / q) * q
    return hp.quniform(name, lower, upper, q=1)


def loguniform_int(name, lower, upper):
    # Do not forget to make a logarithm for the lower and upper bounds.
    return hp.qloguniform(name, np.log(lower), np.log(upper), q=1)


tpe_method = partial(
    hyperopt.tpe.suggest,
    n_EI_candidates=200,  # 30 # Sample __ candidate and select candidate that has highest Expected Improvement (EI)
    gamma=0.2,  # Use 20% of best observations to estimate next set of parameters
    n_startup_jobs=5,  # 10 # First __ trials are going to be random
)


def get_parameter_space_forHPO(trainX):
    h1_lower = int(0.33 * trainX.shape[1])
    h1_upper = int(0.66 * trainX.shape[1])
    parameter_space = {
        'layers': hp.choice('layers', [{
            'n_layers': 1,
            'n_units_layer': [
                uniform_int('n_units_layer_11', h1_lower, h1_upper),
            ],
        }, {
            'n_layers': 2,
            'n_units_layer': [
                uniform_int('n_units_layer_21', max(2, h1_lower), max(2, h1_upper)),
                uniform_int('n_units_layer_22', max(1, int(0.66 * h1_lower)), max(int(0.66 * h1_upper), 1)),
            ],
        }]),
        'lamda': LAMDA_BASE * 10**(-1 * uniform_int("lamda", LAMDA_LOWER, LAMDA_UPPER))
    }
    return parameter_space
