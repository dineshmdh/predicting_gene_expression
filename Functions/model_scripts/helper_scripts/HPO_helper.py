import numpy as np
import hyperopt
from functools import partial
from hyperopt import hp

'''Define some basic HPO functions and variables
LAMDA_LOWER (below) should be > LAMDA_UPPER because of this:
https://github.com/hyperopt/hyperopt/issues/221
'''
LAMDA_BASE = 5  # used to define lamda in parameter space search
LAMDA_LOWER = 3  # lower limit in 10e-{} space to select lamda
LAMDA_UPPER = 6

LR_BASE = 7
LR_LOWER = 1  # in the power of 10**-1
LR_UPPER = 4


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
    h1_lower = int(round(0.33 * trainX.shape[1]))  # if just 4 fts, h1_lower = 1 (see below)
    h1_upper = int(round(0.66 * trainX.shape[1]))  # even if just 4 fts, h1_upper = 3
    parameter_space = {
        'layers': hp.choice('layers', [{
            'n_layers': 1,
            'n_units_layer': [
                uniform_int('n_units_layer_11', max(2, h1_lower), h1_upper),  # don't allow lower lim to be 1
            ],
        }, {
            'n_layers': 2,
            'n_units_layer': [
                uniform_int('n_units_layer_21', max(2, h1_lower), h1_upper),  # don't allow lower lim to be 1
                uniform_int('n_units_layer_22', max(2, int(round(0.66 * h1_lower))), max(3, int(round(0.66 * h1_upper)))),  # h2 layer will have 2-3 nodes; doing max(3,_) b/c if for eg. h1_upper = 3, upper limit has to be > lower limit
            ],
        }]),
        'lamda': LAMDA_BASE * 10**(-1 * uniform_int("lamda", LAMDA_LOWER, LAMDA_UPPER)),
        'starter_learning_rate': LR_BASE * 10**(-1 * uniform_int("starter_learning_rate", LR_LOWER, LR_UPPER)),
        'use_sigmoid_h1': hp.choice('use_sigmoid_h1', [True, False]),
        'use_sigmoid_h2': hp.choice('use_sigmoid_h2', [True, False]),
        'use_sigmoid_yhat': hp.choice('use_sigmoid_yhat', [True, False])
    }
    return parameter_space
