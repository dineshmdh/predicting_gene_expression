--hidden_size

parser.add_argument("-l", "--take_log2_tpm", help="(Default: True)", type=bool, default=True)

put and hidden nodes are set to the correlation values between the node feature and the expression of the gene, and the initial weights between hidden layers or the hidden layer and output is set to 0.5


parser.add_argument("-ft", "--frac_test", help="(Default:0.2)", default=0.2, type=float)
parser.add_argument("-lr", "--learning_rates", help="", nargs="+", default=[0.05], type=float)


--slurmrank == --run_id

self.tf_corr_threshold_high = tf_corr_threshold_high
self.tf_corr_threshold_low = tf_corr_threshold_low

# update weight parameters
self.use_random_wts = False
self.use_corr_wts = False
self.use_expDecay_wts = False

if (use_wts == "corr"):
    self.use_corr_wts = True
elif (use_wts == "random"):
    self.use_random_wts = True
elif (use_wts == "expDecay"):
    self.use_expDecay_wts = True
else:
    raise Exception("For parameter 'use_wts', choose one from 'corr', 'random' or 'expDecay'..")
assert sum([self.use_random_wts, self.use_corr_wts, self.use_expDecay_wts]) == 1


# want a file w/ these info:
- gene loc
- gene name ensemble and gene symbol
- tad info
- list of known enhancers for this gene
- Expression across cell types

# want another similar file w/ these info:
- dhs loc
- accessibility across cell types
