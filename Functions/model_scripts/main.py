'''Created on Jan 1, 2018

__author__ = "Dinesh Manandhar"


Notes:
1. Right now, users cannot
    i) manipulate self.pcc_lowerlimit_to_filter_tfs and self.log2_tpm variables.
    ii) choose validation group - it is fixed as 'ENCODE2012'


'''
import logging
import re
import time
import os  # os is being used to set up default outputDir
import argparse
import sys
import hyperopt
from hyperopt import fmin

start_time = time.time()
sys.path = sys.path[1:]
sys.path.insert(0, os.path.join(os.getcwd(), "helper_scripts"))

from global_variables_final_for_git import Global_Vars
from prep_for_model_for_git import Model_preparation
from tensorflow_model_for_git import Tensorflow_model
import HPO_helper
from HPO_helper import uniform_int, loguniform_int, tpe_method, get_parameter_space_forHPO

############################################################
# ####### Set up the parser arguments ###### #
############################################################
parser = argparse.ArgumentParser(description="Using Neural Net, model the expression of genes using (1) DHS sites around the TSS, (2) TFs predicted to regulate the expression of the gene, or (3) both.")
parser.add_argument("gene", help="Gene of interest", type=str)

# ============= Arguments pertaining to DHS sites ===========
parser.add_argument("-d", "--distance", help="Distance from TSS (in kb) to cover as region of interest (Default: 150)", type=int, default=150)
parser.add_argument("-u", "--use_tad_info", help="Use TAD boundaries to demarcate the boundaries for the region of interest. (Default: True)", type=bool, default=True)
parser.add_argument("-dl", "--pcc_lowerlimit_to_filter_dhss", help="Lower limit of the absolute PCC(dhs site and gene expression) to be used in filtering top dhs sites. All DHS sites with pcc below this threshold are ignored. This option only applies for selecting real DHS sites around the TSS (not when '-rd' option is also set for random feature selection). (Default: 0.2)", type=float, default=0.2)
parser.add_argument("-Fd", "--take_this_many_top_dhss", help="Take this many DHS sites. If this is set to '-1', then all the known DHS sites in the region TSS +/- --distance or regulatory TFs is used. Note that if random set of features are to be used, namely by setting '-rd' option, then the same number of DHS sites are considered as in the non-random (i.e. original) set. (See details on '-rd' below.) (Default: 5)", type=int, default=5)
parser.add_argument("-rd", "--use_random_DHSs", help="If set, a set of DHS sites are randomly selected from the genome. The size of this set equals the size of the original (or non-random) feature set, which is equal to at most '--take_this_many_top_fts' DHS sites in the region of interest (eg. gene TSS+/-200kb) that have pearson correlation coefficient of at least '--pcc_lowerlimit_to_filter_dhss' value with the expression of the gene. (Default: False)", action="store_true")

# ============= Arguments pertaining to the TFs ===========
parser.add_argument("-Ft", "--take_this_many_top_tfs", help="Take this many TFs. If this is set to '-1', then all the known TFs that are predicted to be regulatory for the gene are used. Note that if random set of features are to be used, namely by setting '-rt' option, then the same number of TFs are considered as in the non-random (i.e. original) set. (See details on '-rt' below.) (Default: 8)", type=int, default=8)
parser.add_argument("-tff", "--filter_tfs_by", help="For the TF-TG association, filter the predicted list of regulatory TFs for the given gene using one of two measures: 1) Pearson Correlation Coefficient between the expression of TF and the target gene TG, or 2) Z-score indicating the significance of one TF-TG association given perturbation measurements of the expression of the TF and the TG across various experimental or biological conditions (see CellNet paper and CLR algorithm). (Default: 'zscores')", choices=["pcc", "zscore"], type=str, default="zscore")
parser.add_argument("-tfl", "--lowerlimit_to_filter_tfs", help="Lower limit of the measure --filter-tfs-by in absolute value. The value should be >0 for '--filter-tfs-by pcc' and >= 4.0 for '--filter-tfs-by zscores'. Note that the respective upper limits are 1.0 and infinity respectively, and therefore need not be declared. (Default: 6.0 for the default '--filter-tfs-by zscores'.)", default=6.0, type=float)
parser.add_argument("-rt", "--use_random_TFs", help="If set, instead of using cell-net predicted TFs that make up the GRN for this gene, same number of random TFs as in the original set are selected at random. (Default: Not set)", action="store_true")

# ============= Arguments pertaining to algorithm ===========
parser.add_argument("-w", "--init_wts_type", help="Relates to the initial wts set between the nodes. If 'random', random initial wts are set between any two nodes; if 'corr', initial wts between input and hidden nodes are set to the correlation values between the node feature and the expression of the gene, and the initial weights between hidden layers or the hidden layer and output is set to 0.5 (Default: 'corr')", choices=["random", "corr"], type=str, default="corr")
parser.add_argument("-m", "--max_iter", help="Maximum number of interations for neural net optimization (Default: 300)", type=int, default=300)  # stopping "early" also helps reduce over-fitting

# ============= Other arguments ===========
parser.add_argument("-o", "--outputDir", help="Output directory. A directory for this gene of interest and set of parameters used is created at this location. (Default is '../Output')", type=str, default=os.path.join(os.getcwd(), "../../Output"))
parser.add_argument("-k", "--run_id", help="Run_id for multiple parallel runs. This is useful in slurm. (Default: -1)", type=int, default=-1)
parser.add_argument("-p", "--plot_all", help="If set, all supplemental plots are also generated in addition to the scatterplots showing the performances after hyperparameter optimization and re-training (i.e. training the full training and validation set using the optimized hyperparameters). (Default: Not set)", action="store_true")

args = parser.parse_args()

# Check for basic argument setup
if (args.filter_tfs_by == "zscore"):
    try:
        assert args.lowerlimit_to_filter_tfs >= 4.0
    except:
        raise Exception("Make sure --lowerlimit_to_filter_tfs is set to match --filter_tfs_by.")
else:
    try:
        assert 0 < args.lowerlimit_to_filter_tfs < 1
    except:
        raise Exception("Make sure --lowerlimit_to_filter_tfs is set to match --filter_tfs_by.")


############################################################
# ####### End of parser setup; Set up the logger info ###### #
############################################################


def get_output_dir(args):
    # output files + directory parameters
    assert os.path.exists(args.outputDir)  # outputDir is updated below
    outputDir = "{0}/{1}_{2}kb_{3}_t{4}".format(args.outputDir, args.gene.upper(), args.distance, args.filter_tfs_by, args.lowerlimit_to_filter_tfs)
    outputDir += "_m" + str(args.max_iter)
    if (args.use_random_TFs):
        outputDir += "_rTFs"
    if (args.use_random_DHSs):
        outputDir += "_rDHSs"
    if (args.run_id > 0):  # rank is >=1; if > 0, means running for multiple set of random TFs for the same gene_ofInterest
        outputDir += "_run" + str(args.run_id)
    if (not os.path.exists(outputDir)):
        os.makedirs(outputDir)
    return outputDir


# create output dir, set the logging handlers (file + stream) and params
outputDir = get_output_dir(args)  # creates a specific directory in args.outputDir
formatter = logging.Formatter('%(asctime)s: %(name)-12s: %(levelname)-8s: %(message)s')

file_handler = logging.FileHandler(os.path.join(outputDir, args.gene.upper() + ".log"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info("Command line arguments: {}".format(args))


############################################################
# ####### Set up the data for the model ###### #
############################################################
start_time = time.time()

gv = Global_Vars(args, outputDir)  # gene and condition specific outputDir
mp = Model_preparation(gv)

'''Run HPO on differen train/test splits'''
for test_idx in range(0, 18):
    if (test_idx == 4):  # 4 corresponds to val_group of "ENCODE2012"; 2 to brain; 6 to ESC
        continue

    tm = Tensorflow_model(gv, mp, test_eid_group_index=test_idx)
    trials = hyperopt.Trials()

    best_params = hyperopt.fmin(tm.train_tensorflow_nn, trials=trials,
                                space=get_parameter_space_forHPO(tm.trainX),
                                algo=tpe_method, max_evals=15)

    to_log = tm.get_log_into_to_save(trials, best_params)
    logger.info(to_log)

    plot_title = re.split(";best_params", to_log)[0]
    plot_title = re.split(";median_pc_error|;PCC", plot_title)
    plot_title = plot_title[0] + "\nmed_pc_err" + plot_title[1] + "\nPCC" + plot_title[2]
    tm.plot_scatter_performance(trials, gv, plot_title=gv.gene_ofInterest + ";" + plot_title)

    del tm, trials, best_params

logger.info("Total time taken: {}".format(time.time() - start_time))
