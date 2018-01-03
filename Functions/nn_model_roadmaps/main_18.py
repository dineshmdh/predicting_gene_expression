# Created on Jan 1, 2018

import time
import os  # os is being used to set up default outputDir
import argparse
import sys

start_time = time.time()
sys.path = sys.path[1:]
sys.path.insert(0, os.path.join(os.getcwd(), "helper_scripts"))

import logging
from helper_functions import get_output_dir
import global_variables_git

############################################################
# ####### Set up the parser arguments ###### #
############################################################
parser = argparse.ArgumentParser(description="Using Neural Net, model the expression of genes using (1) DHS sites around the TSS, (2) TFs predicted to regulate the expression of the gene, and (3) both.")
parser.add_argument("gene", help="Gene of interest", type=str)

# ============= Arguments pertaining to DHS sites ===========
parser.add_argument("-d", "--distance", help="Distance from TSS (in kb) to cover as region of interest (Default: 150)", type=int, default=150)
parser.add_argument("-u", "--use_tad_info", help="Use TAD boundaries to demarcate the boundaries for the region of interest. (Default: True)", type=bool, default=True)
parser.add_argument("-F", "--take_this_many_top_dhs_fts", help="Take this many DHS sites that are most correlated (in absolute value) with the expession of the gene. If this is set to '-1', then all the known DHS sites in the region TSS +/- --distance is used. (Default: 20)", type=int, default=20)
parser.add_argument("-rd", "--use_random_DHSs", help="If set, a set of --take_this_many_top_dhs_fts number of DHS sites are randomly selected from the genome. (Default: False)", action="store_true")

# ============= Arguments pertaining to the TFs ===========
parser.add_argument("-tff", "--filter_tfs_by", help="For the TF-TG association, filter the predicted list of regulatory TFs for the given gene using one of two measures: 1) Pearson Correlation Coefficient between the expression of TF and the target gene TG, or 2) Z-score indicating the significance of one TF-TG association given perturbation measurements of the expression of the TF and the TG across various experimental or biological conditions (see CellNet paper and CLR algorithm). (Default: 'zscores')", choices=["pearson_corr", "zscores"], type=str, default="zscores")
parser.add_argument("-tfl", "--lowerlimit_to_filter_tfs", help="Lower limit of the measure --filter-tfs-by. The value should be >0 for '--filter-tfs-by pearson_corr' and >= 4.0 for '--filter-tfs-by zscores'. Note that the respective upper limits are 1.0 and infinity respectively, and therefore need not be declared. (Default: 5.0 for the default '--filter-tfs-by zscores'.)", default=5.0, type=float)
parser.add_argument("-rt", "--use_random_TFs", help="If set, instead of using cell-net predicted TFs that make up the GRN for this gene, same number of random TFs as in the original set are collected for this gene. (Default: False)", action="store_true")

# ============= Arguments pertaining to algorithm ===========
parser.add_argument("-w", "--init_wts_type", help="Relates to the initial wts set between the nodes. If 'random', random initial wts are set between any two nodes; if 'corr', initial wts between input and hidden nodes are set to the correlation values between the node feature and the expression of the gene, and the initial weights between hidden layers or the hidden layer and output is set to 0.5 (Default: 'corr')", choices=["random", "corr"], type=str, default="corr")
parser.add_argument("-s", "--to_seed", help="If set, numpy seed number is set (only) for random splitting of the training and test samples. The seed number is set to 4. (Default: False)", action="store_true")
parser.add_argument("-m", "--max_iter", help="Maximum number of interations for neural net optimization (Default: 300)", type=int, default=300)

# ============= Other arguments ===========
parser.add_argument("-o", "--outputDir", help="Output directory. A directory for this gene of interest and set of parameters used is created at this location. (Default is '../Output')", type=str, default=os.path.join(os.getcwd(), "../Output"))
parser.add_argument("-k", "--run_id", help="Run_id for multiple parallel runs. This is useful in slurm. (Default: -1)", type=int, default=-1)

args = parser.parse_args()

############################################################
# ####### End of parser setup; Set up the logger info ###### #
############################################################

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


def get_gv_and_model_prep_instances(args, outputDir):
    gv = Global_Vars(args, outputDir)
    return gv
