# Created on October 10
# Command line version of the training_neural_net_on_a_gene_using_TAD_info_improved_v4.ipynb script (also created today)

import pdb
import time
import os
import argparse  # should be after importing os (as os is being used for outputDir)
import sys

############################################################
# ####### Set up the parser arguments ###### #
############################################################

parser = argparse.ArgumentParser(description="Using Neural Net, model the expression of genes using (1) DHS sites around the TSS, (2) TFs predicted to regulate the expression of the gene, and (3) both.")
parser.add_argument("gene", help="Gene of interest", type=str)
parser.add_argument("-hs", "--hidden_size", help="Percent of the input layer size to be used in the hidden layer (Default: 60)", default=60, type=int)
parser.add_argument("-d", "--distance", help="Distance from TSS (in kb) to cover as region of interest (Default: 150)", type=int, default=150)
parser.add_argument("-m", "--max_iter", help="Maximum number of interations for neural net optimization (Default: 300)", type=int, default=300)
parser.add_argument("-T", "--tf_corr_threshold_high", help="Upper limit of the correlation threshold to select TFs for this gene. All TFs with expression in correlation with this gene in between this and the tf_corr_threshold_low values are selected for the model (Default: 1.0)", type=float, default=1.0)
parser.add_argument("-t", "--tf_corr_threshold_low", help="Lower limit of the correlation threshold to select TFs for this gene. All TFs with expression in correlation with this gene in between this and the *_high values are selected for the model (Default: 0.3)", type=float, default=0.3)
parser.add_argument("-en", "--enforce_corr_thresholding", help="If set, correlation based thresholding is turned on for when random TFs are selected for the target gene. Only applicable for selecting random TFs. (Default: Not set)", action="store_true")
parser.add_argument("-u", "--use_tad_info", help="Use TAD boundaries to demarcate the boundaries for the region of interest (Default: True)", type=bool, default=True)
parser.add_argument("-f", "--take_top_fts", help="only pertains to DHS sites. If True, only self.take_this_many_top_fts top DHS sites ranked by PCC with self.gene_ofInterest expression profile are selected. If False, all DHS sites are used. If the total number of DHS sites for this gene_ofInterest is less than self.take_this_many_top_fts, then all DHS sites will be used. This helps in limiting the number of parameters to learn and from dying on its own sometimes because of excessive memory consumption. (Default: True)", type=bool, default=True)
parser.add_argument("-F", "--take_this_many_top_fts", help="Take this many DHS sites that are most correlated (in absolute value) with the expession of the gene (Default: 20)", type=int, default=20)
parser.add_argument("-l", "--take_log2_tpm", help="If set, TPM values are log2 transformed before being scaled. (Recommended) (Default: True)", type=bool, default=True)
parser.add_argument("-w", "--init_wts_type", help="Relates to the initial wts set between the nodes. If 'random', random initial wts are set between any two nodes; if 'corr', initial wts between input and hidden nodes are set to the correlation values between the node feature and the expression of the gene, and the initial weights between hidden layer and output is set to 0.5 (Default: 'corr')", choices=["random", "corr"], type=str, default="corr")
parser.add_argument("-rd", "--use_random_DHSs", help="If set, a random set of --take_this_many_top_fts number of DHS sites are selected. The rest is as default. (Default: False)", action="store_true")
parser.add_argument("-rt", "--use_random_TFs", help="If set, instead of using cell-net predicted TFs that make up the GRN for this gene, random TFs (same number as the original set) are collected for this gene. Note that for the neural net modeling, the number of TFs that pass the corr_threshold might be smaller than the original set. (Default: False)", action="store_true")
parser.add_argument("-ft", "--frac_test", help="Fraction of cell lines to be used in the test set. The rest will be used to train the models. (Default:0.2)", default=0.2, type=float)
parser.add_argument("-lr", "--learning_rates", help="Use these learning rates to train the neural net model. Use as '-lr 0.1 0.01 etc'. Also note that the neural net model and performance plots specific to a learning rate are saved in single plot for DHS-only, TF-only and DHS-with-TF modeling cases (Default: '0.05')", nargs="+", default=[0.05], type=float)
parser.add_argument("-o", "--outputDir", help="Output directory. A directory for this gene of interest and set of parameters is created at this location. (Default is '../Output')", type=str, default=os.path.join(os.getcwd(), "../Output"))
parser.add_argument("-k", "--slurmrank", help="Slurm rank id for multiple parallel runs (Default: -1)", type=int, default=-1)
parser.add_argument("-s", "--to_seed", help="If set, numpy seed number is set (only) before random splitting of the training and test samples. The seed is not set before sampling random TFs (because the TF sampling is - if set - is done one TF at a time) and not before getting the weight matrices. The seed number is set to 4. (Default: False)", action="store_true")
args = parser.parse_args()
############################################################
# ####### End of parser setup ###### #
############################################################

start_time = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # To ignore TF warning in stdout (source: https://github.com/tensorflow/tensorflow/issues/7778)

sys.path = sys.path[1:]
sys.path.insert(0, os.path.join(os.getcwd(), "helper_scripts"))

import global_variables_01
import prep_for_model_02
import draw_neural_net
import tensorflow_model_functions

from global_variables_01 import Global_Vars
from prep_for_model_02 import Model_preparation
# from get_agenes_cellnetTF_gexes import Get_CellNet_tfs
from draw_neural_net import Draw_NN
from tensorflow_model_functions import *  # has functions to train tensorflow model and plot performance
# from helper_functions import get_layer1_layer2_sizes  # computes importance score of NN nodes
# from helper_functions import save_top_dhs_sites  # writes top DHS sites based on sizes / importance scores
# from helper_functions import plot_dhs_heatmap_wdhss_ranked_by_NN  # plots cell line by DHS (ranked by NN) heatmap
from helper_functions import merge_pdfs
from helper_functions import get_output_dir
from helper_functions import plot_corrs_vs_nodeSizes

############################################################
# ####### Set up the logger info ###### #
############################################################
# create output dir, set the logging handlers (file + stream) and params
outputDir = get_output_dir(args)  # creates a specific directory in args.outputDir

import logging
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
# ####### Get output directory ###### #
############################################################


def get_gv_and_model_prep_instances(args):
    # Set up the global + key variables
    gv_kws = {"gene_ofInterest": args.gene.upper(),
              "dist_lim": args.distance,
              "tf_corr_threshold_high": args.tf_corr_threshold_high,  # only used in writing the out file suffix; TF with |corr| in between this and *low are selected
              "tf_corr_threshold_low": args.tf_corr_threshold_low,  # TF with |corr| in between this and *high are selected
              "use_tad_info": args.use_tad_info,
              "take_top_fts": args.take_top_fts,
              "take_this_many_top_fts": args.take_this_many_top_fts,  # pertains to DHS features
              "take_log2_tpm": args.take_log2_tpm,  # only used in writing the out file suffix
              "use_wts": args.init_wts_type,  # only used in writing the out file suffix
              "use_random_DHSs": args.use_random_DHSs,
              "use_random_TFs": args.use_random_TFs,  # if True, random TFs are selected for this gene
              "outputDir": outputDir}
    gv = Global_Vars(**gv_kws)

    '''Set up the training and test data.'''
    model_prep_kws = {"args": args, "gv": gv}
    m = Model_preparation(**model_prep_kws)
    return m


############################################################
# ####### Set up the model instances and dataframes ###### #
############################################################

m = get_gv_and_model_prep_instances(args)


def plot_df_dhss_in_roi(m):
    goi = m.gene_ofInterest_info
    goi_train = goi[goi.index.isin(m.train_celllines)]
    goi_train_sorted = goi_train.sort_values(ascending=False)

    df_goi = goi_train_sorted.to_frame()
    df_goi = df_goi[df_goi.columns].astype(float)

    df = m.df_dhss_in_roi
    df["loc"] = df["chr_dhs"] + ":" + df["ss_dhs"].astype(str) + "-" + df["es_dhs"].astype(str)
    df.set_index(["loc"], inplace=True)
    pdb.set_trace()

    df_train = df[m.train_celllines]
    df_train = df_train[goi_train_sorted.index.tolist()]
    df_train = df_train[df_train.columns].astype(float)

    pdb.set_trace()

    sns.set(font_scale=1.3)
    plt.figure(figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    sns.heatmap(df_train.transpose(), vmax=30, yticklabels=True)
    plt.xticks(rotation=85)
    plt.tight_layout()
    plt.savefig(m.outputDir + "/df_dhss_in_roi_for_figure1.pdf")
    plt.close()


'''First set up for dhs only data'''
m.main_for_dhss()  # gets self.trainX etc only considering dhss as predictors.
plot_df_dhss_in_roi(m)  # at this point, dhs sites are filtered by their correlation scores
w1_dhssOnly, w2_dhssOnly = m.get_init_w1_w2_wts(args.hidden_size)

'''Now set up for tfs only or dhs with tfs data'''
df_trainX_wtfs, df_testX_wtfs, df_trainX_tfsOnly, df_testX_tfsOnly = m.main_for_tfsOnly_and_wTFs()
if not (df_trainX_wtfs is None):  # the other dfs right above are also None (b/c there was no TFs that passed the tf_corr_threshold)
    w1_wtfs, w2_wtfs, w1_tfsOnly, w2_tfsOnly = m.get_init_w1_w2_wts_wtfs_orForJust_tfs(df_trainX_wtfs, w1_dhssOnly, args.hidden_size)

# pdb.set_trace()

############################################################
# ################# Modeling in Tensorflow ############### #
############################################################


def plot_tf_nn_model(updates, m, sizes, labels, outFileName_model, num_top_labels_to_plot=8, fig_size=(8, 8)):
    '''Wrapper function that draws NN(using Draw_NN class) for the tensorflow model.
       : Arguments:
                                    - m is the Model_preparation class instance
                                    - sizes is a dict of layer1 and layer2 sizes: {"layer1": [], "layer2": []}.
                                    The sizes are normalized in the Draw_NN class.
    '''
    layer_sizes = [updates["W1"].shape[0], updates["W1"].shape[1], updates["W2"].shape[1]]  # [inputLayerSize, hiddenLayerSize, outputLayerSize]

    d = Draw_NN(m.outputDir, outFileName_model, sizes["layer1"], sizes["layer2"], layer_sizes, updates["W1"],
                updates["W2"], labels, num_labels_to_plot=num_top_labels_to_plot, geneName=m.gene_ofInterest, figsize=fig_size)
    nn_model = d.plot_nn()
    return nn_model


def get_train_test_params(mode):
    # Returns a dictionary of parameters given a mode in {"dhss", "tfs", "dhss_tfs"}

    params = {}
    assert mode in ["dhss", "tfs", "dhss_tfs"]
    if (mode == "dhss"):
        params["w1"] = w1_dhssOnly
        params["w2"] = w2_dhssOnly
        params["trainX"] = m.trainX
        params["trainY"] = m.trainY
        params["testX"] = m.testX
        params["testY"] = m.testY
        params["pdf_performance_fileName"] = m.gene_ofInterest + "_performance_wDHSs_lr" + str(alr) + ".pdf"
        params["nn_labels"] = m.get_input_labels_for_NN_plot()
        params["pdf_nnModel_fileName"] = m.gene_ofInterest + "_NN_model_onlyDHSs_lr" + str(alr) + ".pdf"
        params["num_top_labels_to_plot"] = 10
    elif (mode == "tfs"):
        params["w1"] = w1_tfsOnly
        params["w2"] = w2_tfsOnly
        params["trainX"] = m.trainX_tfsOnly
        params["trainY"] = m.trainY
        params["testX"] = m.testX_tfsOnly
        params["testY"] = m.testY
        params["pdf_performance_fileName"] = m.gene_ofInterest + "_performance_wTFs_lr" + str(alr) + ".pdf"
        params["nn_labels"] = df_trainX_tfsOnly.columns.tolist()
        params["pdf_nnModel_fileName"] = m.gene_ofInterest + "_NN_model_onlyTFs_lr" + str(alr) + ".pdf"
        params["num_top_labels_to_plot"] = 10
    elif (mode == "dhss_tfs"):
        params["w1"] = w1_wtfs
        params["w2"] = w2_wtfs
        params["trainX"] = m.trainX_wtfs
        params["trainY"] = m.trainY
        params["testX"] = m.testX_wtfs
        params["testY"] = m.testY
        params["pdf_performance_fileName"] = m.gene_ofInterest + "_performance_wDHSs_and_TFs_lr" + str(alr) + ".pdf"
        params["nn_labels"] = m.get_input_labels_for_NN_plot() + df_trainX_tfsOnly.columns.tolist()
        params["pdf_nnModel_fileName"] = m.gene_ofInterest + "_NN_model_wDHSsAndTFs_lr" + str(alr) + ".pdf"
        params["num_top_labels_to_plot"] = 10
    else:
        raise Exception("The mode is not defined.")
    return params


'''Train and test over a number of learning rates'''
if (df_trainX_wtfs is None):  # there is no TF associated with this gene present or passing the threshold
    modes = ["dhss"]
else:
    modes = ["dhss", "tfs", "dhss_tfs"]

for alr in args.learning_rates:
    logger.info("training with learning rate {}..".format(alr))
    pdfs_perf_to_merge = []  # performance pdfs to merge for each learning rate
    pdfs_nnmodel_to_merge = []
    nodeSizes_for_modes = {}  # will be used to plot corr vs node sizes scatter plot later

    for amode in modes:
        # logger.info("    Working in mode: {}".format(amode))  # better to use the performance log info
        params = get_train_test_params(mode=amode)
        updates, sizes = train_tensorflow_nn(logger, params["w1"], params["w2"],
                                             params["trainX"], params["trainY"], params["testX"], params["testY"],
                                             max_iter=args.max_iter, pkeep_train=0.7, starter_learning_rate=alr)  # sizes = {"layer1":[], "layer2":[]}
        nodeSizes_for_modes[amode] = sizes

        plot_costs_and_performance(updates, m, logger, mode=amode, learning_rate=alr, plot_name_with_suffix=params["pdf_performance_fileName"])
        plot_tf_nn_model(updates, m, sizes, params["nn_labels"], params["pdf_nnModel_fileName"], params["num_top_labels_to_plot"])
        pdfs_perf_to_merge.append(os.path.join(m.outputDir, params["pdf_performance_fileName"]))
        pdfs_nnmodel_to_merge.append(os.path.join(m.outputDir, params["pdf_nnModel_fileName"]))
        logger.info("    gene:{}, mode:{}, lr:{}, layer1_node_and_sizes: {}, layer2_node_sizes: {}".format(
            m.gene_ofInterest, amode, alr,
            zip(params["nn_labels"], [round(x, 3) for x in sizes["layer1"].tolist()]),
            [round(x, 3) for x in sizes["layer2"].tolist()]))

        if (amode == 'dhss'):
            plt.plot(sizes["layer1"])
            plt.savefig("{}/layer1_{}_randomSeed.pdf".format(m.outputDir, m.gene_ofInterest))
            plt.close()
            sns.heatmap(updates["W1"])
            plt.savefig("{}/w1_{}_afterTrain_randomSeed.pdf".format(m.outputDir, m.gene_ofInterest))
            plt.close()
            plt.plot(sizes["layer2"])
            plt.savefig("{}/layer2_{}_randomSeed.pdf".format(m.outputDir, m.gene_ofInterest))
            plt.close()
            sns.heatmap(updates["W2"])
            plt.savefig("{}/w2_{}_afterTrain_randomSeed.pdf".format(m.outputDir, m.gene_ofInterest))
            plt.close()
    plot_corrs_vs_nodeSizes(nodeSizes_for_modes, m, alr, logger)

    # Merge the performance pdfs for this learning rate and remove the individual plots afterwards
    pdf_perf_merged = os.path.join(m.outputDir, m.gene_ofInterest + "_performances_merged_for_lr" + str(alr) + ".pdf")
    merge_pdfs(pdfs_perf_to_merge, pdf_perf_merged)
    for apdf in pdfs_perf_to_merge:
        os.remove(apdf)

    # similarly merge the models and remove the individual models
    pdf_nnmodel_merged = os.path.join(m.outputDir, m.gene_ofInterest + "_NN_models_merged_for_lr" + str(alr) + ".pdf")
    merge_pdfs(pdfs_nnmodel_to_merge, pdf_nnmodel_merged)
    for apdf in pdfs_nnmodel_to_merge:
        os.remove(apdf)

logger.info("Total time taken is {}".format(time.time() - start_time))
