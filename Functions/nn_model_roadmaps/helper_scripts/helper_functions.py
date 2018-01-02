'''
Created on August 14, 2017

A collection of helper functions for gene expression prediction models.
'''

import os
import re
import time
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import scipy.stats
from PyPDF2 import PdfFileMerger


def get_output_dir(args):
    # output files + directory parameters
    assert os.path.exists(args.outputDir)  # outputDir is updated below
    outputDir = "{0}/{1}_{2}kb_{3}_t{4}".format(args.outputDir, args.gene.upper(), args.distance, args.filter_tfs_by, args.lowerlimit_to_filter_tfs)
    if (args.init_wts_type == "random"):
        outputDir += "_rWts"
    elif (args.init_wts_type == "corr"):
        outputDir += "_cWts"
    else:
        raise Exception()
    outputDir += "_m" + str(args.max_iter)
    if (args.use_random_TFs):
        outputDir += "_rTFs"
    if (args.use_random_DHSs):
        outputDir += "_rDHSs"
    if (args.run_id > 0):  # rank is >=1; if > 0, means running for multiple set of random TFs for the same gene_ofInterest
        outputDir += "_run" + str(args.run_id)
    if (args.to_seed):
        outputDir += "_s"
    if (not os.path.exists(outputDir)):
        os.makedirs(outputDir)
    return outputDir


def get_genome_wide_enhancers_df(csv_enhancer_tss, logger):
    '''Load the fantom enhancer-tss info'''
    handleIn = open(csv_enhancer_tss, "r")
    lines = handleIn.readlines()
    handleIn.close()

    '''Converting fantom bed12 enhancer-tss track lines to bed4 df of format: chrom, enh_ss, enh_es, geneName'''
    list_enhancer_gene = []
    genes_not_included_count = 0

    for aline in lines[2:]:
        chrom, ss, es, name, score, strand, thickStart, thickEnd, itemRgb, blockCount, blockSizes, blockStarts = re.split("\s+", aline.strip())

        blockSizes = [int(x) for x in re.split(",", blockSizes) if len(x) > 0]
        blockStarts = [int(x) for x in re.split(",", blockStarts.strip()) if len(x) > 0]  # removing empty strings (if any)
        assert int(blockCount) == len(blockSizes) == len(blockStarts)

        try:
            geneName = re.split(";", name)[2]
        except:
            genes_not_included_count += 1
            if (genes_not_included_count == 1):
                logger.debug("The following gene names are not included in get_genome_wide_enhancers_df():")
            logger.debug("    {}".format(name))

        for i in xrange(int(blockCount)):
            enhancer_gene_item = [chrom, int(ss) + blockStarts[i], int(ss) + blockStarts[i] + blockSizes[i], geneName]
            list_enhancer_gene.append(enhancer_gene_item)

    logger.warning("Total number of gene names not considered in get_genome_wide_enhancers_df() is {}".format(genes_not_included_count))
    df_enh_tsss_info = pd.DataFrame.from_records(list_enhancer_gene, columns=["chrom", "enh_ss", "enh_es", "gene"])

    '''Note: Each line in this enhancer-tss file has 2 blocks of enhancers specified (field#10 in the file) -
    so b/c there are 66942 such lines, there are 133884 enhancers in total - see below.
    But, for multiple blocks of enhancers corresponding to a gene, the first block (out of the two in a line)
    seems to be exact. So, only selecting the unique enhancer-tss values. So, Uniquifying the enhancer-tss
    associations by dropping duplicates'''
    # print("removing duplicate enhancers..")
    df_enh_tsss_info.drop_duplicates(inplace=True)
    df_enh_tsss_info.sort_values(["chrom", "enh_ss"], inplace=True)

    return df_enh_tsss_info


def get_full_tpm_info_df(csv_tpm_merged, df_gencode_all_mrnas):
    '''Return a df with gene loc as well as gene expression info for samples in csv_tpm_merged file.

    Arguments:
    csv_tpm_merged -- has fields ensemble_gene_id sample_name1 sample_name2 etc.
    csv_gencode_mrnas -- gencode file with fields specified below.
    '''
    # st = time.time()
    df_tpm = pd.read_csv(csv_tpm_merged, header=0, sep="\t")
    '''Merge the gencode_df and tpm_df.'''
    df_tpms_anno = pd.merge(df_gencode_all_mrnas, df_tpm, on="gene_id")
    # print("get_full_tpm_info_df():", time.time() - st)
    return df_tpms_anno


def get_dhss_all_df(csv_dhs_normed, logger):
    '''Return df_dhs_signal with bed3 format loc info.

    Argument:
    csv_dhs_normed -- has fields chr:ss-es (as "Unnamed: 0"), sample1_dhs_signal,
                              sample2_dhs_signal, etc
    '''
    st = time.time()
    df_alldhss_normed = pd.read_csv(csv_dhs_normed, sep="\t", header=0)

    '''The first column name is 'Unnamed: 0'. Changing it'''
    df_alldhss_normed.rename(columns={'Unnamed: 0': 'loc_dhss'}, inplace=True)
    '''Extracting the locs to bed3 format'''
    df_all_dhss_locsOnly = df_alldhss_normed["loc_dhss"].str.split("[:-]", expand=True)
    df_all_dhss_locsOnly.columns = ["chr_dhs", "ss_dhs", "es_dhs"]
    '''Updating the df with this bed3 loc and dropping the old loc field'''
    df_alldhss_normed = pd.concat([df_all_dhss_locsOnly, df_alldhss_normed], axis=1)
    df_alldhss_normed.drop(["loc_dhss"], axis=1, inplace=True)

    logger.debug("Time taken for get_dhss_all_and_for_this_tad(): {}".format(time.time() - st))
    return df_alldhss_normed


def get_layer1_layer2_sizes(N, X, Y):
    '''Compute the node sizes for the Neural Net N. The size of a node is computed by
    seeing how much removing that node affects the overall RMSE.
    '''
    sizes_layer1 = np.zeros(N.W1.shape[0])
    rmse_base = np.sqrt(sum([x**2 for x in N.forward(X).flatten() - Y.flatten()]) / Y.flatten().shape[0])

    for i in range(0, N.W1.shape[0]):
        w1 = copy.deepcopy(N.W1)
        w1[i, :] = 0
        assert not np.array_equal(N.W1, w1)
        rmse_shuf = np.sqrt(sum([x**2 for x in N.forward_(X, w1, N.W2).flatten() - Y.flatten()]) / Y.flatten().shape[0])

        sizes_layer1[i] = (rmse_shuf - rmse_base) * 100

    '''Similarly get the weights for the second layer'''
    sizes_layer2 = np.zeros(N.W2.shape[0])

    for i in range(0, N.W2.shape[0]):
        w2 = copy.deepcopy(N.W2)  # shape is (#hidden layers, 1)
        w2[i, 0] = 0
        assert not np.array_equal(N.W2, w2)
        rmse_shuf = np.sqrt(sum([x**2 for x in N.forward_(X, N.W1, w2).flatten() - Y.flatten()]) / Y.flatten().shape[0])
        sizes_layer2[i] = (rmse_shuf - rmse_base) * 100

    return sizes_layer1, sizes_layer2


def save_top_dhs_sites(labels, sizes_layer1, outputDir):
    '''Given a list of labels (each of format chr:ss-es) and another with corresponding sizes,
    save the dhs sites in bed format to view in genome browser later.'''
    assert len(labels) == len(sizes_layer1)
    if (len(labels) >= 10):
        saveThisManyDhsFts_list = np.arange(2, 10, 3)
    else:
        saveThisManyDhsFts_list = np.arange(2, len(labels), 3)
    list_sizes_sorted = sorted(dict(zip(labels, sizes_layer1)).items(), reverse=True, key=lambda x: x[1])

    '''Write the top dhs fts bed file'''
    for saveThisManyDhsFts in saveThisManyDhsFts_list:
        handleOut = open(outputDir + "/" + str(saveThisManyDhsFts) + "topDhsFts.bed", "w")
        for i in range(0, saveThisManyDhsFts):
            vals = re.split("[:-]", list_sizes_sorted[i][0])
            handleOut.write("\t".join(vals) + "\n")
    handleOut.close()


def plot_corrs_vs_nodeSizes(nodeSizes_for_modes, m, lr, logger):
    '''A mode is one of "dhss", "tfs" or "dhss_tfs".
    Arugments:
    - {mode: sizes_dict}, where
    sizes_dict == {"layer1": np.array([node1_size, etc])
                   "layer2": np.array([node1_size, etc])}
    - m is the model preparation object. It has trainX,
    trainX_wtfs or trainX_onlyTFs matrices that will be used
    with trainY arrays to compute the correlations.
    - lr : current learning rate

    Note:
    1. only trainX/Y are used.
    2. Assumption is that the order of node sizes in sizes_dict
    matches the order in trainX.
    '''
    sns.set(font_scale=1.2)
    if (len(nodeSizes_for_modes.keys()) == 1):  # just one mode
        assert nodeSizes_for_modes.keys() == ["dhss"]
        plt.figure(0, (4, 3.8))
        corrs = []
        for i in range(0, m.trainX.shape[1]):  # m.trainX has examples as rows and sites as columns
            corrs.append(np.corrcoef(m.trainX[:, i].flatten(), m.trainY.flatten())[0, 1])
        plt.scatter(corrs, nodeSizes_for_modes["dhss"]["layer1"])
        plt.xlabel("PCC with {} gex".format(m.gene_ofInterest))
        plt.ylabel("Model node sizes")
    else:
        assert set(nodeSizes_for_modes.keys()) == set(["dhss", "tfs", "dhss_tfs"])
        plt.figure(0, (12, 4.3))
        plt.suptitle("PCC versus node sizes for {}".format(m.gene_ofInterest))

        plt.subplot(1, 3, 1)
        corrs = []
        for i in range(0, m.trainX.shape[1]):  # m.trainX has examples as rows and sites as columns
            corrs.append(np.corrcoef(m.trainX[:, i].flatten(), m.trainY.flatten())[0, 1])
        plt.scatter(corrs, nodeSizes_for_modes["dhss"]["layer1"])
        plt.xlabel("PCC")
        plt.ylabel("Model node sizes")
        plt.title("DHS sites only")

        plt.subplot(1, 3, 2)
        corrs = []
        for i in range(0, m.trainX_tfsOnly.shape[1]):  # m.trainX has examples as rows and sites as columns
            corrs.append(np.corrcoef(m.trainX_tfsOnly[:, i].flatten(), m.trainY.flatten())[0, 1])
        plt.scatter(corrs, nodeSizes_for_modes["tfs"]["layer1"])
        plt.xlabel("PCC")
        plt.ylabel("Model node sizes")
        plt.title("TFs only")

        plt.subplot(1, 3, 3)
        corrs = []
        for i in range(0, m.trainX_wtfs.shape[1]):  # m.trainX has examples as rows and sites as columns
            corrs.append(np.corrcoef(m.trainX_wtfs[:, i].flatten(), m.trainY.flatten())[0, 1])
        plt.scatter(corrs, nodeSizes_for_modes["dhss_tfs"]["layer1"])
        plt.xlabel("PCC")
        plt.ylabel("Model node sizes")
        plt.title("DHS sites with TFs")
    plt.tight_layout()
    plt.savefig("{}/{}_corr_vs_pcc_at_lr{}.pdf".format(m.outputDir, m.gene_ofInterest, lr))
    plt.close()


def plot_dhs_heatmap_wdhss_ranked_by_NN(df_dhss_in_roi, labels, sizes_layer1, outputDir,
                                        train_celllines, test_celllines, gene_ofInterest, gene_ofInterest_info,
                                        take_log2_tpm=False, add_corr_in_names=True):
    '''Plot heatmap with cell lines as columns and dhs sites sorted by NN scores as rows.
    Arguments:
    - df_dhss_in_roi -- main df obtained from Global_var() class.
                        It is trimmed to get top dhs sites, has has_known_enhancer and
                        init_wt params as fields as well.
    - labels -- dhs sites in chr:ss-es format
    - sizes_layer1 -- importance score of each label / input node obtained with
                      get_layer1_layer2_sizes(N, X, Y) function above.

    - take_log2_tpm  # False is prefered b/c dnase values are not-logged.
    - add_corr_in_names  # if True, correlations are added in the heatmap dhs names
    '''

    '''Reindex'''
    new_indices = zip(df_dhss_in_roi["chr_dhs"], df_dhss_in_roi["ss_dhs"], df_dhss_in_roi["es_dhs"])
    new_indices = [x[0] + ":" + str(x[1]) + "-" + str(x[2]) for x in new_indices]
    df_dhss_in_roi.index = new_indices

    '''Get the dhs_df. Need to resort the labels.'''
    list_sizes_sorted = sorted(dict(zip(labels, sizes_layer1)).items(), reverse=True, key=lambda x: x[1])
    df_toplot_dhss = df_dhss_in_roi[train_celllines + test_celllines].transpose()
    df_toplot_dhss = df_toplot_dhss[[x[0] for x in list_sizes_sorted]]  # columns (i.e. dhs sites) are reordered

    '''Get gene_expn_dfs '''
    df_toplot_thisgene = pd.DataFrame(gene_ofInterest_info[train_celllines + test_celllines])
    df_toplot_thisgene.columns = [gene_ofInterest]

    '''Merge the dfs'''
    assert np.array_equal(df_toplot_dhss.index, df_toplot_thisgene.index)
    df_toplot = pd.concat([df_toplot_thisgene, df_toplot_dhss], axis=1)

    '''Also, this was needed - to convert to float type for all columns. genename column were not in float'''
    df_toplot = df_toplot[df_toplot.columns].astype(float)

    if (take_log2_tpm):
        df_toplot[gene_ofInterest] = df_toplot[gene_ofInterest].apply(lambda x: np.log2(x + 1))

    if (add_corr_in_names):
        # print("adding correlations to the dhs names..")
        '''Get dhs-gex correlation info to sort'''
        list_dhss_to_pccs = []  # has (dhs_label, pcc) tuples; need to add the corr values to the dhs names
        for i in range(1, len(df_toplot.columns)):  # the first one is the gene expn
            pcc = scipy.stats.spearmanr(df_toplot[gene_ofInterest].tolist(), df_toplot[df_toplot.columns[i]].tolist())[0]
            list_dhss_to_pccs.append((df_toplot.columns[i], pcc))

        '''Rename the indices to add correlation values'''
        dict_name_to_newNames = {k: k + " " + str(round(dict(list_dhss_to_pccs)[k], 2)) for k in dict(list_dhss_to_pccs).keys()}
        df_toplot.rename(columns=dict_name_to_newNames, inplace=True)

    '''Plot'''
    sns.set(font_scale=1.3)
    fig = plt.figure(figsize=(15, 12))
    sns.heatmap(df_toplot, vmax=25, yticklabels=True, xticklabels=True)
    plt.tight_layout()
    plt.savefig(outputDir + "/" + gene_ofInterest + "_heatmap_dhss_sortedByNN.pdf")
    plt.close()

    return fig


def merge_pdfs(pdfs_list, out_pdf_path_and_name):
    '''Given a list of pdfs in pdfs_list (with full path), merge and return the output pdf.
    Source: https://stackoverflow.com/questions/17104926/pypdf-merging-multiple-pdf-files-into-one-pdf
    This function will be used to merge the performances for DHS-only, TF-only and DHS-TF-joint models.
    '''
    merger = PdfFileMerger()
    for pdf in pdfs_list:
        merger.append(pdf)
    merger.write(out_pdf_path_and_name)
