
# coding: utf-8

# ### Given a list of gene symbols, plot the expression of the genes across the celllines.
#
# ** Doing this because the DNase NN model / chromatin accessibility around some genes does not
# explain the variation in expression across the cell lines.. **
# ### Sometimes the sorted list of gene symbols from the cell net don't exactly match the ones extracted from gencode list (while merging as done above). We need to check for and remove these genes.
#
# There are two ways that the match doesn't happen:
# 1. The genes from cell net is not found in the list from gencode. If this is the case, the corresponding gene is removed from the genes_to_plot list.
# 2. There may be more than one row with the same gene symbol present in the gencode derived df. An example is SHOX.

import pdb
import pandas as pd
import random  # if self.get_random_TFs is True
import numpy as np
import os
import collections as col
import scipy.stats
import logging

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns

pd.options.mode.chained_assignment = None  # default='warn' (see README.md)
pd.set_option('expand_frame_repr', False)

plt.rcParams['figure.figsize'] = (20.0, 20.0)  # (6.0,4.0)
plt.rcParams['font.size'] = 15  # 10
plt.rcParams['savefig.dpi'] = 500  # 72
plt.rcParams['figure.subplot.bottom'] = 0.1


class Get_CellNet_tfs(object):

    def __init__(self, gene_ofInterest, inputDir, outputDir,
                 tf_corr_threshold_high, tf_corr_threshold_low, enforce_corr_thresholding,
                 df_allgenes_tpm_anno, add_corr_in_names=True, take_pearsonr=True, get_random_TFs=False):

        # set the logging handlers and params
        formatter = logging.Formatter('%(asctime)s: %(name)-12s: %(levelname)-8s: %(message)s')

        file_handler = logging.FileHandler(os.path.join(outputDir, gene_ofInterest + '.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.info("Setting up the cellnet instance..")
        self.gene_ofInterest = gene_ofInterest  # plotting the expression of regulatory genes corresponding to this gene of interest
        self.add_corr_in_names = add_corr_in_names  # if True, the gene names will have correlation scores appended to the end in the plot
        self.take_pearsonr = take_pearsonr  # if false, spearmancorr will be used
        self.inputDir = inputDir  # has input files to be used here
        self.outputDir = outputDir
        self.tf_corr_threshold_high = tf_corr_threshold_high
        self.tf_corr_threshold_low = tf_corr_threshold_low
        self.enforce_corr_thresholding = enforce_corr_thresholding

        # LOAD BASIC DFS FOR ALL GENES
        self.csv_cellnet = os.path.join(self.inputDir, "Human_Big_GRN_032014.csv")
        self.df_cellnet = pd.read_csv(self.csv_cellnet, header=1,
                                      names=["TG", "TF", "zscore", "corr", "type", "species"])
        self.df_allgenes_tpm_anno = df_allgenes_tpm_anno

        self.all_tfs = list(set(self.df_cellnet["TF"]))

        # GET DFS FOR THIS GENE OF INTEREST OR ITS REGULATORY TFS
        self.df_geneOfInterest = self.df_cellnet[self.df_cellnet["TG"] == self.gene_ofInterest]  # get tfs for tg
        '''Get a list of genes to plot. The first one is the gene of interest.
        Note the same TF can be listed more than once (for eg: MYF5 for TNNC2), which is why, I am converting to a set first below'''

        self.genes_to_plot = [self.gene_ofInterest] + list(set(self.df_geneOfInterest["TF"]))
        self.df_genes_to_plot_tpm_anno = self.df_allgenes_tpm_anno[self.df_allgenes_tpm_anno["geneSymbol"].isin(self.genes_to_plot)]

        self.check_if_cellnet_gene_isInGencode()  # updates self.genes_to_plot if needed
        self.check_if_geneSymbol_unique()  # updates self.df_genes_to_plot_tpm_anno if needed

        self.logger.debug("There are 4 things to note in the rest of the initialization:")
        self.logger.debug("    1. self.df_to_plot (with ALL TFs as rows and ALL samples as columns) is used to plot the heatmap here. In the prep_for_model.py later, this df is 'tidied up' and is called df_tpm_cellnetTFs there. But it is not used later other than to check if len(its.index)>0 in prep_for_model.py - if it is then it means that there is no TF for this gene in cellnet.")
        self.logger.debug("    2. self.genes_to_plot will and should always have self.gene_ofInterest as its first element. This will also only contain selected TFs (i.e. TFs that pass corr threshold) by the end of the initialization.")
        self.logger.debug("    3. the tfs_to_corr and tfs_to_tfsWCorrs sorted dicts do not contain self.gene_ofInterest. The TFs (or keys) are sorted by the corr values.")
        self.logger.debug("    4. When random TFs are selected, same number of TFs as in the original set that pass the PCC are selected. These TFs are not filtered by their corr values.")

        '''Protocol: Now get the final df to plot, get and add corr vals, plot the df, update dicts, update the dfs and genes_to_plot w/ selected tfs'''
        df_to_plot = self.df_genes_to_plot_tpm_anno[["geneSymbol"] +
                                                    self.df_genes_to_plot_tpm_anno.columns[8:].tolist()]
        df_to_plot.set_index(["geneSymbol"], inplace=True)
        self.df_to_plot = df_to_plot[df_to_plot.columns].astype(float)  # this step seems to have solved the TAL1 heatmap plot issue!!
        self.tfs_to_corrs_sortedDict, self.tfs_to_tfsWCorrs_sortedDict = self.get_tfs_to_corrs_or_tfsWCorrs_dict(self.df_to_plot)  # used if random TFs are to be selected; else will be imported by prep_for_model.py; does not contain self.gene_ofInterest
        self.df_to_plot = self.add_and_sortBy_TFcorr_in_df_indices(self.df_to_plot, self.add_corr_in_names, self.tfs_to_tfsWCorrs_sortedDict)
        self.plot_final_df(self.df_to_plot, plot_title="df_to_plot with ALL TFs and all {} samples".format(len(self.df_to_plot.columns)))  # cellnet TFs are not sorted by their corr values

        self.tfs_to_corrs_sortedDict, self.tfs_to_tfsWCorrs_sortedDict = self.filter_tfs_to_corrs_dict_by_corr_vals(self.tfs_to_corrs_sortedDict, self.tfs_to_tfsWCorrs_sortedDict, filter_by_corr=True)

        self.genes_to_plot = [self.gene_ofInterest] + self.tfs_to_corrs_sortedDict.keys()  # updated with TFs passing corr thresholds
        for i, j in enumerate(self.tfs_to_corrs_sortedDict.items()):
            self.logger.info("    TF selected {}, {}:{:.2f}".format(i, j[0], j[1]))

        '''If random TFs are to be selected, select same number of TFs as in genes_to_plot (minus self) at random from the cellnet list.
        The rest is pretty much the same as for original TFs.'''
        self.get_random_TFs = get_random_TFs
        if (self.get_random_TFs):
            self.genes_to_plot = [self.gene_ofInterest] + self.get_random_tfs(tfs_to_use=self.tfs_to_corrs_sortedDict.keys(),
                                                                              enforce_corr_thresholding=self.enforce_corr_thresholding)  # the input here is filtered list of TFs
            self.df_genes_to_plot_tpm_anno = self.df_allgenes_tpm_anno[self.df_allgenes_tpm_anno["geneSymbol"].isin(self.genes_to_plot)]
            self.check_if_cellnet_gene_isInGencode()  # updates self.genes_to_plot if needed
            self.check_if_geneSymbol_unique()  # updates self.df_genes_to_plot_tpm_anno if needed

            df_to_plot = self.df_genes_to_plot_tpm_anno[["geneSymbol"] +
                                                        self.df_allgenes_tpm_anno.columns[8:].tolist()]
            df_to_plot.set_index(["geneSymbol"], inplace=True)
            self.df_to_plot = df_to_plot[df_to_plot.columns].astype(float)
            self.tfs_to_corrs_sortedDict, self.tfs_to_tfsWCorrs_sortedDict = self.get_tfs_to_corrs_or_tfsWCorrs_dict(self.df_to_plot)  # these will be imported by prep_for_model.py; does not contain self.gene_ofInterest
            self.df_to_plot = self.add_and_sortBy_TFcorr_in_df_indices(self.df_to_plot, self.add_corr_in_names, self.tfs_to_tfsWCorrs_sortedDict)
            self.plot_final_df(self.df_to_plot, plot_title="df_to_plot with same num of rnd TFs as orig TFs filtd by PCC but all {} samples".format(len(self.df_to_plot.columns)))

            self.tfs_to_corrs_sortedDict, self.tfs_to_tfsWCorrs_sortedDict = self.filter_tfs_to_corrs_dict_by_corr_vals(self.tfs_to_corrs_sortedDict, self.tfs_to_tfsWCorrs_sortedDict, filter_by_corr=False)  # we already have exact number of TFs as in orig filtered set (so not filtering now) - filtering would make no difference
            self.genes_to_plot = [self.gene_ofInterest] + self.tfs_to_corrs_sortedDict.keys()  # updated with TFs passing corr thresholds
            for i, j in enumerate(self.tfs_to_corrs_sortedDict.items()):
                self.logger.info("    random TF selected {}, {}:{:.2f}".format(i, j[0], j[1]))

    def add_and_sortBy_TFcorr_in_df_indices(self, df, to_add_corr_or_not, tfs_to_tfsWCorrs_sortedDict):
        '''Add corr values in df.index names.
        Assumption is that df.index is a TF name.

        Argument:
        - df : think of this as df_to_plot; with TFs as rows and C/T-samples as columns
        - to_add_corr_or_not: self.add_corr_in_names (T/F)
        - tfs_to_tfsWCorrs_sortedDict: {"tf": "tf <corr_val>"} sorted by corr val
        '''
        # Rename the indices to add correlation values'''
        if (to_add_corr_or_not is True):
            df.rename(index=tfs_to_tfsWCorrs_sortedDict, inplace=True)
            df = df.reindex(tfs_to_tfsWCorrs_sortedDict.values())
        else:
            df = df.reindex(tfs_to_tfsWCorrs_sortedDict.key())  # use sorted keys (/TFs)
        return df

    def get_random_tfs(self, tfs_to_use, enforce_corr_thresholding=False):
        # Involves 2 steps:
        # 1. get tf_corr_dict tfs (that already pass the thresholds) - which is the input
        # 2. till we have the same number of tfs (passing the thresholds), select TFs at random
        rand_genes_to_use = []
        while (len(rand_genes_to_use) != len(tfs_to_use)):
            rand_gene = random.sample(self.all_tfs, 1)[0]
            if ((rand_gene not in rand_genes_to_use) and  # to make sure genes are unique (i.e. condition 2)
                    (self.df_allgenes_tpm_anno["geneSymbol"].tolist().__contains__(rand_gene))):  # rand_gene is in the list of gencode genes (and therefore has tpm values) (i.e. condition 1)
                if (enforce_corr_thresholding):
                    gex_for_thisGene = self.df_allgenes_tpm_anno[self.df_allgenes_tpm_anno["geneSymbol"] == self.gene_ofInterest].values.flatten()[8:].tolist()  # values returns 2d array; gex values starts at 9th index
                    df_gex_for_randGene = self.df_allgenes_tpm_anno[self.df_allgenes_tpm_anno["geneSymbol"] == rand_gene].drop_duplicates("geneSymbol")  # some genes/tfs have multiple rows
                    gex_for_randGene = df_gex_for_randGene.values.flatten()[8:].tolist()

                    try:
                        corr_for_rand_gene = np.corrcoef(gex_for_thisGene, gex_for_randGene)[0, 1]
                    except ValueError:
                        pdb.set_trace()

                    # filter by corr values
                    if ((abs(corr_for_rand_gene) < self.tf_corr_threshold_high) and
                            (abs(corr_for_rand_gene) > self.tf_corr_threshold_low)):
                        rand_genes_to_use.append(rand_gene)
                else:
                    rand_genes_to_use.append(rand_gene)

        return rand_genes_to_use

    def check_if_geneSymbol_unique(self):
        ''' CONDITION 2:
        Checking the second condition first: i.e., is there a geneSymbol in more than one row?
        If there is, save the one with the maximum correlation.'''
        if not np.array_equal(sorted(list(set(self.df_genes_to_plot_tpm_anno["geneSymbol"]))), sorted(self.df_genes_to_plot_tpm_anno["geneSymbol"].tolist())):  # sorted set vs sorted list

            '''Save a dict of this format {geneSymbol: count_in_tpm_anno_df}'''
            dict_geneSymbolCount_in_df_tpm = {agene: self.df_genes_to_plot_tpm_anno["geneSymbol"].tolist().count(agene) for agene in set(self.df_genes_to_plot_tpm_anno["geneSymbol"])}
            for ageneSymbol, ageneCount in dict_geneSymbolCount_in_df_tpm.items():
                if (ageneCount > 1):

                    '''Save a temp df with the rows for this gene. Pick a row with the max corr with gene of interest from this temp df'''
                    df_temp = self.df_genes_to_plot_tpm_anno[self.df_genes_to_plot_tpm_anno["geneSymbol"] == ageneSymbol]

                    '''Also, remove all the rows corresponding to this gene - will concat the saved row to this later'''
                    self.df_genes_to_plot_tpm_anno = self.df_genes_to_plot_tpm_anno[self.df_genes_to_plot_tpm_anno["geneSymbol"] != ageneSymbol]

                    '''Create a dict to save correlation by index on df_temp'''
                    dict_index_to_corr = col.OrderedDict()  # of format {index_in_df_temp: pearsonr with gene_of_interest expn}
                    for i in range(0, df_temp.shape[0]):  # for each row with the same geneSymbol
                        '''Compute correlation with the gene of interest and pick the row with the highest correlation'''
                        dict_index_to_corr[i] = scipy.stats.pearsonr(df_temp.iloc[i][8:], self.df_genes_to_plot_tpm_anno[self.df_genes_to_plot_tpm_anno["geneSymbol"] == self.gene_ofInterest].iloc[0][8:])[0]

                    '''first remove the nan corrs. saw that the correlation is nan for the second SHOX gene'''
                    for i in range(0, len(dict_index_to_corr.items())):
                        a, b = dict_index_to_corr.items()[i]
                        if (np.isnan(b)):  # saw that this is true for second SHOX gene
                            del dict_index_to_corr[a]

                    '''Out of the ones that are left - i.e. the ones without corr == nan'''
                    index_to_save = dict_index_to_corr.values().index(max(dict_index_to_corr.values()))  # save the row with the maximum correlation

                    '''Finally, concatenate the dfs: one without this gene, and another with the one saved from df_temp'''
                    self.df_genes_to_plot_tpm_anno = pd.concat([self.df_genes_to_plot_tpm_anno, pd.DataFrame(df_temp.iloc[index_to_save]).transpose()])

    def check_if_cellnet_gene_isInGencode(self):
        '''CONDITION 1 above:
        Checking and removing genes in the list not accounted for in the gencode.
        Note that just checking the length is not enough'''
        if not np.array_equal(sorted(self.df_genes_to_plot_tpm_anno["geneSymbol"].tolist()), sorted(self.genes_to_plot)):  # just checking the length is not enough
            genes_notFound = []
            for agene in self.genes_to_plot:
                if not (self.df_genes_to_plot_tpm_anno["geneSymbol"].tolist().__contains__(agene)):
                    genes_notFound.append(agene)

            if (len(genes_notFound) > 0):
                self.logger.warning("The following genes were not found in df_to_plot. (Removing these genes from the list of genes to plot)")
                for i, agene in enumerate(genes_notFound):
                    self.logger.warning("    {}:{}".format(i, agene))
                    self.genes_to_plot.remove(agene)

    def filter_tfs_to_corrs_dict_by_corr_vals(self, tfs_to_corrs_dict, tfs_to_tfsWCorrs_dict, filter_by_corr=True):
        if (filter_by_corr):
            tfs_to_corrs_dict = {k: float(v) for k, v in tfs_to_corrs_dict.items()
                                 if (abs(float(v)) <= self.tf_corr_threshold_high) and
                                 (abs(float(v)) >= self.tf_corr_threshold_low)}
            tfs_to_corrs_sortedDict, tfs_to_tfsWCorrs_sortedDict = self.get_sorted_dicts_one_wVals(tfs_to_corrs_dict)
            return tfs_to_corrs_sortedDict, tfs_to_tfsWCorrs_sortedDict
        return tfs_to_corrs_dict, tfs_to_tfsWCorrs_dict

    def get_tfs_to_corrs_or_tfsWCorrs_dict(self, df_to_plot):
        ''' Return tfs_to_corrs_dict of form: {atf:corr_w_gex_of_this_gene, etc}, and
         tfs_to_tfsWCorrs_sortedDict of form: {atf: atf+" "+corr_w_gex_of_this_gene}.
         For both dicts, the corr values here are sorted in descending order.

        Argument:
        - df_to_plot has this gene_ofInterest and other TFs to plot (as rows)
        '''
        corrs = []
        for agene in self.genes_to_plot:
            if (self.take_pearsonr):
                try:
                    corr = np.corrcoef(df_to_plot.loc[self.genes_to_plot[0]].tolist(), df_to_plot.loc[agene].tolist())[0, 1]
                    corrs.append(corr)
                except:
                    self.logger.info("DEBUG: ")
                    self.logger.info("  agene:{}".format(agene))
                    self.logger.info("  df_to_plot.index".format(df_to_plot.index))
                    self.logger.info("  self.genes_to_plot".format(self.genes_to_plot))
                    pdb.set_trace()
                    self.logger.info("  x:{}, y:{}".format(df_to_plot.loc[self.genes_to_plot[0]].tolist(), df_to_plot.loc[agene].tolist()))

                    raise Exception()
                    # pdb.set_trace()
            else:
                corr = scipy.stats.spearmanr(df_to_plot.loc[self.genes_to_plot[0]].tolist(), df_to_plot.loc[agene].tolist())[0]
                corrs.append(corr)

        tfs_to_corrs_dict = dict(zip(self.genes_to_plot, corrs))
        tfs_to_corrs_dict = {k: float(v) for k, v in tfs_to_corrs_dict.items() if (k != self.gene_ofInterest)}

        tfs_to_corrs_sortedDict, tfs_to_tfsWCorrs_sortedDict = self.get_sorted_dicts_one_wVals(tfs_to_corrs_dict)
        return tfs_to_corrs_sortedDict, tfs_to_tfsWCorrs_sortedDict

    def get_sorted_dicts_one_wVals(self, adict):
        '''Given adict of {k:v}, get one dict sorted by vals v, and
        another of form: {"k":"k v"} also sorted by vals.
        '''
        adict_sorted = col.OrderedDict()
        adict_wVals_sorted = col.OrderedDict()

        adict_as_sortedList = sorted(adict.items(), reverse=True, key=lambda x: x[1])
        for atf, acorr in adict_as_sortedList:
            adict_sorted[atf] = adict[atf]
            adict_wVals_sorted[atf] = "{0} {1:.2f}".format(atf, acorr)
        return adict_sorted, adict_wVals_sorted

    def plot_final_df(self, df_to_plot, plot_title):
        '''df_to_plot has TFs as rows and samples as columns. Note: The first in the df is the gene_ofInterest though.'''
        if (df_to_plot.shape[0] <= 1):  # i.e. there is just the gene-specific row, and no associated TF for this gene
            return
        for use_vmax in [50]:
            if (len(self.genes_to_plot) >= 20):
                plt.figure(figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
            else:
                plt.figure(figsize=(11, 10), dpi=80, facecolor='w', edgecolor='k')

            sns.set(font_scale=1)
            try:
                sns.heatmap(df_to_plot.transpose(), vmax=use_vmax, yticklabels=True)
            except:
                pdb.set_trace()
                self.logger.debug("{}".format(df_to_plot.head(2)))
                sns.heatmap(df_to_plot.set_index(["geneSymbol"]).transpose(), vmax=use_vmax, yticklabels=True)
                raise Exception()
            plt.xticks(rotation=75)
            plt.yticks(rotation=0)
            plt.title(plot_title)
            plt.tight_layout()
            plt.savefig("{0}/{1} {2}_vmax{3}.pdf".format(self.outputDir, self.gene_ofInterest, plot_title, use_vmax))
            plt.close()
