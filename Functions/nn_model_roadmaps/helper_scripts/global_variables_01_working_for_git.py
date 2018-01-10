'''
Created on Jan 2, 2018

A class for global variables
  - will contain all variables + bed and dfs that will be used downstream for training.
  - should be able to load enhancers for the gene right in this
  - should be able to process cell net TFs and return the dfs right in this / helper function
  - will generate all the plots that i should be able to get when i call gv.plots.<name> etc.


To update later:

3. check chromosome boundaries in self.get_true_roi_dhs_df() function. Also, make this function independent of tad info.
4. ModelPreparation.tf_corr_threshold_low variable is not being used at the moment.
5. Make sure that when choosing the TFs, the PCC used to choose the TFs with high PCC are the absolute PCCs.
6. Add noise to the correlation while initializing the weights w1 and w2.

Just in case:
cbar_kws={"orientation": "horizontal", "pad": 0.25, "shrink": 0.3, "aspect": 25, "label": "TPM"})

'''
import pdb
import re
import os
import logging
import time
import random

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns

import pandas as pd
from pybedtools import BedTool as bedtools
# default bedtools column names: chrom, start, stop, name, score and strand.

from helper_functions import get_genome_wide_enhancers_df
from helper_functions import get_full_tpm_info_df
from helper_functions import get_dhss_all_df

pd.options.mode.chained_assignment = None  # default='warn' (see README.md)

pd.options.display.max_rows = 100
pd.options.display.max_columns = 20  # default is 20
pd.set_option('expand_frame_repr', False)

plt.rcParams['figure.figsize'] = (12.0, 12.0)
plt.rcParams['font.size'] = 10
plt.rcParams['savefig.dpi'] = 120
plt.rcParams['figure.subplot.bottom'] = 0.1


class Global_Vars(object):

    def __init__(self, args, outputDir):

        ######################################################
        ###### set the logging handlers and params ######
        formatter = logging.Formatter('%(asctime)s: %(name)-12s: %(levelname)-8s: %(message)s')

        file_handler = logging.FileHandler(os.path.join(outputDir, args.gene.upper() + '.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.info("Setting up the Global_Vars instance..")
        ######################################################
        ###### set up the basic args variables ######

        self.gene_ofInterest = args.gene.upper()
        self.dist_lim = args.distance  # in kb
        self.use_tad_info = args.use_tad_info
        self.take_this_many_top_dhs_fts = args.take_this_many_top_dhs_fts
        self.take_log2_tpm = args.take_log2_tpm
        self.lowerlimit_to_filter_tfs = args.lowerlimit_to_filter_tfs
        self.init_wts_type = args.init_wts_type
        self.inputDir = os.path.abspath('../Input_files')
        self.outputDir = outputDir  # generated using  get_output_dir from helper_functions
        self.use_random_DHSs = args.use_random_DHSs
        self.use_random_TFs = args.use_random_TFs
        ######################################################
        ###### read and set up the basic data frames ######
        '''
        self.csv_dhs_normed = os.path.join(self.inputDir, "dnaseseq_df_tpmNormed.txt")
        self.csv_tpm_merged = os.path.join(self.inputDir, "merged_tpms.tsv")
        self.csv_enhancer_tss = os.path.join(self.inputDir, "enhancer_tss_associations.bed")
        '''

        self.csv_dhs_normed = os.path.join(self.inputDir, "roadmap.dnase_imputed.merged_by_samplesAndBedtools.pval.signal")
        self.csv_tpm_merged = os.path.join(self.inputDir, "roadmap.rnase_imputed.LogRPKM.signal.mergedWTADlocs")

        self.df_all_dhss_normed = get_dhss_all_df(self.csv_dhs_normed, self.logger)

        '''Get info series on this gene - even if self.use_tad_info is False.'''
        self.gene_ofInterest_info = self.df_allgenes_tpm_anno[self.df_allgenes_tpm_anno["geneSymbol"] == self.gene_ofInterest].iloc[0]

        if not (self.use_random_DHSs):
            self.df_dhss_in_roi = self.get_true_roi_dhs_df()  # checks for self.use_tad_info; roi == region of interest
            self.update_roi_dhs_df_with_enhancers()  # update self.df_dhss_in_roi with any enhancers outside of roi; adds has_knownEnhancers field
        else:
            self.df_dhss_in_roi = self.get_random_roi_dhs_df()
            self.df_dhss_in_roi["has_knownEnhancers"] = 0  # just to match the format for when orig DHSs are used

    def __repr__(self):
        out_info = "\nGeneral Info: " + self.gene_ofInterest
        out_info += "\nTotal number of genes in this TAD: " + str(len(self.genes_in_this_tad))
        return out_info

    def get_true_roi_dhs_df(self):
        '''Fix the region of interest for DHSs for this gene'''
        basal_upstream = self.dist_lim * 1000  # bp; all dhss upto this distance upstream will be selected
        basal_downstream = self.dist_lim * 1000  # bp ; similar to basal_upstream

        if (not self.use_tad_info):
            raise Exception("update this function to get roi..")

        if (self.gene_ofInterest_info["strand_gene"] == "+"):
            roi_ss = max(self.gene_ofInterest_info["ss_gene"] - basal_upstream,
                         self.df_this_tad.iloc[0]["start_tad"])
            roi_es = min(self.gene_ofInterest_info["es_gene"] + basal_downstream,
                         self.df_this_tad.iloc[0]["stop_tad"])
        else:
            roi_ss = max(self.gene_ofInterest_info["ss_gene"] - basal_downstream,
                         self.df_this_tad.iloc[0]["start_tad"])
            roi_es = min(self.gene_ofInterest_info["es_gene"] + basal_upstream,
                         self.df_this_tad.iloc[0]["stop_tad"])

        bed_roi = bedtools(self.gene_ofInterest_info["chr_gene"] + "\t" + str(roi_ss) + "\t" + str(roi_es), from_string=True)
        bed_dhss_in_roi = bedtools.from_dataframe(self.df_dhss_this_tad).intersect(bed_roi, wa=True)  # yields no bed2 info
        df_dhss_in_roi = pd.read_table(bed_dhss_in_roi.fn, names=self.df_dhss_this_tad.columns.tolist())
        '''Sorting the locations in the df_dhss_in_roi. The df is not sorted otherwise (so coming back to add this). Useful in plotting later.'''
        df_dhss_in_roi.sort_values(by=["ss_dhs"], axis=0, ascending=True, inplace=True)
        return df_dhss_in_roi

    def get_random_roi_dhs_df(self):
        if (not self.use_tad_info):
            raise Exception("update this function to get roi..")

        rand_ints = sorted(random.sample(range(0, self.df_all_dhss_normed.shape[0]), self.take_this_many_top_fts))
        df_dhss_in_roi = self.df_all_dhss_normed.loc[rand_ints, :]

        df_dhss_in_roi.sort_values(by=["chr_dhs", "ss_dhs"], axis=0, ascending=True, inplace=True)
        return df_dhss_in_roi

    def plot_dhs_matrix_for_this_tad(self):
        '''To Plot the dhss_matrix for this tad'''
        sns.set(font_scale=2)
        plt.figure(figsize=(15, 5))
        sns.heatmap(self.df_dhss_this_tad.iloc[:, 3:].transpose(), vmax=10, square=False, xticklabels=False, yticklabels=False)
        plt.xlabel("DHS sites on this TAD")
        plt.ylabel("Cell lines")
        plt.savefig(self.outputDir + "/" + self.gene_ofInterest + "_tad_dhss_matrix.png")
        plt.close()

    def update_roi_dhs_df_with_enhancers(self):
        '''Add any missed enhancer outside of roi (detailed below) and add an "has_knownEnhancer" field.
        A couple of things:
        1. my overall set of DHS sites may not overlap that of FANTOM enhancer list.
        (Although it is unlikely that there exist a fantom enhancer that does not
        overlap a dhs site in any cell line)
        2. The dhs sites between an enhancer and a dhs site will not be precise.
        '''
        self.logger.info("There is a total of {} known enhancers for this gene.".format(self.df_enh_thisgene_info.shape[0]))
        self.logger.info("    Of these, {} overlap in this tad.".format(bedtools.from_dataframe(self.df_this_tad[["chrom_tad", "start_tad", "stop_tad"]]).intersect(bedtools.from_dataframe(self.df_enh_thisgene_info)).count()))

        bed_enhs_accounted_for_inTAD = bedtools.from_dataframe(self.df_dhss_in_roi).intersect(bedtools.from_dataframe(self.df_enh_thisgene_info))
        self.logger.info("    {} are accounted for in this TAD.".format(bed_enhs_accounted_for_inTAD.count()))

        bed_enhs_accountedFor_inThisChr = bedtools.from_dataframe(self.df_all_dhss_normed).intersect(bedtools.from_dataframe(self.df_enh_thisgene_info), wa=True)
        self.logger.info("    And {} are accounted for in this chromosome.".format(bed_enhs_accountedFor_inThisChr.count()))

        '''Adding all the enhancers and removing duplicates'''
        self.df_dhss_in_roi = pd.concat([self.df_dhss_in_roi, pd.read_table(bed_enhs_accountedFor_inThisChr.fn, names=self.df_dhss_in_roi.columns)])
        self.df_dhss_in_roi.drop_duplicates(inplace=True)

        '''Now updating df_dhss_in_roi with 1 (if the dhs overlaps enhancer for this gene) or 0 (else)'''
        self.df_dhss_in_roi["has_knownEnhancer"] = 0
        '''Get the known enhancers for this gene'''
        df_enhs_accountedFor_inThisChr = pd.read_table(bed_enhs_accountedFor_inThisChr.fn, names=self.df_dhss_in_roi.columns).iloc[:, :3]
        '''Make a df_temp to get the indices to update the cell values..'''
        df_temp = self.df_dhss_in_roi[(self.df_dhss_in_roi["chr_dhs"].isin(df_enhs_accountedFor_inThisChr["chr_dhs"])) &
                                      (self.df_dhss_in_roi["ss_dhs"].isin(df_enhs_accountedFor_inThisChr["ss_dhs"])) &
                                      (self.df_dhss_in_roi["es_dhs"].isin(df_enhs_accountedFor_inThisChr["es_dhs"]))]
        for aindex in df_temp.index:
            self.df_dhss_in_roi.loc[aindex, "has_knownEnhancer"] = 1


if __name__ == '__main__':
    kw_dict = {"gene_ofInterest": "MYOG",
               "dist_lim": 200,
               "tf_corr_threshold": 0.3,
               "use_tad_info": True,
               "take_top_fts": True,
               "take_this_many_top_fts": 20,
               "take_log2_tpm": True,
               "use_wts": "random"}
    gv = Global_Vars(**kw_dict)
