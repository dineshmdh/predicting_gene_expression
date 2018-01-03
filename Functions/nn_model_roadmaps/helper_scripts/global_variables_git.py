'''
Created on Jan 2, 2018

A class for global variables
  - will contain all variables + bed and dfs that will be used downstream for training.  - doing
  - should be able to load enhancers for the gene right in this  - done
  - should be able to process cell net TFs and return the dfs right in this / helper function
  - will generate all the plots that i should be able to get when i call gv.plots.<name> etc.


To update later:
1. Generate a df_allgenes_tpm_anno file -- so that I don't merge the two files using
get_full_tpm_info_df(csv_tpm_merged, csv_gencode_mrnas) static function.
2. Remove redundancy also in having to find tad index first. make a tall format file
(so i don't have to go through the for loop searching for the tad file)
3. check chromosome boundaries in self.get_true_roi_dhs_df() function. Also, make this function independent
of tad info.
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
from pybedtools import BedTool as bedtools  # default bedtools column names: chrom, start, stop, name, score and strand.

from helper_functions import get_genome_wide_enhancers_df, get_full_tpm_info_df, get_dhss_all_df

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

        self.csv_gencode_mrnas = os.path.join(self.inputDir, "gencode.v19.annotation.genesOnly.reOrdered.txt")
        self.csv_tads_selected = os.path.join(self.inputDir, "df_tads_all.csv")

        self.csv_dhs_normed = os.path.join(self.inputDir, "dnaseseq_df_tpmNormed.txt")
        self.csv_tpm_merged = os.path.join(self.inputDir, "merged_tpms.tsv")
        self.csv_enhancer_tss = os.path.join(self.inputDir, "enhancer_tss_associations.bed")

        assert os.path.exists(self.csv_gencode_mrnas)
        assert os.path.exists(self.csv_tads_selected)
        assert os.path.exists(self.csv_dhs_normed)
        assert os.path.exists(self.csv_tpm_merged)
        assert os.path.exists(self.csv_enhancer_tss)

        '''Get genome-wide tpm and dhs signal info, and enhancer info - using helper functions'''
        df_gencode_all_mrnas = pd.read_csv(self.csv_gencode_mrnas, sep="\t",
                                           names=["chr_gene", "ss_gene", "es_gene", "strand_gene",
                                                  "geneSymbol", "gene_id", "anno", "gtf_line"])
        self.df_gencode_all_mrnas = df_gencode_all_mrnas.drop(["anno", "gtf_line"], axis=1)
        self.df_allgenes_tpm_anno = get_full_tpm_info_df(self.csv_tpm_merged, self.df_gencode_all_mrnas)
        self.df_all_dhss_normed = get_dhss_all_df(self.csv_dhs_normed, self.logger)
        self.df_enh_tsss_info = get_genome_wide_enhancers_df(self.csv_enhancer_tss, self.logger)

        '''Get info series on this gene - even if self.use_tad_info is False.'''
        self.gene_ofInterest_info = self.df_allgenes_tpm_anno[self.df_allgenes_tpm_anno["geneSymbol"] == self.gene_ofInterest].iloc[0]

        '''Get enhancers for this gene'''
        self.df_enh_thisgene_info = self.df_enh_tsss_info[self.df_enh_tsss_info["gene"] == self.gene_ofInterest]

        if (self.df_enh_thisgene_info.shape[0] == 0):
            self.logger.info("{} has no predicted enhancer.".format(self.gene_ofInterest))

        '''Get dfs + info specific to this TAD'''
        self.tad_index = None
        self.genes_in_this_tad = None
        self.df_tpms_anno_this_tad = None
        self.df_dhss_this_tad = None

        if (self.use_tad_info):
            self.tad_index, self.genes_in_this_tad, self.df_tads_selected = self.get_tad_index_for_this_gene()
            # self.genes_in_this_tad here is a list of ensemble ids for the genes in this TAD
            self.df_tpms_anno_this_tad = self.df_allgenes_tpm_anno[self.df_allgenes_tpm_anno["gene_id"].isin(self.genes_in_this_tad)]
            self.plot_tpm_matrix_for_this_tad()
            self.df_this_tad, self.df_dhss_this_tad = self.get_dhs_df_for_this_tad()
            self.plot_dhs_matrix_for_this_tad()

        if not (self.use_random_DHSs):
            self.df_dhss_in_roi = self.get_true_roi_dhs_df()  # checks for self.use_tad_info; roi == region of interest
            self.update_roi_dhs_df_with_enhancers()  # update self.df_dhss_in_roi with any enhancers outside of roi; adds has_knownEnhancers field
        else:
            self.df_dhss_in_roi = self.get_random_roi_dhs_df()
            self.df_dhss_in_roi["has_knownEnhancers"] = 0  # just to match the format for when orig DHSs are used

    def __repr__(self):
        out_info = "\nGeneral Info: " + self.gene_ofInterest
        if (self.use_tad_info):
            out_info += "\nTAD index: " + str(self.tad_index)
        else:
            out_info += "\nTAD info not used."
        out_info += "\nTotal number of genes in this TAD: " + str(len(self.genes_in_this_tad))
        return out_info

    def get_out_suffix(self):
        '''Get suffix for all output files to be saved.'''
        out_suffix = ""
        if (self.use_random_wts):
            out_suffix += "_wRndmWts"
        elif (self.use_corr_wts):
            out_suffix += "_wCorrWts"
        else:
            out_suffix += "_wExpDecayWts"

        if (self.take_top_fts):
            out_suffix += "_on" + str(self.take_this_many_top_fts) + "DHSs"

        out_suffix += "_T" + str(self.tf_corr_threshold_high)
        out_suffix += "_t" + str(self.tf_corr_threshold_low)
        out_suffix += "_at" + str(self.dist_lim) + "kb" + ["_tpmLogged" if self.take_log2_tpm else ""][0]
        return out_suffix

    def get_tad_index_for_this_gene(self):
        '''Return tad_index for the gene_ofInterest in the file:
        self.csv_tads_selected. Also return the ensemble gene name
        for this gene_ofInterest.
        '''
        st = time.time()
        df_tads_selected = pd.read_csv(self.csv_tads_selected, sep=",", header=0)
        # has fields: chrom_tad start_tad stop_tad numGenes genes (genes has the Ensemble Names in a list)

        # get the gene_ensemble_name
        gene_ens_name = self.df_gencode_all_mrnas[self.df_gencode_all_mrnas["geneSymbol"] == self.gene_ofInterest.upper()]["gene_id"]
        gene_ens_name = re.split("\s+", gene_ens_name.to_string())[1]

        # Now search for the tad index'''
        for i in range(0, df_tads_selected.shape[0]):
            genes_in_this_tad = re.split("', '", df_tads_selected.iloc[i]["genes"][2:-2])
            if gene_ens_name in genes_in_this_tad:
                tad_index_selected = df_tads_selected.index[i]
                break
        else:
            raise Exception("TAD index not found for this gene..")
        self.logger.debug("Time taken for get_tad_index_for_this_gene(): {}".format(time.time() - st))
        return tad_index_selected, genes_in_this_tad, df_tads_selected

    def get_dhs_df_for_this_tad(self):
        '''Get df_dhss with dnase signal across samples for self.tad_index.
        This function generates bedtools object and does intersect; these steps take some time.'''
        st = time.time()

        bed_alldhss_normed = bedtools.from_dataframe(self.df_all_dhss_normed)
        self.logger.debug("    Time taken for getting bed_alldhss_normed: {}".format(time.time() - st))

        # Similarly, get the bed object for this tad
        series_this_tad = self.df_tads_selected.iloc[self.tad_index]
        df_this_tad = series_this_tad.to_frame().transpose()
        bed_this_tad = bedtools.from_dataframe(df_this_tad)
        self.logger.debug("    Time taken for getting bed_this_tad (from start of function): {}".format(time.time() - st))

        # Intersect
        bed_thistad_dhss = bed_this_tad.intersect(bed_alldhss_normed, wb=True)
        self.logger.debug("    Time taken for intersecting (from start of function):".format(time.time() - st))

        # convert this bed to df, and remove the tad domain specific columns
        df_thistad_dhss = pd.read_table(bed_thistad_dhss.fn,
                                        names=df_this_tad.columns.tolist() + self.df_all_dhss_normed.columns.tolist()).iloc[:, 5:]
        self.logger.debug("Time taken for get_dhs_df_for_this_tad(): {}".format(time.time() - st))
        return df_this_tad, df_thistad_dhss

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

        # pdb.set_trace()

        rand_ints = sorted(random.sample(range(0, self.df_all_dhss_normed.shape[0]), self.take_this_many_top_fts))
        df_dhss_in_roi = self.df_all_dhss_normed.loc[rand_ints, :]

        # pdb.set_trace()

        df_dhss_in_roi.sort_values(by=["chr_dhs", "ss_dhs"], axis=0, ascending=True, inplace=True)
        return df_dhss_in_roi

    def plot_tpm_matrix_for_this_tad(self):
        '''Plot gene_symbols by cell lines heatmap.'''

        df = self.df_tpms_anno_this_tad[["geneSymbol"] + self.df_tpms_anno_this_tad.columns.tolist()[7:]]
        df.set_index(["geneSymbol"], inplace=True)

        if (df.shape[0] < 30):
            sns.set(font_scale=1.5)
            plt.figure(figsize=(14, 6))
        else:
            sns.set(font_scale=1.3)
            plt.figure(figsize=(14, 14))

        sns.heatmap(df, vmax=5, square=False, xticklabels=False, cbar_kws={"label": "TPM"})
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.xlabel("Cell lines")
        plt.ylabel("Gene Symbols")
        plt.savefig(self.outputDir + "/" + self.gene_ofInterest + "_tad_tpm_matrix.png")
        plt.close()

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
