'''
Created on Jan 1, 2018

__author__ = "Dinesh Manandhar"

'''
import pdb
import re
import os
import random
import logging
import numpy as np
import pandas as pd
from pybedtools import BedTool as bedtools
# default bedtools column names: chrom, start, stop, name, score and strand.

pd.set_option('mode.chained_assignment', None)  # default='warn'; set to "None" to ignore warnings
pd.options.display.max_rows = 100
pd.options.display.max_columns = 20  # default is 20
pd.set_option('expand_frame_repr', False)


class Global_Vars(object):
    '''Here, we:
    1. load the rnase df, and get gene_ofInterest_info (goi).
    2. get the region of interest (roi), dhss with signal in roi (df_dhss).
    3. get the df with tfs (from cellnet) with their pcc and zscore info (df_tfs).
    4. get random df_dhss for this gene, - only take_this_many_top_dhs_fts are selcted
    5. get random df_tfs for this gene - PCC >= 0.3 is used to select random TFs

    To do:
    - add known enhancers to the df (also add "has_known enhancers" col).
    '''

    def __init__(self, args, new_out_dir):
        ######################################################
        ###### set the logging handlers and params ######
        formatter = logging.Formatter('%(asctime)s: %(name)-12s: %(levelname)-8s: %(message)s')

        file_handler = logging.FileHandler(os.path.join(new_out_dir, args.gene.upper() + '.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.info("Setting up the DNase-seq dataframe and gene expression vector..")
        ######################################################
        ###### set up the basic args variables ######

        self.gene_ofInterest = args.gene.upper()
        self.dist_lim = args.distance  # in kb
        self.use_tad_info = args.use_tad_info
        self.pcc_lowerlimit_to_filter_dhss = args.pcc_lowerlimit_to_filter_dhss
        self.take_log2_tpm = args.take_log2_tpm
        self.filter_tfs_by = args.filter_tfs_by
        self.lowerlimit_to_filter_tfs = args.lowerlimit_to_filter_tfs
        if (self.filter_tfs_by == "zscore"):
            self.pcc_lowerlimit_to_filter_tfs = 0.1
        else:
            self.pcc_lowerlimit_to_filter_tfs = self.lowerlimit_to_filter_tfs
        self.take_this_many_top_fts = args.take_this_many_top_fts  # all dhss/tfs will already be filtered by pcc(or zscore)
        self.init_wts_type = args.init_wts_type
        self.inputDir = os.path.abspath("../../Input_files")
        self.outputDir = new_out_dir
        self.use_random_DHSs = args.use_random_DHSs
        self.use_random_TFs = args.use_random_TFs

        ####### other variables specific to NN #######
        self.max_iter = args.max_iter

        ######################################################
        ###### read and set up the basic data frames ######
        # self.csv_enhancer_tss = os.path.join(self.inputDir, "enhancer_tss_associations.bed")

        self.csv_dhss = os.path.join(self.inputDir, "roadmap.dnase_imputed.merged_by_samplesAndBedTools.pval.signal.txt")
        self.csv_rnase = os.path.join(self.inputDir, "roadmap.rnase_imputed.LogRPKM.signal.mergedWTADlocs.txt")
        self.csv_cn = os.path.join(self.inputDir, "cnProc_ctGRNs_overallGRN_grnTable.csv")  # "Human_Big_GRN_032014.csv")

        df_dhss = pd.read_csv(self.csv_dhss, sep="\t", index_col="loc")  # final df will be named self.df_dhss
        df_rnase = pd.read_csv(self.csv_rnase, sep="\t", index_col=["geneName", "loc", "TAD_loc"])
        df_cnTfs = pd.read_csv(self.csv_cn, sep=",", header=0)
        ######################################################
        ###### Get the goi, roi, df_roi_dhss and df_tfs objects (See description above __init__().) ######

        self.goi = self.get_gene_ofInterest_info(df_rnase)
        if (self.take_log2_tpm):
            self.goi = np.log2(self.goi + 1)

        self.roi = self.get_roi(self.goi)  # need self.goi to get gene tss loc from goi.index
        df_roi_dhss = self.get_df_dhss(self.roi, df_dhss)  # df_dhss overlapping self.roi
        self.df_dhss = self.filter_ftsIn_multiIndexed_df_by_pcc(df_roi_dhss)
        if (self.use_random_DHSs):
            self.df_dhss = self.get_random_df_dhss_filtdBy_pcc_and_size(
                df_dhss, max_dhs_num=self.df_dhss.shape[0])

        df_tfs = self.get_df_tfs(df_rnase, df_cnTfs)  # tf gexes are log-transformed before getting pccs
        self.df_tfs = self.filter_tf_fts(df_tfs)
        if (self.use_random_TFs):
            self.df_tfs = self.get_random_df_tfs_filtdBy_pcc_and_size(
                df_cnTfs, df_rnase, max_tfs_num=self.df_tfs.shape[0])

        self.logger.info("Done. Setting up the training and testing split..")
        ################## end of __init__() ######################

    '''Load the rnase df, get gene_ofInterest_info (goi)'''

    def get_gene_ofInterest_info(self, df_rnase):
        '''Note: The output "self.gene_ofInterest_info.name" is a tuple of form:
        (geneName, loc, tad_loc). The output will be abbreviated as "goi" hence-forth.
        The gex values are logged if self.take_log2_tpm == True.'''
        df_gene_ofInterest = df_rnase.iloc[df_rnase.index.get_level_values("geneName") == self.gene_ofInterest]
        if (df_gene_ofInterest.shape[0] == 0):
            self.logger.error("The gene name is not found. Double check the name.")
            raise Exception("The gene name is not found. Double check the name?")
        gene_ofInterest_info = df_gene_ofInterest.iloc[0]
        return gene_ofInterest_info

    '''Get chromosomal region of interest(or roi).'''

    def get_roi(self, gene_ofInterest_info):
        tss_ss_es = re.split(":|-", gene_ofInterest_info.name[1])[1:]
        tss_pos = int(sum([int(x) for x in tss_ss_es]) / 2)

        tad_chr_ss_es = re.split(":|-", gene_ofInterest_info.name[2])
        roi_chr = tad_chr_ss_es[0]
        tad_ss, tad_es = [int(x) for x in tad_chr_ss_es[1:]]

        roi_upstream = max(tss_pos - (self.dist_lim * 1000), tad_ss)
        roi_downstream = min(tss_pos + (self.dist_lim * 1000), tad_es)

        return roi_chr, roi_upstream, roi_downstream

    '''Given roi loc and df_dhss, get only those dhss that overlap it.
    This function takes a few seconds to finish.
    Arguments:
        df_dhss is the original dhs with loc of chr:ss-es as index and
        sample values as cols across.
    '''

    def get_df_dhss(self, roi_loc, df_dhss):
        roi_chr, roi_upstream, roi_downstream = roi_loc
        bed_roi = bedtools("{}\t{}\t{}".format(roi_chr, roi_upstream, roi_downstream), from_string=True)

        '''Get dhs locs only dataframe/bed object to do bedintersect with roi.'''
        dhs_locs = [re.split(":|-", x) for x in df_dhss.index.tolist()]
        dhs_locs = [[x[0], int(x[1]), int(x[2])] for x in dhs_locs]
        df_dhs_locs = pd.DataFrame.from_records(dhs_locs, columns=["chrom", "ss", "es"])
        bed_dhs_locs = bedtools.from_dataframe(df_dhs_locs)

        '''Get roi_dhss_locs - i.e. a list with elements chr:ss-es corresponding to dhs sites'''
        bed_roi_dhss_locs = bed_roi.intersect(bed_dhs_locs, wb=True)
        df_roi_dhss_locs = pd.read_table(bed_roi_dhss_locs.fn, names=["chr_int", "ss_int", "es_int", "chr_dhs", "ss_dhs", "es_dhs"])
        df_roi_dhss_locs = df_roi_dhss_locs[["chr_dhs", "ss_dhs", "es_dhs"]]
        roi_dhss_locs = ["{}:{}-{}".format(x[0], str(x[1]), str(x[2])) for x in
                         zip(df_roi_dhss_locs["chr_dhs"], df_roi_dhss_locs["ss_dhs"], df_roi_dhss_locs["es_dhs"])]
        df_roi_dhss = df_dhss[df_dhss.index.isin(roi_dhss_locs)]

        '''Add PCC(dhs, goi) to the index'''
        pccs = []
        for ix in range(0, df_roi_dhss.shape[0]):
            pccs.append(np.corrcoef(df_roi_dhss.iloc[ix], self.goi)[0, 1])
        df_roi_dhss["pcc"] = pccs
        df_roi_dhss = df_roi_dhss.set_index("pcc", append=True)

        return df_roi_dhss

    '''Filter a multilevel indexed df by its pcc index value in 2 steps:
    1. By abs_pcc: ignore fts with low pcc.
    2. By take_this_many_top_fts: only fts with topmost pccs selected.'''

    def filter_ftsIn_multiIndexed_df_by_pcc(self, df):
        df["abs_pcc"] = abs(df.index.get_level_values("pcc"))
        df = df.sort_values(by=["abs_pcc"], ascending=False)
        df = df[df["abs_pcc"] >= self.pcc_lowerlimit_to_filter_dhss]  # filter by abs_pcc
        df = df.drop(labels=["abs_pcc"], axis=1)
        if (df.shape[0] > self.take_this_many_top_fts > 0):
            df = df[:self.take_this_many_top_fts]  # this df is sorted by abs_pcc (4 steps back)
        return df

    '''Get df_tfs. The csv_cn_tfs file is read fast.'''

    def get_df_cn_tfs(self, df_cnTfs):
        '''df_tfs will have following original columns:
        TG TF zcores corr type species'''
        df_cnTfs = df_cnTfs[df_cnTfs["TG"] == self.gene_ofInterest]
        df_cnTfs = df_cnTfs.drop_duplicates(subset="TF", keep="first")  # some TFs could be present in >1 row
        df_cnTfs = df_cnTfs[["TF", "zscore", "corr"]]
        df_cnTfs.columns = ["geneName", "zscore", "cn_corr"]  # "geneName" will be used to merge with df_tfs later
        return df_cnTfs.set_index('geneName')

    def get_df_tfs(self, df_rnase, df_cnTfs):
        '''df_tfs has "geneName", "loc", "TAD_loc", "zscore", "cn_corr" and "pcc" as index.
        The cols are gexes in samples / cell types. The TFs (i.e. "geneName") are filtered
        by self.filter_tfs_by argument (i.e. "zscore" or "pearson_corr") threshold and
        subsequently by self.take_this_many_top_fts on the same argument.
        '''
        df_cnTfs = self.get_df_cn_tfs(df_cnTfs)  # has zscores and cn_corr as cols, and "geneName" (i.e. TFs) as indices
        df_tfs = df_rnase.iloc[df_rnase.index.get_level_values("geneName").isin(df_cnTfs.index)]
        if (self.take_log2_tpm):
            df_tfs = np.log2(df_tfs + 1)

        '''First get the pcc values. Then merge the zscores and corr cols df and the pccs'''
        pccs = []
        for ix in range(0, df_tfs.shape[0]):  # note df_cnTfs has geneNames sorted as in df_tfs.
            pccs.append(np.corrcoef(df_tfs.iloc[ix], self.goi)[0, 1])

        '''Merge df_cnTfs with df_tfs. Add zscore, cn_corr and pcc cols to the merged df'''
        df_tfs = df_cnTfs.join(df_tfs, how='inner')
        df_tfs["pcc"] = pccs
        df_tfs = df_tfs.set_index(["zscore", "cn_corr", "pcc"], append=True)

        return df_tfs

    '''Filter by zcore/pcc lower limit first, then filter again to get only top fts'''

    def filter_tf_fts(self, df_tfs):
        if (self.filter_tfs_by == "zscore"):  # zscores are all positive; sorting and filtering is not complicated.
            df_tfs = df_tfs[df_tfs.index.get_level_values("zscore") >= self.lowerlimit_to_filter_tfs]
            if (df_tfs.shape[0] > self.take_this_many_top_fts > 0):  # self.take_this_many_top_fts is set to -1 if all fts are to be used
                df_tfs = df_tfs.sort_index(axis=0, level="zscore", ascending=False)[:self.take_this_many_top_fts]
        else:
            df_tfs = self.filter_ftsIn_multiIndexed_df_by_pcc(df_tfs)
        return df_tfs

    '''Return a random df of dhss filtered only by the self.pcc_lowerlimit_to_filter_dhss argument.
    This will be used later to further randomly select dhss using the argument self.take_this_many_top_fts.
    '''

    def get_random_df_dhss_filtdBy_pcc(self, df_dhss, starting_num_dhss=1000):
        rand_ints = sorted(random.sample(range(0, df_dhss.shape[0]), starting_num_dhss))
        df_random = df_dhss.iloc[rand_ints, :]

        pccs = []
        for ix in range(0, df_random.shape[0]):
            pccs.append(np.corrcoef(df_random.iloc[ix], self.goi)[0, 1])

        df_random["pcc"] = pccs
        df_random["abs_pcc"] = [abs(x) for x in pccs]
        df_random = df_random.sort_values(by=["abs_pcc"], ascending=False)
        df_random = df_random[df_random["abs_pcc"] >= self.pcc_lowerlimit_to_filter_dhss]  # filter by abs_pcc

        return df_random

    '''Return a random df_dhss filtered by both self.pcc_lowerlimit_to_filter_dhss and
    the size of the dhss df to get. The size is the max_dhs_num, which
    is the number of real dhss being considered. Note that this already is the minimum
    total DHS sites in ROI and self.take_this_many_dhss_fts argument.'''

    def get_random_df_dhss_filtdBy_pcc_and_size(self, df_dhss, max_dhs_num):
        # Note that self.goi is not yet log-transformed
        df_random = self.get_random_df_dhss_filtdBy_pcc(df_dhss, starting_num_dhss=1000)
        while (df_random.shape[0] < max_dhs_num):  # which is highly unlikely (given starting_num_dhss is set high)
            df_random = pd.concat([df_random, self.get_random_df_dhss_filtdBy_pcc(df_dhss, starting_num_dhss=500)], axis=0)
            df_random = df_random.drop_duplicates()

        rand_ints = sorted(random.sample(range(0, df_random.shape[0]), max_dhs_num))
        df_random = df_random.iloc[rand_ints, :]
        df_random = df_random.drop(["abs_pcc"], axis=1)
        df_random = df_random.set_index("pcc", append=True)
        return df_random

    '''Function similar to self.get_random_df_dhss_filtdBy_pcc_and_size() above, but for TFs.
    max_tfs_num is the size of the real number of TFs being considered. Also, note that
    only those random TFs that pass the self.pcc_lowerlimit_to_filter_tfs threshold are selected.'''

    def get_random_df_tfs_filtdBy_pcc_and_size(self, df_cnTfs, df_rnase, max_tfs_num):
        # Note that self.goi is not yet log-transformed

        all_tfs = list(set(df_cnTfs["TF"]))
        df_random = df_rnase[df_rnase.index.get_level_values("geneName").isin(all_tfs)]

        pccs = []
        for ix in range(0, df_random.shape[0]):
            pccs.append(np.corrcoef(df_random.iloc[ix], self.goi)[0, 1])

        df_random["pcc"] = pccs
        df_random["abs_pcc"] = [abs(x) for x in pccs]
        df_random = df_random[df_random["abs_pcc"] >= self.pcc_lowerlimit_to_filter_tfs]
        rand_ints = sorted(random.sample(range(0, df_random.shape[0]), max_tfs_num))
        df_random = df_random.iloc[rand_ints, :]
        df_random = df_random.drop(["abs_pcc"], axis=1)
        df_random = df_random.set_index("pcc", append=True)
        return df_random


if __name__ == "__main__":
    class Args(object):
        def __init__(self):
            self.gene = "PRDM2"
            self.distance = 150
            self.use_tad_info = True
            self.pcc_lowerlimit_to_filter_dhss = 0.1
            self.take_log2_tpm = True
            self.filter_tfs_by = "zscore"  # or "pcc"
            self.lowerlimit_to_filter_tfs = 5.0
            self.take_this_many_top_fts = 15  # all dhss/tfs will already be filtered by pcc(or zscore)
            self.init_wts_type = "corr"
            self.outputDir = "/Users/Dinesh/Dropbox/Github/predicting_gex_with_nn_git/Output"
            self.use_random_DHSs = False
            self.use_random_TFs = False
            self.max_iter = 300

    args = Args()
    gv = Global_Vars(args, args.outputDir)
