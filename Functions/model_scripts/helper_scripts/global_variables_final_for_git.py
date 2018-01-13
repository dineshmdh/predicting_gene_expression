import re
import os
import random
import logging
import numpy as np
import pandas as pd
from pybedtools import BedTool as bedtools
# default bedtools column names: chrom, start, stop, name, score and strand.

pd.options.mode.chained_assignment = 'warn'  # default='warn'; set to "None" to ignore warnings
pd.options.display.max_rows = 100
pd.options.display.max_columns = 20  # default is 20
pd.set_option('expand_frame_repr', False)


class Global_Vars(object):
    '''Here, we:
    1. load the rnase df, and get gene_ofInterest_info (goi).
    2. get the region of interest (roi), dhss with signal in roi (df_dhss),
    add known enhancers to the df (also add "has_known enhancers" col).
    3. get the df with tfs (from cellnet) with their pcc and zscore info (df_tfs).

    Note that we do not transform data in any way (eg. by taking log of tpm)

    To do:
    1. add known enhancers to df_roi_dhss (also add "has_known enhancers" col),
    3. get random df_dhss for this gene, - only take_this_many_top_dhs_fts are selcted
    4. get random df_tfs for this gene.
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
        self.take_this_many_top_fts = args.take_this_many_top_fts  # all dhss/tfs will already be filtered by pcc(or zscore)
        self.init_wts_type = args.init_wts_type
        self.inputDir = os.path.abspath("../../Input_files")
        self.outputDir = new_out_dir
        self.use_random_DHSs = args.use_random_DHSs
        self.use_random_TFs = args.use_random_TFs
        ######################################################
        ###### read and set up the basic data frames ######
        # self.csv_enhancer_tss = os.path.join(self.inputDir, "enhancer_tss_associations.bed")

        self.csv_dhss = os.path.join(self.inputDir, "roadmap.dnase_imputed.merged_by_samplesAndBedTools.pval.signal.txt")
        self.csv_rnase = os.path.join(self.inputDir, "roadmap.rnase_imputed.LogRPKM.signal.mergedWTADlocs.txt")
        self.csv_cn = os.path.join(self.inputDir, "Human_Big_GRN_032014.csv")

        '''Get the goi, roi, df_roi_dhss and df_tfs objects
        (See description above __init__().)'''
        df_rnase = pd.read_csv(self.csv_rnase, sep="\t", index_col=["geneName", "loc", "TAD_loc"])
        self.goi = self.get_gene_ofInterest_info(df_rnase)
        self.roi = self.get_roi(self.goi)  # need self.goi to get gene tss loc from goi.index

        df_dhss = self.get_df_dhss(self.roi)
        self.df_dhss = self.filter_ftsIn_multiIndexed_df_by_pcc(df_dhss)
        if (self.use_random_DHSs):
            self.df_dhss = self.get_random_dhs_df()  # dhss could be from different chromosomes

        df_tfs = self.get_df_tfs(df_rnase)
        self.df_tfs = self.filter_tf_fts(df_tfs)
        if (self.use_random_TFs):
            self.df_tfs = self.get_random_tfs_df()
        self.logger.info("Done. Setting up the training and testing split..")
        ################## end of __init__() ######################

    '''Load the rnase df, get gene_ofInterest_info (goi)'''

    def get_gene_ofInterest_info(self, df_rnase):
        '''Note: The output "self.gene_ofInterest_info.name" is a tuple of form:
        (geneName, loc, tad_loc). The output will be abbreviated as "goi" hence-forth.
        The gex values are logged if self.take_log2_tpm == True.'''
        df_gene_ofInterest = df_rnase.iloc[df_rnase.index.get_level_values("geneName") == self.gene_ofInterest]
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

    '''Given roi loc, get only those dhss that overlap it.
    This function takes a few seconds to finish.'''

    def get_df_dhss(self, roi_loc):
        roi_chr, roi_upstream, roi_downstream = roi_loc
        bed_roi = bedtools("{}\t{}\t{}".format(roi_chr, roi_upstream, roi_downstream), from_string=True)

        '''Get dhs locs only dataframe/bed object to do bedintersect with roi.'''
        df_dhss = pd.read_csv(self.csv_dhss, sep="\t", index_col="loc")  # df_dhss is the original dhs with loc of chr:ss-es as index and sample values as cols across
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

    def get_df_cn_tfs(self):
        '''df_tfs will have following original columns:
        TG TF zcores corr type species'''
        df_cnTfs = pd.read_csv(self.csv_cn, sep=",", header=0)
        df_cnTfs = df_cnTfs[df_cnTfs["TG"] == self.gene_ofInterest]
        df_cnTfs = df_cnTfs.drop_duplicates(subset="TF", keep="first")  # some TFs could be present in >1 row
        df_cnTfs = df_cnTfs[["TF", "zscore", "corr"]]
        df_cnTfs.columns = ["geneName", "zscore", "cn_corr"]  # "geneName" will be used to merge with df_tfs later
        return df_cnTfs.set_index('geneName')

    def get_df_tfs(self, df_rnase):
        '''df_tfs has "geneName", "loc", "TAD_loc", "zscore", "cn_corr" and "pcc" as index.
        The cols are gexes in samples / cell types. The TFs (i.e. "geneName") are filtered
        by self.filter_tfs_by argument (i.e. "zscore" or "pearson_corr") threshold and
        subsequently by self.take_this_many_top_fts on the same argument.
        '''
        df_cnTfs = self.get_df_cn_tfs()  # has zscores and cn_corr as cols, and "geneName" (i.e. TFs) as indices
        df_tfs = df_rnase.iloc[df_rnase.index.get_level_values("geneName").isin(df_cnTfs.index)]

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

    '''If random DHSs are to be selected, only select a random
    collection of self.take_this_many_top_dhs_fts from the genome.'''

    def get_random_roi_dhs_df(self):
        df_dhss = pd.read_csv(self.csv_dhss, sep="\t", index_col="loc")  # df_dhss is the original dhs with loc of chr:ss-es as index and sample values as cols across
        rand_ints = sorted(random.sample(range(0, df_dhss.shape[0]), self.take_this_many_top_dhs_fts))  # sorted() will yield sorted list of dhss also
        rand_locs = [df_dhss.index[x] for x in rand_ints]
        df_dhss_random = df_dhss[df_dhss.index.isin(rand_locs)]
        return df_dhss_random

    def get_random_tfs_df(self):
        pass


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
    gv = Global_Vars(args)
