'''
Created on January 6, 2017
'''

import numpy as np
import random
import os
import logging
import pandas as pd
from global_variables_final_for_git import Global_Vars

pd.options.mode.chained_assignment = 'warn'  # default='warn' (see README.md)


class Model_preparation(object):
    '''
    The final output of the script is going to be trainX/Y and testX/Y dataframes,
    for dhssOnly, tfsOnly and wtfs models.
    '''

    def __init__(self, gv):

        ######################################################
        # set the logging handlers and params
        formatter = logging.Formatter('%(asctime)s: %(name)-12s: %(levelname)-8s: %(message)s')

        file_handler = logging.FileHandler(os.path.join(gv.outputDir, gv.gene_ofInterest + '.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.info("Preparing the train/test dataf..")
        ######################################################
        # get train and test split (+ merged - for joint) dfs for models

        self.csv_eids = os.path.join(gv.inputDir, "roadmap_EID_info.txt")
        self.train_eids, self.test_eids = self.get_train_and_test_eids(self.csv_eids)  # self.test_eids could be of size 1

        '''Get train and test X matrices for the models.
        Note these X matrices have fts as rows and samples as cols.'''
        self.train_dhss, self.test_dhss = self.get_normalized_train_test_dfs(gv.df_dhss, self.train_eids, self.test_eids)
        self.train_tfs, self.test_tfs = self.get_normalized_train_test_dfs(gv.df_tfs, self.train_eids, self.test_eids)
        self.train_joint = self.merge_dhs_and_tf_dfs(self.train_dhss, self.train_tfs, gv)  # for dhs+tf joint model
        self.test_joint = self.merge_dhs_and_tf_dfs(self.test_dhss, self.test_tfs, gv)

        '''Now get the train and test Y vectors.
        Note these Y "vectors" are pandas series objects.'''
        self.train_goi, self.test_goi = self.get_normalized_train_and_test_goi(gv)

        self.fig_gex_in_train = None  # will be updated once this plot is made
        self.fig_gex_in_test = None  # will be updated later
        ############## end of init() #############

    '''Returns a list of EIDs in train and test samples.'''

    def get_train_and_test_eids(self, csv_eids):
        df_eids = pd.read_csv(csv_eids, sep="\t", header=0)
        train_eid_groups, test_eid_group = self.get_train_and_test_eid_groups(df_eids)
        train_eids = df_eids[df_eids["GROUP"].isin(train_eid_groups)]["EID_info"].tolist()
        test_eids = df_eids[df_eids["GROUP"] == test_eid_group]["EID_info"]
        if (len(test_eids) == 1):
            test_eids = [test_eids.iloc[0]]  # train/test_eids are panda series objects
        else:
            test_eids = test_eids.tolist()
        return train_eids, test_eids

    '''Get train and test splits of EID groups. There will be only one
    group to be tested on. This function will be used in
    self.get_train_and_test_eids() above.'''

    def get_train_and_test_eid_groups(self, df_eids):
        '''df_eids has columns:
        ['Epigenome ID (EID)', 'Standardized Epigenome name', 'Epigenome Mnemonic', 'GROUP', 'EID_info']
        '''
        eid_groups = sorted(list(set(df_eids["GROUP"])))
        eid_groups.remove("ENCODE2012")  # this will not be used to test the model
        test_eid_group = eid_groups[random.sample(range(0, len(eid_groups)), 1)[0]]
        eid_groups.remove(test_eid_group)
        train_eid_groups = eid_groups + ["ENCODE2012"]

        return train_eid_groups, test_eid_group

    '''Given a df, and train and test sample lists,
    return normalized df_train and df_test dataframes.
    Argument: df is one of gv.df_dhss or gv.df_tfs.
    This function is tested for when len(test_eids)==1.'''

    def get_normalized_train_test_dfs(self, df, train_eids, test_eids):

        train_df = df[train_eids]
        max_in_train = np.amax(np.array(train_df))
        train_df_normed = train_df.div(max_in_train)

        test_df = df[test_eids]
        test_df_normed = test_df.div(max_in_train)
        return train_df_normed, test_df_normed

    '''Normalize and split goi to train and test vectors.'''

    def get_normalized_train_and_test_goi(self, gv):
        '''First log transform the data'''
        if (gv.take_log2_tpm):
            gv.goi = np.log2(gv.goi)

        train_goi = gv.goi[gv.goi.index.isin(self.train_eids)]
        test_goi = gv.goi[gv.goi.index.isin(self.test_eids)]
        assert np.array_equal(self.train_eids, train_goi.index.tolist())  # check the order of samples
        assert np.array_equal(self.test_eids, test_goi.index.tolist())

        '''Now normalize'''
        max_gex_in_train = max(train_goi)
        train_goi = train_goi / max_gex_in_train
        test_goi = test_goi / max_gex_in_train

        return train_goi, test_goi

    '''Resetting the indices in df_dhss and df_tfs to merge for joint model.
    The reason indices need to be reset is that these dfs have different
    indices. Only the "feat" and "conf" (for confidence score) will be kept
    in these dfs. Note that these dfs have to be normalized before being merged.'''

    def merge_dhs_and_tf_dfs(self, df_dhss, df_tfs, gv):
        '''For df_tfs: Reset, drop (some) cols, rename other cols and set them as index'''
        df_tfs = df_tfs.reset_index()
        zscore_or_pcc_to_pop = set(["zscore", "pcc"]).difference(set([gv.filter_tfs_by])).pop()
        cols_to_pop = ["loc", "TAD_loc", "cn_corr", zscore_or_pcc_to_pop]
        df_tfs = df_tfs.drop(labels=cols_to_pop, axis=1)
        df_tfs = df_tfs.rename(columns=dict(zip(["geneName", gv.filter_tfs_by], ["feat", "conf"])))
        df_tfs = df_tfs.set_index(keys=["feat", "conf"])

        '''For df_dhss: Reset indices, change their (now column) names and set them as indices'''
        df_dhss = df_dhss.reset_index()
        df_dhss = df_dhss.rename(columns=dict(zip(["loc", "pcc"], ["feat", "conf"])))
        df_dhss = df_dhss.set_index(keys=["feat", "conf"])

        return pd.concat([df_dhss, df_tfs])
