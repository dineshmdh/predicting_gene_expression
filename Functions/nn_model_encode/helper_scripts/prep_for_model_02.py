'''
Created on August 14, 2017

Before anything, the cell lines for the gene should match
with that of the dhss. So, changing the cell line names as done
previously. Note that, there are some cell lines that do not
need to have their names changed. These include:
- astrocyte.rep1/2
- bipolar_spindle_neuron_derived_from_induced_pluripotent_stem_cell.rep1/2
- fibroblast_of_arm_male_adult_53_years.rep1/2
- GM12878.rep1/2
- H1-hESC.rep1
- hepatocyte_derived_from_H9.rep1/2
- HepG2.rep1/2
- IMR-90_female_fetal_16_weeks.rep1/2
- induced_pluripotent_stem_cell_male_adult_53_years_derived_from_fibroblast_of_arm.rep1/2
- K562.rep1/2
- Karpas-422.rep1/2
- MCF-7.rep1/2
- neural_progenitor_cell_derived_from_H9.rep1/2
- OCI-LY7.rep1/2
- ovary_female_adult_51_year.rep1
- PC-3.rep1/2
- sigmoid_colon_female_adult_53_years.rep1

But for these, the "_pileups" suffix needs to be removed first from self.df_dhss_in_roi.
'''

import pdb
import numpy as np
import random
import os
import re
import logging
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' (see README.md)

import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import get_agenes_cellnetTF_gexes
reload(get_agenes_cellnetTF_gexes)
from get_agenes_cellnetTF_gexes import Get_CellNet_tfs


class Model_preparation(object):

    def __init__(self, df_dhss_in_roi, gene_ofInterest_info, gene_ofInterest, df_gencode_all_mrnas,
                 df_allgenes_tpm_anno, inputDir, outputDir=os.getcwd(), take_log2_tpm_forNNwTFs=True,
                 use_random_wts=False, use_corr_wts=False, use_expDecay_wts=False,
                 take_top_fts=True, take_this_many_top_fts=20,
                 tf_corr_threshold_high=1.0, tf_corr_threshold_low=0.2,
                 enforce_corr_thresholding=False, frac_test=0.2, get_random_TFs=False, to_seed=False):
        '''
        Arguments:
        - df_allgenes_tpm_anno -- output of helper function get_full_tpm_info_df(csv_tpm_merged, csv_gencode_mrnas)
            which has gene location as well as expression profiles across the cell lines.
        - take_log2_tpm_forNNwTFs --
            if True, TPM values for all TFs from CellNet are log2 transformed i.e. x = np.log2(x+1)
                - before plotting gex profiles in train + test sets (just for the plot)
                - right after trainY and testY are set up (changing trainY and testY)
            This variable should be the same variable as Global_Vars variable take_log2_tpm

        - pc_input_for_h1_layer -- percentage of number of input nodes for hidden layer 1.
            (Note this is after only top features are selected - i.e.
            if self.take_top_fts is True)

        - tf_corr_threshold[_high/_low] -- Only consider TFs with PCC in expression with this gene in between these values
             for the model.
        '''

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

        self.logger.info("Setting up the Model_Preparation object..")
        self.to_seed = to_seed
        if (self.to_seed):
            self.seed_num = 4

        self.take_log2_tpm_forNNwTFs = take_log2_tpm_forNNwTFs
        self.gene_ofInterest = gene_ofInterest
        self.df_gencode_all_mrnas = df_gencode_all_mrnas
        self.df_allgenes_tpm_anno = df_allgenes_tpm_anno
        self.inputDir = inputDir
        self.outputDir = outputDir

        self.use_random_wts = use_random_wts
        self.use_corr_wts = use_corr_wts
        self.use_expDecay_wts = use_expDecay_wts
        assert sum([self.use_random_wts, self.use_corr_wts, self.use_expDecay_wts]) == 1

        self.take_top_fts = take_top_fts
        self.take_this_many_top_fts = take_this_many_top_fts
        self.tf_corr_threshold_high = tf_corr_threshold_high  # upper threshold for the TF corr; only TF with corr in between this and self.tf_corr_threshold_low are selected
        self.tf_corr_threshold_low = tf_corr_threshold_low  # lower threshold for the TF corr
        self.enforce_corr_thresholding = enforce_corr_thresholding
        self.frac_test = frac_test  # fraction of cell lines to be used in the test set
        self.get_random_TFs = get_random_TFs  # default is False

        self.fig_gex_in_train = None  # will be updated once this plot is made
        self.fig_gex_in_test = None  # will be updated later
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None

        self.trainX_tfsOnly = None  # for tfs-only model
        self.testX_tfsOnly = None
        self.trainX_wtfs = None  # for dhs+tf joint model
        self.testX_wtfs = None

        self.df_tpm_cellnetTFs = None  # will be saved after scaling (see main_for_tfs() function)
        self.dict_tf_to_corr = None  # {tf: PCC(tf, thisGene)} for high PCC (see main_for_tfs() function)
        # Note that w1 and w2 are not saved as attributes, but can be obtained from a function below

        '''Load df_dhss_in_roi, remove "_pileups" suffixes and, update the column names.'''
        self.df_dhss_in_roi = df_dhss_in_roi
        self.get_column_names_without_pileups()
        self.dhss_newSampleNames = {"adrenal_gland_male_fetal_108_days.rep1": "adrenal_gland_male",
                                    "body_of_pancreas_male_adult_54_years.rep1": "body_of_pancreas.rep1",
                                    "body_of_pancreas_male_adult_54_years.rep2": "body_of_pancreas.rep2",
                                    "cardiac_muscle_cell.rep12": "cardiac_muscle_cell",
                                    "fibroblast_of_dermis_female_adult.rep1": "fibroblast_of_dermis_female_adult.rep1",  # name not changed but being changed for rnaseq below
                                    "fibroblast_of_dermis_female_adult.rep2": "fibroblast_of_dermis_female_adult.rep2",  # name not changed but being changed for rnaseq below
                                    "fibroblast_of_lung.rep1": "fibroblast_of_lung.rep1",  # name not changed but being changed for rnaseq below
                                    "fibroblast_of_lung.rep2": "fibroblast_of_lung.rep2",  # name not changed but being changed for rnaseq below
                                    "gastrocnemius_medialis_male_adult_54_years.rep1": "gastrocnemius_medialis_adult",
                                    "heart_left_ventricle_female_fetal_101_day_and_female_fetal_136_days.rep1": "heart_left_ventricle_female.rep1",  # ADULT IN RNASEQ
                                    "heart_left_ventricle_female_fetal_101_day_and_female_fetal_136_days.rep2": "heart_left_ventricle_female.rep2",  # ADULT IN RNASEQ
                                    "lung_female_fetal_108_days.rep1": "lung_fetal.rep1",
                                    "lung_male_fetal_108_days.rep1": "lung_fetal.rep2",
                                    "mammary_epithelial_cell_female.rep1": "mammary_epithelial_cell_female",
                                    "myotube_derived_from_skeletal_muscle_myoblast.rep1": "myotube_derived_from_skeletal_muscle_myoblast.rep1",  # name not changed but being changed for rnaseq below
                                    "myotube_derived_from_skeletal_muscle_myoblast.rep2": "myotube_derived_from_skeletal_muscle_myoblast.rep2",  # name not changed but being changed for rnaseq below
                                    "osteoblast.rep0": "osteoblast",
                                    "ovary_female_adult_51_year.rep1": "ovary_female_adult.rep1",
                                    "ovary_female_adult_30_year.rep1": "ovary_female_adult.rep2",  # rep1 is for adult 51 years
                                    "sigmoid_colon_male_adult_37_years.rep1": "sigmoid_colon_male_adult.rep1",
                                    "skeletal_muscle_myoblast.rep12": "skeletal_muscle_myoblast",
                                    "spinal_cord_female_fetal_113_days.rep1": "spinal_cord_fetal.rep1",
                                    "spinal_cord_female_fetal_89_days.rep1": "spinal_cord_fetal.rep2",
                                    "spleen_fetal_112_days.rep1": "spleen",
                                    "stomach_female_fetal_105_days.rep1": "stomach_fetal.rep1",
                                    "stomach_female_fetal_107_days.rep1": "stomach_fetal.rep2",
                                    "thyroid_gland_male_adult_37_years.rep1": "thyroid_gland.rep1",
                                    "thyroid_gland_male_adult_37_years.rep2": "thyroid_gland.rep2",
                                    }
        self.df_dhss_in_roi.rename(columns=self.dhss_newSampleNames, inplace=True)

        '''Load gene_ofInterest_info series and update its columns as well.'''
        self.gene_ofInterest_info = gene_ofInterest_info
        self.rnaseq_newSampleNames = {"adrenal_gland_male_adult_37_years.rep1": "adrenal_gland_male",
                                      "body_of_pancreas_female_adult_51_reps.rep1": "body_of_pancreas.rep1",
                                      "body_of_pancreas_female_adult_53_year.rep1": "body_of_pancreas.rep2",
                                      "cardiac_muscle_cell.rep1": "cardiac_muscle_cell",
                                      "fibroblast_of_dermis_female_adult_44_years_and_female_adult_55_years.rep1": "fibroblast_of_dermis_female_adult.rep1",
                                      "fibroblast_of_dermis_female_adult_44_years_and_female_adult_55_years.rep2": "fibroblast_of_dermis_female_adult.rep2",
                                      "fibroblast_of_lung_female_adult_83_years_and_male_adult_23_years.rep1": "fibroblast_of_lung.rep1",
                                      "fibroblast_of_lung_female_adult_83_years_and_male_adult_23_years.rep2": "fibroblast_of_lung.rep2",
                                      "gastrocnemius_medialis_female_adult_51_year.rep1": "gastrocnemius_medialis_adult",
                                      "heart_left_ventricle_female_adult_51_year.rep1": "heart_left_ventricle_female.rep1",
                                      "heart_left_ventricle_female_adult_53_year.rep1": "heart_left_ventricle_female.rep2",
                                      "lung_female_fetal_20_weeks_and_female_fetal_24_weeks.rep1": "lung_fetal.rep1",
                                      "lung_female_fetal_20_weeks_and_female_fetal_24_weeks.rep2": "lung_fetal.rep2",
                                      "mammary_epithelial_cell_female_adult_23_years.rep1": "mammary_epithelial_cell_female",
                                      "myotube_NONE_and_derived_from_skeletal_muscle_myoblast.rep1": "myotube_derived_from_skeletal_muscle_myoblast.rep1",
                                      "myotube_NONE_and_derived_from_skeletal_muscle_myoblast.rep2": "myotube_derived_from_skeletal_muscle_myoblast.rep2",
                                      "osteoblast_female_adult_56_years_and_male_adult_62_years.rep1": "osteoblast",
                                      "ovary_female_adult_51_year.rep1": "ovary_female_adult.rep1",
                                      "ovary_female_adult_53_years.rep1": "ovary_female_adult.rep2",
                                      "sigmoid_colon_male_adult_54_years.rep1": "sigmoid_colon_male_adult.rep1",
                                      "skeletal_muscle_myoblast.rep1": "skeletal_muscle_myoblast",
                                      "spinal_cord_female_fetal_24_weeks_and_male_fetal_22_weeks.rep1": "spinal_cord_fetal.rep1",
                                      "spinal_cord_female_fetal_24_weeks_and_male_fetal_22_weeks.rep2": "spinal_cord_fetal.rep2",
                                      "spleen_female_adult_51_year.rep1": "spleen",
                                      "stomach_female_fetal_40_weeks_and_male_fetal_36_weeks.rep1": "stomach_fetal.rep1",
                                      "stomach_female_fetal_40_weeks_and_male_fetal_36_weeks.rep2": "stomach_fetal.rep2",
                                      "thyroid_gland_female_fetal_37_weeks_and_female_fetal_40_weeks.rep1": "thyroid_gland.rep1",
                                      "thyroid_gland_female_fetal_37_weeks_and_female_fetal_40_weeks.rep2": "thyroid_gland.rep2"
                                      }
        self.gene_ofInterest_info.rename(self.rnaseq_newSampleNames, inplace=True)

        '''Get cell lines or tissue types are in common between the two sets - with reps and those that are unique'''
        self.celllines_incommon_wreps = set(self.gene_ofInterest_info.index).intersection(set(self.df_dhss_in_roi.columns))
        self.celllines_incommon_unique = set([x[:-5] if x.__contains__(".rep") else x for x in self.celllines_incommon_wreps])
        self.logger.info("There are {0} samples in total for the NN training. Out of this, {1} are unique celllines/tissue types.".format(len(self.celllines_incommon_wreps), len(self.celllines_incommon_unique)))
        self.celllines = sorted(list(self.celllines_incommon_wreps))  # these are the only cell lines of interest from here forth

        '''Get training and test cell lines'''
        self.train_celllines, self.test_celllines = self.get_train_and_test_celllines()

    def __repr__(self):
        return self.__class__.__name__ + " instance"

    def get_column_names_without_pileups(self):
        '''Removing the "_pileup" suffixes from the column names.'''
        new_dhss_names_noPileups = self.df_dhss_in_roi.columns.tolist()[:3] \
            + [x[:-8] for x in self.df_dhss_in_roi.columns.tolist()[3:-1]] \
            + ["has_knownEnhancer"]
        self.df_dhss_in_roi.rename(columns=dict(zip(self.df_dhss_in_roi.columns.tolist(),
                                                    new_dhss_names_noPileups)), inplace=True)

    def get_train_and_test_celllines(self):
        '''Generate training and test samples
        (Old version had this: selected = np.random.multinomial(len(self.celllines), [1. / len(self.celllines)] * len(self.celllines))  # "selected" if >0)'''
        if (self.to_seed):
            np.random.seed(self.seed_num)
        to_select_from = np.random.rand(len(self.celllines))
        train_celllines = []
        test_celllines = []

        # pdb.set_trace()

        for i, y_i in enumerate(to_select_from):
            if (y_i > self.frac_test):
                train_celllines.append(self.celllines[i])
            else:
                test_celllines.append(self.celllines[i])
        self.logger.info("Number of training cell lines: {}".format(len(train_celllines)))
        self.logger.info("Number of test cell lines: {}".format(len(test_celllines)))
        self.logger.info("The following samples were used in the test set:")
        for i, j in enumerate(test_celllines):
            self.logger.info("    Test sample {}: {}".format(i, j))
        return train_celllines, test_celllines

    def plot_gex_in_training_set(self):
        '''Note: The gene expressions are not logged in the data structures yet.
        This is done right before Ytrain and Ytest are normalized later.
        '''
        sns.set(font_scale=1.5)
        sns.set_style("whitegrid")
        fig = plt.figure(0, (12, 10))

        if (self.take_log2_tpm_forNNwTFs):
            training_dict = dict(zip(self.train_celllines, [np.log2(x + 1) for x in self.gene_ofInterest_info[self.train_celllines].tolist()]))
            test_dict = dict(zip(self.test_celllines, [np.log2(x + 1) for x in self.gene_ofInterest_info[self.test_celllines].tolist()]))
            x_lab = self.gene_ofInterest + " log2(TPM+1)"
        else:
            training_dict = dict(zip(self.train_celllines, [x for x in self.gene_ofInterest_info[self.train_celllines].tolist()]))
            test_dict = dict(zip(self.test_celllines, [x for x in self.gene_ofInterest_info[self.test_celllines].tolist()]))
            x_lab = self.gene_ofInterest + " TPM"

        '''Sort the dicts before plotting'''
        training_dict_sorted = sorted(training_dict.items(), reverse=True, key=lambda x: x[1])
        test_dict_sorted = sorted(test_dict.items(), reverse=True, key=lambda x: x[1])

        '''First plot the expns on training set'''
        plt.barh(np.arange(len(self.train_celllines)), [x[1] for x in training_dict_sorted], align='center', alpha=1)
        plt.yticks(np.arange(len(self.train_celllines)), [x[0] for x in training_dict_sorted])
        plt.xlabel(x_lab)
        plt.xlim((0, max(max(training_dict.values()), max(test_dict.values())) + 1))
        plt.title(self.gene_ofInterest + " expression in Training set")
        plt.tight_layout()
        # plt.show()
        plt.savefig(self.outputDir + "/" + self.gene_ofInterest + "_expn_in_training_set.pdf")
        self.fig_gex_in_train = fig
        plt.close()

        return training_dict, test_dict, test_dict_sorted, x_lab

    def plot_gex_in_test_set(self, training_dict, test_dict, test_dict_sorted, x_lab):
        sns.set(font_scale=1.5)
        fig = plt.figure(0, (12, 5))
        sns.set_style("whitegrid")

        plt.barh(np.arange(len(self.test_celllines)), [x[1] for x in test_dict_sorted], align='center', alpha=1)
        plt.yticks(np.arange(len(self.test_celllines)), [x[0] for x in test_dict_sorted])
        plt.xlabel(x_lab)
        plt.xlim((0, max(max(training_dict.values()), max(test_dict.values())) + 1))
        plt.title(self.gene_ofInterest + " expression in Test set")
        plt.tight_layout()
        # plt.show()
        plt.savefig(self.outputDir + "/" + self.gene_ofInterest + "_expn_in_test_set.pdf")
        self.fig_gex_in_test = fig
        plt.close()

    def get_corr_wts(self):
        '''Add a "init_wt" field in self.df_dhss_in_roi with
        correlation weights betweeen dhs site and the gex profile for self.gene_ofInterest

        IMPORTANT. Initially I had assert self.use_corr_wts at the top of the function, but
        this is removed since the df_dhss_in_roi["init_wt"] field is used to select top DHS
        features (based on correlations) if --take_top_fts is turned on.'''

        self.df_dhss_in_roi["init_wt"] = -2
        '''Using correlation on dhss and gexes for all cell lines atm (i.e. both train and test data)'''
        dhss = []
        corrs = []
        assert np.array_equal(self.gene_ofInterest_info[self.celllines].index, self.df_dhss_in_roi[self.celllines].columns)  # this is important

        for i in range(0, self.df_dhss_in_roi.shape[0]):
            dhss.append(self.df_dhss_in_roi.iloc[i]["chr_dhs"] + ":" + str(self.df_dhss_in_roi.iloc[i]["ss_dhs"]) + "-" + str(self.df_dhss_in_roi.iloc[i]["es_dhs"]))
            corr_ = np.corrcoef(self.df_dhss_in_roi[self.celllines].iloc[i].tolist(), self.gene_ofInterest_info[self.celllines].tolist())[0, 1]
            corrs.append(corr_)
            self.df_dhss_in_roi.loc[self.df_dhss_in_roi.index[i], "init_wt"] = corr_
        '''Plot the correlations'''
        pd.DataFrame.from_records(sorted(dict(zip(dhss, corrs)).items(), key=lambda x: x[1]), columns=["dhs_loc", "corr_wGex"]).plot(kind='barh', legend=False, use_index=False)
        # plt.show()
        plt.savefig(self.outputDir + "/" + self.gene_ofInterest + "_corr_w_dhss.png")
        plt.close()

    def trim_dhs_fts_size(self):
        ''' Trim the df size (for faster computation), if self.take_top_fts is true..
        Trimming is done for DHS sites that have low correlation / expDecay score.
        '''
        if self.take_top_fts and (self.df_dhss_in_roi.shape[0] >= self.take_this_many_top_fts):
            self.df_dhss_in_roi["abs_init_wt"] = self.df_dhss_in_roi["init_wt"].apply(lambda x: abs(x))
            self.df_dhss_in_roi.sort_values(by=["abs_init_wt"], ascending=False, inplace=True)

            self.df_dhss_in_roi = self.df_dhss_in_roi.iloc[:self.take_this_many_top_fts, :]
            self.df_dhss_in_roi = self.df_dhss_in_roi.sort_values(by=["chr_dhs", "ss_dhs"])
            '''Drop this abs_init_wt column'''
            self.df_dhss_in_roi.drop(["abs_init_wt"], axis=1, inplace=True)

    def get_train_test_XY(self):
        '''Return normalized training and test data matrices'''
        trainX = []
        trainY = []
        testX = []
        testY = []
        for i, s in enumerate(self.celllines):
            if (s in self.train_celllines):
                trainX.append(self.df_dhss_in_roi[s].tolist())
                trainY.append([self.gene_ofInterest_info[s]])
            else:
                assert (s in self.test_celllines)
                testX.append(self.df_dhss_in_roi[s].tolist())
                testY.append([self.gene_ofInterest_info[s]])

        trainX = np.array(trainX, dtype=np.float32)
        trainY = np.array(trainY, dtype=np.float32)
        testX = np.array(testX, dtype=np.float32)
        testY = np.array(testY, dtype=np.float32)

        if (not self.take_log2_tpm_forNNwTFs):
            self.logger.info("Note: Not taking log2 of TPM values for this gene across samples before data normalization.")
        else:
            trainY = np.log2(trainY + 1)
            testY = np.log2(testY + 1)

        '''Normalize the data matrices'''
        trainX_max = np.amax(trainX)
        trainY_max = np.amax(trainY)
        self.trainX = trainX / trainX_max  # normalize by the maximum score in the region of interest
        self.trainY = trainY / trainY_max  # normalize by the maximum gex score
        self.testX = testX / trainX_max  # normalize by the same value as in the training set
        self.testY = testY / trainY_max  # same as in testX

    def get_init_w1_w2_wts(self, percent_inputNum_for_h1, variance=0.001):
        '''The assumption is that the self.df_dhss_in_roi is trimmed for top dhss. Asserting this.'''
        if (self.take_top_fts):
            assert (self.df_dhss_in_roi.shape[0] <= self.take_this_many_top_fts)
        input_num = self.df_dhss_in_roi.shape[0]
        hidden_num = int(percent_inputNum_for_h1 * 0.01 * input_num)
        out_num = 1
        self.logger.info("i, h, o (with just dhss): {}, {}, {}".format(input_num, hidden_num, out_num))

        if (self.use_random_wts):
            self.logger.info("Using random weights..note: the df is trimmed aforehand based on corr/expDecay coeffs")
            if (self.to_seed):
                np.random.seed(random.randrange(0, 2000))  # to make sure not seeded with self.seed_num
            w1 = np.random.randn(input_num, hidden_num)
            if (self.to_seed):
                np.random.seed(random.randrange(0, 2000))  # to make sure not seeded with self.seed_num
            w2 = np.random.randn(hidden_num, out_num)

        if (self.use_expDecay_wts or self.use_corr_wts):
            self.logger.info("Using expDecay wts.." if self.use_expDecay_wts else "using corr wts..")
            mean = np.zeros(input_num)
            # previously:
            # w1 = np.zeros((input_num, hidden_num))
            # w2 = np.ones((hidden_num, out_num))  # * 0.5 # is it that w2=0.5 gave worse results?
            for i in range(0, self.df_dhss_in_roi.shape[0]):  # for all dhss
                mean[i] = self.df_dhss_in_roi.iloc[i]["init_wt"]
            cov = np.identity(len(mean)) * variance

            if (self.to_seed):
                np.random.seed(random.randrange(0, 2000))
            w1 = np.random.multivariate_normal(mean, cov, hidden_num).transpose()  # some noise added to mean
            if (self.to_seed):
                np.random.seed(random.randrange(0, 2000))
            w2 = np.random.multivariate_normal(np.zeros(hidden_num), np.identity(hidden_num) * 0.001, 1).transpose()  # some noise added to vector of 1s

        return w1, w2

    def get_input_labels_for_NN_plot(self):
        '''These labels are used to plot the NN model plot
        (after this function is called, i.e.)'''
        labels = []
        chrs = self.df_dhss_in_roi["chr_dhs"].tolist()
        sses = [str(x) for x in self.df_dhss_in_roi["ss_dhs"]]
        eses = [str(x) for x in self.df_dhss_in_roi["es_dhs"]]
        assert len(chrs) == len(sses) == len(eses)

        for i in range(0, len(chrs)):
            labels.append(chrs[i] + ":" + sses[i] + "-" + eses[i])
        return labels

    def get_tpm_cellnet_df(self):
        '''Returns a gene expression df with gene_ids as rows and cell lines as columns
        for the tfs in cellnet.

        Below, df_cellnet or cellnet.df_to_plot has TFs not filtered by corrs
        '''
        cellnet = Get_CellNet_tfs(self.gene_ofInterest, self.inputDir, self.outputDir,
                                  self.tf_corr_threshold_high, self.tf_corr_threshold_low,
                                  self.enforce_corr_thresholding,
                                  df_gencode_all_mrnas=self.df_gencode_all_mrnas,
                                  df_allgenes_tpm_anno=self.df_allgenes_tpm_anno,
                                  get_random_TFs=self.get_random_TFs)
        df_cellnet = cellnet.df_to_plot  # has rows as TFs (first one is the gene_ofInterest); TFs are by default sorted by corr values
        if (df_cellnet.shape[0] == 1):  # there is no TF associated with this gene
            return None, None

        dict_tf_to_corr = cellnet.tfs_to_corrs_sortedDict  # already filtered by corr thresholds; {tf: gex_corr}
        if (len(dict_tf_to_corr) == 0):  # there is no TF that passed the corr threshold
            return None, None

        # Tidy up this df_tf_to_corr to have unique TFs that pass the corr threshold (step 1) and by samples (step 2)
        # # df_tpm_cellnetTFs = self.df_allgenes_tpm_anno[self.df_allgenes_tpm_anno["geneSymbol"].isin(dict_tf_to_corr.keys())]  # this didn't ensure geneSymbols were unique for rows
        df_tpm_cellnetTFs = cellnet.df_genes_to_plot_tpm_anno[cellnet.df_genes_to_plot_tpm_anno["geneSymbol"].isin(dict_tf_to_corr.keys())]  # step 1; the dict has TFs that pass corr thrd.
        df_tpm_cellnetTFs.rename(columns=self.rnaseq_newSampleNames, inplace=True)  # Renames the column names
        df_tpm_cellnetTFs = df_tpm_cellnetTFs[["geneSymbol"] + self.train_celllines + self.test_celllines]  # step 2
        df_tpm_cellnetTFs.set_index("geneSymbol", inplace=True)
        df_tpm_cellnetTFs = df_tpm_cellnetTFs[df_tpm_cellnetTFs.columns].astype(float)  # sometimes, not having this step will raise Exception (as with the case of TAL1)

        try:
            df_tpm_cellnetTFs = cellnet.add_and_sortBy_TFcorr_in_df_indices(df_tpm_cellnetTFs, True, cellnet.tfs_to_tfsWCorrs_sortedDict)  # True is for to_add_corr_values_or_not == True
        except:
            pdb.set_trace()

        cellnet.plot_final_df(df_tpm_cellnetTFs, plot_title="df_to_plot filtd by corr and num samples _ has {} samples)".format(len(df_tpm_cellnetTFs.columns)))

        self.plot_sorted_gexes(df_tpm_cellnetTFs)  # temp

        if (not self.take_log2_tpm_forNNwTFs):
            self.logger.info("Note: Not taking log2 of tpms for this gene + other TFs in the GRN..")
        else:
            df_tpm_cellnetTFs = np.log2(df_tpm_cellnetTFs + 1)

        return df_tpm_cellnetTFs, dict_tf_to_corr

    def plot_sorted_gexes(self, df_tpm_cellnetTFs):
        '''Plot the gene of interest df and tfs df for the train set sorted by gex for this gene
        Plotting to be used in figure 1'''

        goi = self.gene_ofInterest_info
        goi_train = goi[goi.index.isin(self.train_celllines)]
        goi_train_sorted = goi_train.sort_values(ascending=False)

        df_goi = goi_train_sorted.to_frame()
        df_goi = df_goi[df_goi.columns].astype(float)
        df_goi = np.log2(df_goi + 1)

        # db.set_trace()

        sns.set(font_scale=1.3)
        plt.figure(figsize=(8, 10), dpi=80, facecolor='w', edgecolor='k')
        sns.heatmap(df_goi, vmax=6, yticklabels=True)
        plt.xticks(rotation=85)
        plt.tight_layout()
        plt.savefig(self.outputDir + "/df_goi_train_for_figure1.pdf")
        plt.close()

        df = df_tpm_cellnetTFs
        df_train = df[self.train_celllines]
        df_train = df_train[goi_train_sorted.index.tolist()]  # is TFs by samples
        df_train = np.log2(df_train + 1)

        plt.figure(figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
        sns.heatmap(df_train.transpose(), vmax=6, yticklabels=True)
        plt.xticks(rotation=85)
        plt.tight_layout()
        plt.savefig(self.outputDir + "/df_gex_train_for_figure1.pdf")
        plt.close()

    def scale_tpm_df(self, df_tpm_cellnetTFs):
        '''Before merging with train/test dfs with dhss, it is important to scale
         df_tpm_cellnetTFs separately by dividing with max(gexes in train set)
         '''
        max_gex_inTrain = np.amax(np.amax(df_tpm_cellnetTFs[self.train_celllines]))
        df_tpm_cellnetTFs = df_tpm_cellnetTFs.transpose() / max_gex_inTrain
        return df_tpm_cellnetTFs

    def get_merged_or_tfsOnly_Xmatrices(self):
        '''This is an updated version of the old self.merge_dhs_and_tf_matrices() function that
        only returned the merged trainX and testX dfs for the joint (dhs+tf) model.
        '''

        '''Creating trainX_dhss and testX_dhss dfs'''
        df_trainX_dhss = pd.DataFrame(self.trainX)
        df_trainX_dhss = df_trainX_dhss.rename(index=dict(zip(df_trainX_dhss.index, self.train_celllines)))  # the order of cell lines and trainX rows is the same
        df_testX_dhss = pd.DataFrame(self.testX)
        df_testX_dhss = df_testX_dhss.rename(index=dict(zip(df_testX_dhss.index, self.test_celllines)))

        '''Now the new tf dfs to merge with (for both train and test sets)'''
        df_trainX_tfs = self.df_tpm_cellnetTFs[self.df_tpm_cellnetTFs.index.isin(self.train_celllines)]
        df_testX_tfs = self.df_tpm_cellnetTFs[self.df_tpm_cellnetTFs.index.isin(self.test_celllines)]
        df_trainX_tfs = df_trainX_tfs.reindex(self.train_celllines)  # IMPORTANT to reindex
        df_testX_tfs = df_testX_tfs.reindex(self.test_celllines)  # IMPORTANT to reindex

        '''Merge the trainX_dhss with the trainX_tfs info and testX_dhss with testX_tfs'''
        assert np.array_equal(df_trainX_dhss.index, df_trainX_tfs.index)
        assert np.array_equal(df_testX_dhss.index, df_testX_tfs.index)
        df_trainX_wtfs = pd.concat([df_trainX_dhss, df_trainX_tfs], axis=1, join="inner")
        df_testX_wtfs = pd.concat([df_testX_dhss, df_testX_tfs], axis=1, join="inner")

        return df_trainX_wtfs, df_testX_wtfs, df_trainX_tfs, df_testX_tfs

    def get_init_w1_w2_wts_wtfs_orForJust_tfs(self, df_trainX_wtfs, w1, percent_inputNum_for_h1, variance=0.001):
        ''' (Updated on Oct 5, 2017) To get w1 and w2 weights for tfs-only model. Previously, the name of the function was
        get_init_w1_w2_wts_wtfs().
        Arguments:
        - w1 -- is the weight with dhss only.
        - df_trainX_wtfs -- is being used to get order of tfs in trainX to get their respective wts (as corrs).

        Below, self.dict_tf_to_corr is of form: {tf: PCC(tf, thisgene)}
        for all tfs with PCC in between self.tf_corr_threshold_high and self.tf_corr_threshold_low

        # Previous version of the script:
        w1_tfs = [self.dict_tf_to_corr[atf] for atf in df_trainX_wtfs.columns[-len(self.dict_tf_to_corr):]]  # to preserve the order in trainX
        w1_wtfs = np.array(list(w1[:, 0]) + w1_tfs)  # merging dhs w1 and w1_wtfs weights
        hidden_num_wtfs = int(w1_wtfs.shape[0] * percent_inputNum_for_h1 * 0.01)
        w1_wtfs = np.tile(w1_wtfs, (hidden_num_wtfs, 1)).transpose()  # rows are input nodes, cols are wts for h1 nodes

        out_num = 1
        w2_wtfs = np.ones((hidden_num_wtfs, out_num))

        return w1_wtfs, w2_wtfs
        '''
        try:
            corrs_tfs = [self.dict_tf_to_corr[re.split("\s+", atf)[0]] for atf in df_trainX_wtfs.columns[-len(self.dict_tf_to_corr):]]  # to preserve the order in trainX; atf here is has corr val also (eg: "FOXK1 0.59")
        except:
            for k, v in self.dict_tf_to_corr.items():
                self.logger.debug("{}, {}".format(k, v))
            self.logger.info("{}".format(df_trainX_wtfs.columns[-len(self.dict_tf_to_corr):]))
            pdb.set_trace()

        '''The following tfs-only specific code is added later (October 5, 2017).
        Added/edited by copying the script for the joint model
        Further updated on October 27, 2017 -- to add an option of getting random weights'''

        if (self.use_corr_wts):
            h = int(len(corrs_tfs) * percent_inputNum_for_h1 * 0.01)  # h == number of hidden units with tfs
            if (self.to_seed):
                np.random.seed(random.randrange(0, 2000))
            w1_tfsOnly = np.random.multivariate_normal(np.array(corrs_tfs),
                                                       np.identity((len(corrs_tfs))) * variance, h).transpose()
            if (self.to_seed):
                np.random.seed(random.randrange(0, 2000))
            w2_tfsOnly = np.random.multivariate_normal(np.zeros(h),
                                                       np.identity(h) * variance, 1).transpose()  # some noise added to vector of 0s
            ''' Now For the joint (dhs + tf) model'''
            mean_corrs = np.array(list(np.mean(w1, axis=1)) + corrs_tfs)  # merging dhs w1 and w1_wtfs weights
            cov = np.identity((len(mean_corrs))) * variance
            h = int(len(mean_corrs) * percent_inputNum_for_h1 * 0.01)  # h == number of hidden units with tfs

            if (self.to_seed):
                np.random.seed(random.randrange(0, 2000))
            w1_wtfs = np.random.multivariate_normal(mean_corrs, cov, h).transpose()  # some noise added to mean
            if (self.to_seed):
                np.random.seed(random.randrange(0, 2000))
            w2_wtfs = np.random.multivariate_normal(np.zeros(h),
                                                    np.identity(h) * variance, 1).transpose()  # some noise added to vector of 0s
        elif (self.use_random_wts):
            h = int(len(corrs_tfs) * percent_inputNum_for_h1 * 0.01)  # for tfs
            w1_tfsOnly = np.random.randn(len(corrs_tfs), h)
            w2_tfsOnly = np.random.randn(h, 1)

            i = w1.shape[0] + len(corrs_tfs)  # for joint model
            h = int(i * percent_inputNum_for_h1 * 0.01)  # for joint model
            w1_wtfs = np.random.randn(i, h)
            w2_wtfs = np.random.randn(h, 1)
        else:
            raise Exception()

        self.logger.info("i, h, o (tfs only): {}, {}, {}".format(w1_tfsOnly.shape[0], w2_tfsOnly.shape[0], 1))
        self.logger.info("i, h, o (with dhss and tfs): {}, {}, {}".format(w1_wtfs.shape[0], w2_wtfs.shape[0], 1))

        return w1_wtfs, w2_wtfs, w1_tfsOnly, w2_tfsOnly

    def main_for_dhss(self):
        '''This function is deprecated. Use main_for_tfsOnly_and_wTFs(self) (see below).'''
        training_dict, test_dict, test_dict_sorted, x_lab = self.plot_gex_in_training_set()
        self.plot_gex_in_test_set(training_dict, test_dict, test_dict_sorted, x_lab)

        '''Get corr wts and trim the df - regardless of the type of wts (random, corr, expDecay) being used'''
        self.get_corr_wts()  # run even when init_wts_type == "random" (see function); updates self.df_dhss_in_roi with "init_wt"
        self.trim_dhs_fts_size()  # only takes the dhs sites with highest pcc / expDecay scores

        self.get_train_test_XY()  # returns normalized

    def main_for_wtfs(self):
        df_tpm_cellnetTFs, self.dict_tf_to_corr = self.get_tpm_cellnet_df()
        if (len(self.dict_tf_to_corr) > 0):  # < 0 happens when no single TF passes the tf_corr_threshold (like with ITGA7 and tf_corr_threshold of 0.4) OR when there is no TF for th gene to begin with.
            if (self.tpm_df_is_scaled is False):  # to make sure that I re-scale it if I have already done this for TFs-only model.
                self.df_tpm_cellnetTFs = self.scale_tpm_df(df_tpm_cellnetTFs)  # divides by maximum in train Set
                self.tpm_df_is_scaled = True
            df_trainX_new, df_testX_new = self.merge_dhs_and_tf_matrices()
            self.trainX_wtfs = np.array(df_trainX_new)
            self.testX_wtfs = np.array(df_testX_new)
            return df_trainX_new, df_testX_new
        else:
            return None, None

    def main_for_tfsOnly_and_wTFs(self):
        ''' Return the trainX dfs for tfs-only and with tfs models. This function is an updated version of
        main_for_wtfs(self) function that only handled the joint dhs-tfs model.

        Below:
        - df_tpm_cellnetTFs has TFs as rows and samples as columns.
        - df_trainX_tfs and df_testX_tfs have samples as rows and TFs as columns.

        '''
        df_tpm_cellnetTFs, self.dict_tf_to_corr = self.get_tpm_cellnet_df()
        if (self.dict_tf_to_corr is None) or (len(self.dict_tf_to_corr) == 0):  # the dict is None when there is no TF for th gene to begin with. And len == 0 happens when no single TF passes the tf_corr_threshold (like with ITGA7 and tf_corr_threshold of 0.4).
            return None, None, None, None

        self.df_tpm_cellnetTFs = self.scale_tpm_df(df_tpm_cellnetTFs)  # divides by maximum in train Set
        df_trainX_wtfs, df_testX_wtfs, df_trainX_tfsOnly, df_testX_tfsOnly = self.get_merged_or_tfsOnly_Xmatrices()
        self.trainX_wtfs = np.array(df_trainX_wtfs)
        self.testX_wtfs = np.array(df_testX_wtfs)
        self.trainX_tfsOnly = np.array(df_trainX_tfsOnly)
        self.testX_tfsOnly = np.array(df_testX_tfsOnly)

        return df_trainX_wtfs, df_testX_wtfs, df_trainX_tfsOnly, df_testX_tfsOnly
