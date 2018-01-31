--hidden_size

parser.add_argument("-l", "--take_log2_tpm", help="(Default: True)", type=bool, default=True)

put and hidden nodes are set to the correlation values between the node feature and the expression of the gene, and the initial weights between hidden layers or the hidden layer and output is set to 0.5


parser.add_argument("-ft", "--frac_test", help="(Default:0.2)", default=0.2, type=float)
parser.add_argument("-lr", "--learning_rates", help="", nargs="+", default=[0.05], type=float)


--slurmrank == --run_id

self.tf_corr_threshold_high = tf_corr_threshold_high
self.tf_corr_threshold_low = tf_corr_threshold_low

# update weight parameters
self.use_random_wts = False
self.use_corr_wts = False
self.use_expDecay_wts = False

if (use_wts == "corr"):
    self.use_corr_wts = True
elif (use_wts == "random"):
    self.use_random_wts = True
elif (use_wts == "expDecay"):
    self.use_expDecay_wts = True
else:
    raise Exception("For parameter 'use_wts', choose one from 'corr', 'random' or 'expDecay'..")
assert sum([self.use_random_wts, self.use_corr_wts, self.use_expDecay_wts]) == 1


# want a file w/ these info:
- gene loc
- gene name ensemble and gene symbol
- tad info
- list of known enhancers for this gene
- Expression across cell types

# want another similar file w/ these info:
- dhs loc
- accessibility across cell types

== == == == == == == ==== == == == == == == == == == ==== == == == == == == == == == ==

WIKI

== == == == == == == ==== == == == == == == == == == ==== == == == == == == == == == ==


Here we explain the idea behind the modeling framework, comparison against current state - of - the - art, and some drawbacks of the model.

# Idea behind Neural Network based modeling framework
Being a cross - cell type supervised model, training, validation and testing the gene - regulation neural net model is done using a disjoint subset of cell types that are grouped by their lineage. We use neural networks to model non - linear, synergistic and potentially hierarchical regulatory relationships between regulatory chromatin sites and TFs. For every gene, ENCODE enhancers near its TSS and a set of predicted regulatory TFs(Transcription Factors) obtained from [CellNet](https: // www.ncbi.nlm.nih.gov / pubmed / 25126793) are used as features.

![Pipeline and Modeling Framework](https: // github.com / dineshmdh / predicting_gene_expression / blob / master / Images / methods_protocol_and_NNframework.pdf "Modeling framework")


# Comparison against JEME model
To our knowledge, this is the first cross - cell type prediction framework that also makes use of the expression level of regulatory TFs in the test cell type(s). Previously, cross - cell type gene expression prediction models have been developed, the best - to - date being the[JEME](https: // www.nature.com / articles / ng.3950)(short for Joint Effect of Multiple Enhancers) model published in 2017. A major limitation of JEME is that it requires larger array of datasets and features – namely, three different histone markers for enhancers(H3K4me1, H3K27ac, H3K27me3) in addition to the DNase - seq data across all cis - enhancers, promoter and regions in between. Our model comparatively, just requires chromatin accessibility(like DNase - seq) and transcriptional expression(like RNA - seq) data. Hence, our model shows that biological prediction with two important regulatory factors - namely, cis - regulatory elements, and regulatory TFs - can yield similar performance as those that make use of large collection of(potentially just correlative) histone modification features.

# Some drawbacks of the model
1. One major limitation of the model is that the list of regulatory TFs predicted to regulate the gene of interest is not complete. The TF - TG(i.e. TF - Target Gene) list is obtained from CellNet has low recall, meaning the number of false negatives is high. A higher quality human TF - TG network would therefore help improve the model performance.
2. Another limitation is that we inherently do not consider other modes of gene regulation, such as 3 - D enhancer loops, micro - RNA and eQTL based regulation. Despite this, our model seems to perform well, indicating that the the regulation of many genes can be predicted efficiently without data on "higher order" modes of regulation.
3. By design of our leave - one - tissue - group - out learning framework, our model does not perform well for tissue - specific genes tested against the same tissues.


== == == == == == == == == == == ==== == == == == == == == == == ==== == == == == == == == == == ==
# edited functions (now also copied) to normalize to [0-1] scale..

    def get_normalized_train_val_test_dfs(self, df, train_eids, val_eids, test_eids):

        train_df = df[train_eids]
        min_in_train = np.amin(np.array(train_df))
        train_shifted = (train_df - min_in_train)
        max_in_train = np.amax(np.array(train_shifted))
        train_df_normed = train_shifted.div(max_in_train)

        val_df = df[val_eids]
        val_df_normed = (val_df - min_in_train).div(max_in_train)

        test_df = df[test_eids]
        test_df_normed = (test_df - min_in_train).div(max_in_train)
        return train_df_normed, val_df_normed, test_df_normed

    '''Normalize and split goi to train and test vectors.'''

    def get_normalized_train_val_and_test_goi(self, gv, train_eids, val_eids, test_eids):

        train_goi = gv.goi[gv.goi.index.isin(train_eids)]
        val_goi = gv.goi[gv.goi.index.isin(val_eids)]
        test_goi = gv.goi[gv.goi.index.isin(test_eids)]
        assert np.array_equal(train_eids, train_goi.index.tolist())  # check the order of samples
        assert np.array_equal(val_eids, val_goi.index.tolist())
        assert np.array_equal(test_eids, test_goi.index.tolist())

        '''Now normalize'''
        min_gex_in_train = min(train_goi)
        train_goi_shifted = (train_goi - min_gex_in_train)
        max_gex_in_train = max(train_goi_shifted)

        train_goi = train_goi_shifted / max_gex_in_train
        val_goi = (val_goi - min_gex_in_train) / max_gex_in_train
        test_goi = (test_goi - min_gex_in_train) / max_gex_in_train

        return train_goi, val_goi, test_goi


== == == == ==== == ==


# original/ what i had before

    def get_normalized_train_val_test_dfs(self, df, train_eids, val_eids, test_eids):

        train_df = df[train_eids]
        max_in_train = np.amax(np.array(train_df))
        train_df_normed = train_df.div(max_in_train)

        val_df = df[val_eids]
        val_df_normed = val_df.div(max_in_train)

        test_df = df[test_eids]
        test_df_normed = test_df.div(max_in_train)
        return train_df_normed, val_df_normed, test_df_normed

    '''Normalize and split goi to train and test vectors.'''

    def get_normalized_train_val_and_test_goi(self, gv, train_eids, val_eids, test_eids):

        train_goi = gv.goi[gv.goi.index.isin(train_eids)]
        val_goi = gv.goi[gv.goi.index.isin(val_eids)]
        test_goi = gv.goi[gv.goi.index.isin(test_eids)]
        assert np.array_equal(train_eids, train_goi.index.tolist())  # check the order of samples
        assert np.array_equal(val_eids, val_goi.index.tolist())
        assert np.array_equal(test_eids, test_goi.index.tolist())

        '''Now normalize'''
        max_gex_in_train = max(train_goi)
        train_goi = train_goi / max_gex_in_train
        val_goi = val_goi / max_gex_in_train
        test_goi = test_goi / max_gex_in_train

        return train_goi, val_goi, test_goi

====================================================================================
====================================================================================

'''script to plot feature importance subplots.'''

'''The following dict is obtained for TNNC2. Each value element or list is
supposed to of length 18 (for 18 rounds of testing, each pertaining to a test group).
Also, the number of items is 16 b/c I had used 6 dhss and 10 tfs as features.'''

d = {}
d[0] = [0.015224665, 0.0033840537, 0.0, 0.010297842, 5.748868e-05, 0.0003817156, 0.0029598996, 0.0066587925, 0.00053277612, 0.0079500154, 0.0050224662]
d[1] = [0.0062778294, 0.00033967197, 0.0, 0.0026238933, 0.0010964423, 0.001374498, 0.0042876899, 0.0012267157, 0.0023483485, 0.00097648799, 0.0032420754]
d[2] = [0.0034249425, 5.3867698e-05, 0.0, 0.0012852773, 0.00099158287, 0.00069842488, 0.0023539811, 0.00038716942, 0.00020509958, 0.0001186803, 0.0044412017]
d[3] = [0.0009559691, 0.0017743558, 0.0, 0.0013032928, 0.0023678839, 0.0041616186, 0.0019502416, 0.0017919093, 0.0015158206, 0.0014498606, 0.003790319]
d[4] = [0.00039163232, 0.0019534677, 0.0, 0.012159422, 0.00055508316, 0.0040820688, 0.014856763, 0.0039739013, 0.0001296252, 0.0088003799, 0.001960218]
d[5] = [0.00053822994, 0.0006852597, 0.0, 0.00081842393, 0.0007095933, 0.0016770288, 0.0011497289, 0.00037362427, 0.00078949332, 0.001386337, 0.00058346987]
d[6] = [0.018143773, 0.004214108, 0.0, 0.050536297, 0.0056488216, 0.041602179, 0.025937408, 0.043233477, 0.0092135817, 0.036152668, 0.004325211]
d[7] = [0.0090920031, 0.0017567724, 0.0, 0.022203699, 0.0044720918, 0.006667085, 0.022469379, 0.021180645, 0.0087945312, 0.013434723, 0.005915761]
d[8] = [0.012895048, 0.0040948093, 0.0, 0.0060472116, 0.0010125935, 0.004688397, 0.0076433569, 0.0091721192, 0.0073656142, 0.0091525838, 0.0048410892]
d[9] = [0.0014486909, 0.0037166029, 0.0, 0.0052391812, 0.00045517087, 0.0063833147, 0.0048678219, 0.0092318505, 0.0024903119, 0.0071937293, 0.0010662079]
d[10] = [0.0022007525, 0.00079360604, 0.0, 0.0015201867, 0.0016310215, 0.00095891207, 0.0027072802, 0.0015228242, 0.00053489208, 0.00013497472, 0.0024524927]
d[11] = [0.0029843152, 0.00028859079, 0.0, 0.0092552453, 0.00015974045, 0.0047836453, 0.0060867295, 0.019476719, 0.0047390759, 0.015177488, 0.00684762]
d[12] = [0.0034172833, 0.003230527, 0.0, 0.014105573, 0.0033515841, 0.017508239, 0.0024132729, 0.014280505, 0.0031297803, 0.0027508885, 0.017104387]
d[13] = [0.0012870133, 0.00027030706, 0.0, 0.0014535114, 0.0011903048, 0.00081889331, 0.00067373365, 0.0002046451, 0.0020390004, 0.00044480711, 0.00088638067]
d[14] = [0.0011057854, 0.00015234947, 0.0, 0.0067916363, 0.001093179, 0.0069456995, 0.0051265359, 0.0095151067, 0.0010976642, 0.0047262236, 0.0015255809]
d[15] = [0.0039838254, 0.0024195611, 0.0, 0.01245001, 0.004886806, 0.0088336989, 0.01125443, 0.0084168538, 0.0086370856, 0.0082981363, 0.00012540817]

'''Get the df using the dict, and melt it for boxplot'''
df = pd.DataFrame.from_dict(d, orient='columns', dtype=None)
df_m = pd.melt(df, id_vars=["index"], value_vars=df.columns.tolist())
# note: first 2 elements in columns list are "level_0" and index" now
df_m.columns = ["index", "Features", "Importance Score"]
df_m = df_m[["Feature", "Importance Score"]]

'''Make the boxplot'''
plt.plot(figsize=(16,5))
sns.set(font_scale=1.5)
sns.boxplot(x="Features", y="Importance Score", data=df_m)
plt.xticks(rotation=85)
plt.tight_layout()
plt.savefig(os.path.join(gv.outputDir, "imp_scores_boxplot.pdf"))

'''Get the gv.df_dhss, and gv.df_tfs, and shorten their column names'''
dhss = gv.df_dhss.reset_index()
dhss["DHS_sites"] = ["{},{:.2f}".format(a,b) for a,b in zip(dhss["loc"].tolist(), dhss["pcc"].tolist())]
dhss = dhss.set_index("DHS_sites")
dhss = dhss.drop(["loc", "pcc"], axis=1)
dhss.head()

'''similarly for thd df_tfs'''
tfs = gv.df_tfs.reset_index()  # new cols: geneName loc TAD_loc zscore  cn_corr pcc
tfs["TFs"] = ["{},{:.2f}".format(a,b) for a,b in zip(tfs["geneName"].tolist(), tfs["pcc"].tolist())]
tfs = tfs.set_index("TFs")
tfs = tfs.drop(["geneName", "loc", "TAD_loc", "zscore", "cn_corr", "pcc"], axis=1)
tfs.head()

'''Plot the heatmap'''
plt.figure(figsize=(16, 2))
plt.subplot(1,3,1)
sns.heatmap(dhss.transpose(), yticklabels=False)
plt.xticks(rotation=85)
plt.subplot(1,3,2)
sns.heatmap(tfs.transpose(), yticklabels=False, vmax=9, vmin=4.0)
plt.xticks(rotation=85)
plt.subplot(1,3,3)
sns.heatmap(gv.goi.to_frame(), yticklabels=False, vmax=9, vmin=4.0, xticklabels=False)
plt.xticks(rotation=85)
plt.savefig(os.path.join(gv.outputDir, "new_input_feats_and_goi.pdf"))

====================================================================================
====================================================================================

