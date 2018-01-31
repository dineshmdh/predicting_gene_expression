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

== == == == == == == ==

WIKI

== == == == == == == ==


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


== == == == == == == == == == == ==
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


== == == == ==


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


== == =
