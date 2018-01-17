## Cross cell-type prediction of gene expression

This program uses cross cell/tissue-type cis-TSS chromatin accessibility and known TF-TG (i.e. transcription factor to target gene) interactions for a given gene of interest to predict its expression in a new [or given] cell line of interest (not used in training).

## Dependencies and Installation
The program has been developed and tested in `Python v2.7` that comes with `Anaconda v4.4.7`. The following Python packages are required:
```python
numpy, pandas, hyperopt, matplotlib, seaborn, logging, functools, pybedtools, re
```
We recommend using Anaconda because it not only comes with some of these packages installed, but it is also easy to install the rest.
```
conda install -c jaikumarm hyperopt
conda install -c anaconda seaborn
conda install -c hargup/label/pypi logging
conda install -c travis functools
conda install -c bioconda pybedtools
```
To run the program, download and unzip source code, go to the Functions/model_scripts, and call the python script below.

## Usage
```python
python main.py [-h] [-d DISTANCE] [-u USE_TAD_INFO]
                  [-F TAKE_THIS_MANY_TOP_DHS_FTS] [-rd]
                  [-tff {pearson_corr,zscores}]
                  [-tfl LOWERLIMIT_TO_FILTER_TFS] [-w {random,corr}] [-s]
                  [-m MAX_ITER] [-o OUTPUTDIR] [-k RUN_ID]
                  gene

```
Arguments (Abbrv) | Arguments (Full) | Details
--- | --- | ---
`gene` | `gene` | Gene of interest
`-h` | `--help` | show this help message and exit
`-d DISTANCE` | `--distance DISTANCE` | Distance from TSS (in kb) to cover as region of interest (Default: 150)
`-u USE_TAD_INFO` | `--use_tad_info USE_TAD_INFO` | Use TAD boundaries to demarcate the boundaries for the region of interest. (Default: True)
`-F TAKE_THIS_MANY_TOP_DHS_FTS` | `--take_this_many_top_dhs_fts TAKE_THIS_MANY_TOP_DHS_FTS` | Take this many DHS sites that are most correlated (in absolute value) with the expession of the gene. If this is set to '-1', then all the known DHS sites in the region TSS +/- --distance is used. (Default: 20)
`-dl PCC_LOWERLIMIT_TO_FILTER_DHSS` | `--pcc_lowerlimit_to_filter_dhss PCC_LOWERLIMIT_TO_FILTER_DHSS` | Lower limit of the absolute pearson correlation between the DHS site accessibility and expression of this gene to be used in filtering the top dhs sites. All DHS sites with pearson correlation score below this threshold are ignored. (Note, if more than `-F TAKE_THIS_MANY_TOP_DHS_FTS` remain, only top `-F TAKE_THIS_MANY_TOP_DHS_FTS` are selected based on the pearson correlation scores. (Default: 0.2)
`-rd` | `--use_random_DHSs` | If set, a set of --take_this_many_top_dhs_fts number of DHS sites are randomly selected from the genome. The DHS sites selected could be from a different TAD domain or chromosome. (Default: False)
`-tff {pearson_corr,zscores}` | `--filter_tfs_by {pearson_corr,zscores}` | For the TF-TG association, filter the predicted list of regulatory TFs for the given gene using one of two measures: 1) Pearson Correlation Coefficient between the expression of TF and the target gene TG, or 2) Z-score indicating the significance of one TF-TG association given perturbation measurements of the expression of the TF and the TG across various experimental or biological conditions (see CellNet paper and CLR algorithm). (Default: 'zscores')
`-tfl LOWERLIMIT_TO_FILTER_TFS` | `--lowerlimit_to_filter_tfs LOWERLIMIT_TO_FILTER_TFS` | Lower limit of the measure --filter-tfs-by. The value should be >0 for '--filter-tfs-by pearson_corr' and >= 4.0 for '--filter-tfs-by zscores'. Note that the respective upper limits are 1.0 and infinity respectively, and therefore need not be declared. (Default: 5.0 for the default '--filter-tfs-by zscores'.)
`-w {random,corr}` | `--init_wts_type {random,corr}` | Relates to the initial wts set between the nodes. If 'random', random initial wts are set between any two nodes; if 'corr', initial wts between input and hidden nodes are set to the correlation values between the node feature and the expression of the gene, and the initial weights between hidden layers or the hidden layer and output is set to 0.5 (Default: 'corr')
`-m MAX_ITER` | `--max_iter MAX_ITER` | Maximum number of interations for neural net optimization (Default: 300)
`-o OUTPUTDIR` | `--outputDir OUTPUTDIR` | Output directory. A directory for this gene of interest and set of parameters used is created at this location. (Default is '../Output')
`-k RUN_ID` | `--run_id RUN_ID` | Run_id for multiple parallel runs. This is useful in slurm. (Default: -1)

## Output example and preliminary result

![Real vs predicted expression estimates for BIRC5 gene.](https://github.com/dineshmdh/predicting_gene_expression/blob/master/Images/res_example1.png "An output example")

In the prediction scatterplot example above, each dot represents one of 127 Roadmaps Epigenomics cell/tissue type. The pink, magenta and blue dots represent samples used in training, validation and test sets respectively. (The validation and test samples were not used in training.) The title of the plot shows the median test percentage error, i.e. median of (real expression - predicted expression) / real expression for all the cell types in the held out test group. (It should be noted that the 127 samples were grouped into 19 groups altogether on the basis of their developmental origin or lineages. The training-validation-testing partions in this program uses leave-one-group-out procedure. Hence, no muscle cell types were used in training for the following gene, for instance.)

![Preliminary test results on 261 genes](https://github.com/dineshmdh/predicting_gene_expression/blob/master/Images/res_med_test_error.png "Test results on 261 genes")

The overlapping density plot above is an aggregate plot for all such median test percentage errors for a small subset of genes. (These genes are obtained from [this paper](https://academic.oup.com/nar/article/45/20/11684/4107215 "MyoD paper") that are collectively either reprogrammed or non-reprogrammed during a myogenic transdifferentiation study.) The median test percentage error across all of these test groups is just 8.1 +/- 6.7 %.


## Contact
Please email dm237 [at] duke.edu with any question(s), or idea(s).
