## Cross cell-type prediction of gene expression

This program uses cross cell/tissue-type cis-TSS chromatin accessibility and known TF-TG (i.e. transcription factor to target gene) interactions for a given gene of interest to predict its expression in a new [or given] cell line of interest (not used in training).

Please see the [wiki page](https://github.com/dineshmdh/predicting_gene_expression/wiki/Cross-cell-type-prediction-of-gene-expression-WIKI) for details on the modeling framework, comparison against state-of-the-art and some current drawbacks.

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
To run the program:
1. Make sure these python modules are installed.
2. Download and unzip source code.
3. Download and unzip the input files from
[this dropbox link](https://www.dropbox.com/sh/6u9dth28lx6g5as/AACpwMTNjGuctIYN8fA-BOzRa?dl=0 "dropbox link for large files"). (These are the processed Roadmap and CellNet files, and were a bit too large to share on this repo.)
4. Move these input files in a directory named `Input_files` in the main directory for the package.
5. Go to the /directory-to-package/Functions/model_scripts, and call the python script below.

## Usage
```python
usage: main.py [-h] [-d DISTANCE] [-u USE_TAD_INFO]
               [-dl PCC_LOWERLIMIT_TO_FILTER_DHSS]
               [-Fd TAKE_THIS_MANY_TOP_DHSS] [-rd]
               [-Ft TAKE_THIS_MANY_TOP_TFS] [-tff {pcc,zscore}]
               [-tfl LOWERLIMIT_TO_FILTER_TFS] [-rt] [-w {random,corr}]
               [-m MAX_ITER] [-o OUTPUTDIR] [-k RUN_ID] [-p]
               gene
```
Examples:
```python
python main.py SIX1  # run with default parameters
python main.py SIX1 -d 200 -Fd 10  # consider top 10 DHS sites in TSS +/- 200kb region that most correlate with the expression of SIX1 across the samples
python main.py SIX1 -Ft 20 -m 200 -o /output/dir # consider top 6 DHS sites (by default) and top 20 TFs that are predicted to regulate the target gene - SIX1. Run 500 iterations (instead of default, 300) for training the neural net. Send all output to the designated directory.
```

Arguments (Abbrv) | Arguments (Full) | Details
--- | --- | ---
`gene` | `gene` | Gene of interest
`-h` | `--help` | show this help message and exit
`-d DISTANCE` | `--distance DISTANCE` | Distance from TSS (in kb) to cover as region of interest (Default: 150)
`-u USE_TAD_INFO` | `--use_tad_info USE_TAD_INFO` | Use TAD boundaries to demarcate the boundaries for the region of interest. (Default: True)
`-Fd TAKE_THIS_MANY_TOP_DHSS` | `--take_this_many_top_dhss TAKE_THIS_MANY_TOP_DHSS` | Take this many DHS sites. If this is set to '-1', then all the known DHS sites in the region TSS +/- --distance or regulatory TFs is used. Note that if random set of features are to be used, namely by setting '-rd' option, then the same number of DHS sites are considered as in the non-random (i.e. original) set. (See details on '-rd' below.) (Default: 6)
`-dl PCC_LOWERLIMIT_TO_FILTER_DHSS` | `--pcc_lowerlimit_to_filter_dhss PCC_LOWERLIMIT_TO_FILTER_DHSS` | Lower limit of the absolute pearson correlation between the DHS site accessibility and expression of this gene to be used in filtering the top dhs sites. All DHS sites with pearson correlation score below this threshold are ignored. (Note, if more than `-F TAKE_THIS_MANY_TOP_DHS_FTS` remain, only top `-F TAKE_THIS_MANY_TOP_DHS_FTS` are selected based on the pearson correlation scores. (Default: 0.2)
`-rd` | `--use_random_DHSs` | If set, a set of --take_this_many_top_dhs_fts number of DHS sites are randomly selected from the genome. The DHS sites selected could be from a different TAD domain or chromosome. (Default: False)
`-tff {pearson_corr,zscores}` | `--filter_tfs_by {pearson_corr,zscores}` | For the TF-TG association, filter the predicted list of regulatory TFs for the given gene using one of two measures: 1) Pearson Correlation Coefficient between the expression of TF and the target gene TG, or 2) Z-score indicating the significance of one TF-TG association given perturbation measurements of the expression of the TF and the TG across various experimental or biological conditions (see CellNet paper and CLR algorithm). (Default: 'zscores')
 `-Ft` | `--take_this_many_top_tfs TAKE_THIS_MANY_TOP_TFS` | Take this many TFs. If this is set to '-1', then all the known TFs that are predicted to be regulatory <form></form> the gene are used. Note that if random set of features are to be used, namely by setting '-rt' option, then the same number of TFs are considered as in the non- random (i.e. original) set. (See details on '-rt' below.) (Default: 8)
`-tfl LOWERLIMIT_TO_FILTER_TFS` | `--lowerlimit_to_filter_tfs LOWERLIMIT_TO_FILTER_TFS` | Lower limit of the measure --filter-tfs-by in absolute value. The value should be >0 for '--filter-tfs-by pearson_corr' and >= 4.0 for '--filter-tfs-by zscores'. Note that the respective upper limits are 1.0 and infinity respectively, and therefore need not be declared. (Default: 5.0 for the default '--filter-tfs-by zscores'.)
`-w {random,corr}` | `--init_wts_type {random,corr}` | Relates to the initial wts set between the nodes. If 'random', random initial wts are set between any two nodes; if 'corr', initial wts between input and hidden nodes are set to the correlation values between the node feature and the expression of the gene, and the initial weights between hidden layers or the hidden layer and output is set to 0.5 (Default: 'corr')
`-m MAX_ITER` | `--max_iter MAX_ITER` | Maximum number of interations for neural net optimization (Default: 300)
`-o OUTPUTDIR` | `--outputDir OUTPUTDIR` | Output directory. A directory for this gene of interest and set of parameters used is created at this location. (Default is '../Output')
`-k RUN_ID` | `--run_id RUN_ID` | Run_id for multiple parallel runs. This is useful in slurm. (Default: -1)
 `-p` | `--plot_all` | If set, all supplemental plots are also generated in addition to the scatterplots showing the performances after hyperparameter optimization and re-training (i.e. training the full training and validation set using the optimized hyperparameters). (Default: Not set)

## Output examples

<img src="https://github.com/dineshmdh/predicting_gene_expression/blob/master/Images/res_example1.png" alt="Real vs predicted expression estimates for BIRC5 gene" style="width:65px; height:60px;" />

In the prediction scatterplot example above, each dot represents one of 127 Roadmaps Epigenomics cell/tissue type. The pink, magenta and blue dots represent samples used in training, validation and test sets respectively. (The validation and test samples were not used in training.) The title of the plot shows the median test percentage error, i.e. median of (real expression - predicted expression) / real expression for all the cell types in the held out test group. (It should be noted that the 127 samples were grouped into 19 groups altogether on the basis of their developmental origin or lineages. The training-validation-testing partions in this program uses leave-one-group-out procedure. Hence, no muscle cell types were used in training for the following gene, for instance.)


![Feature importance scores for TNNC2](https://github.com/dineshmdh/predicting_gene_expression/blob/master/Images/feature_importance_scores_for_TNNC2.png "Feature importance scores for TNNC2")

The figure above shows the importance scores computed for the features used in predicting TNNC2 under joint (or DHSs-plus-TFs) model. Importance scores were obtained by imputing feature values (to an array of 0) and computing change in error. Heatmap shows the log transformed DNase-seq or RNA-seq signal across all Roadmap samples for the corresponding feature. For each feature, the suffix shows the Pearson correlation coefficient between the feature signal and TNNC2 gene expression (shown on the right).

## Contact
Please email dm237 [at] duke [dot] edu with any question(s), or idea(s).
