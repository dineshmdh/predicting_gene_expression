## Cross cell-type prediction of gene expression

This program uses cross cell/tissue-type cis-TSS chromatin accessibility and known TF-TG (i.e. transcription factor to target gene) interactions for a given gene of interest to predict its expression in a new [or given] cell line of interest (not used in training).


### USAGE
```python
python main_18.py [-h] [-d DISTANCE] [-u USE_TAD_INFO]
                  [-F TAKE_THIS_MANY_TOP_DHS_FTS] [-rd]
                  [-tff {pearson_corr,zscores}]
                  [-tfl LOWERLIMIT_TO_FILTER_TFS] [-w {random,corr}] [-s]
                  [-m MAX_ITER] [-o OUTPUTDIR] [-k RUN_ID]
                  gene

positional arguments:
  gene                  Gene of interest

optional arguments:
  -h, --help            show this help message and exit
  -d DISTANCE, --distance DISTANCE
                        Distance from TSS (in kb) to cover as region of
                        interest (Default: 150)
  -u USE_TAD_INFO, --use_tad_info USE_TAD_INFO
                        Use TAD boundaries to demarcate the boundaries for the
                        region of interest. (Default: True)
  -F TAKE_THIS_MANY_TOP_DHS_FTS, --take_this_many_top_dhs_fts TAKE_THIS_MANY_TOP_DHS_FTS
                        Take this many DHS sites that are most correlated (in
                        absolute value) with the expession of the gene. If
                        this is set to '-1', then all the known DHS sites in
                        the region TSS +/- --distance is used. (Default: 20)
  -rd, --use_random_DHSs
                        If set, a set of --take_this_many_top_dhs_fts number
                        of DHS sites are randomly selected from the genome.
                        (Default: False)
  -tff {pearson_corr,zscores}, --filter_tfs_by {pearson_corr,zscores}
                        For the TF-TG association, filter the predicted list
                        of regulatory TFs for the given gene using one of two
                        measures: 1) Pearson Correlation Coefficient between
                        the expression of TF and the target gene TG, or 2)
                        Z-score indicating the significance of one TF-TG
                        association given perturbation measurements of the
                        expression of the TF and the TG across various
                        experimental or biological conditions (see CellNet
                        paper and CLR algorithm). (Default: 'zscores')
  -tfl LOWERLIMIT_TO_FILTER_TFS, --lowerlimit_to_filter_tfs LOWERLIMIT_TO_FILTER_TFS
                        Lower limit of the measure --filter-tfs-by. The value
                        should be >0 for '--filter-tfs-by pearson_corr' and >=
                        4.0 for '--filter-tfs-by zscores'. Note that the
                        respective upper limits are 1.0 and infinity
                        respectively, and therefore need not be declared.
                        (Default: 5.0 for the default '--filter-tfs-by
                        zscores'.)
  -w {random,corr}, --init_wts_type {random,corr}
                        Relates to the initial wts set between the nodes. If
                        'random', random initial wts are set between any two
                        nodes; if 'corr', initial wts between input and hidden
                        nodes are set to the correlation values between the
                        node feature and the expression of the gene, and the
                        initial weights between hidden layers or the hidden
                        layer and output is set to 0.5 (Default: 'corr')
  -s, --to_seed         If set, numpy seed number is set (only) for random
                        splitting of the training and test samples. The seed
                        number is set to 4. (Default: False)
  -m MAX_ITER, --max_iter MAX_ITER
                        Maximum number of interations for neural net
                        optimization (Default: 300)
  -o OUTPUTDIR, --outputDir OUTPUTDIR
                        Output directory. A directory for this gene of
                        interest and set of parameters used is created at this
                        location. (Default is '../Output')
  -k RUN_ID, --run_id RUN_ID
                        Run_id for multiple parallel runs. This is useful in
                        slurm. (Default: -1)

```



