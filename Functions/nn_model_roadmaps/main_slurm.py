# Created on October 22, 2017; updated on October 27, 2017

import os

rank = os.environ["SLURM_ARRAY_TASK_ID"]  # first 100 are for original DHSs and TFs
genes = ["SKI", "CDK4", "PRDM2", "DES", "GATA2"]
outputDir = "/data/gordanlab/dinesh/predicting_gex/Output"

for agene in genes:
    cmd_basic = "python main.py {} -t 0.3 -lr 0.5 0.05 0.005 -m 800".format(agene)
    if (int(rank) <= 50):  # same train/test (with -s); init_wts_type == corr
        cmd = "{} -w corr -s".format(cmd_basic)
    elif (int(rank) > 50) and (int(rank) <= 100):  # same train/test (with -s); init_wts_type == random
        cmd = "{} -w random -s".format(cmd_basic)
    elif (int(rank) > 100) and (int(rank) <= 200):  # different train/test (without -s); init_wts_type == corr
        cmd = "{} -w random".format(cmd_basic)
    else:
        if (int(rank) > 200) and (int(rank) <= 400):
            cmd = "{} -w corr -rt -en -s".format(cmd_basic)  # random TFs, corr based thresholding enforced
        if (int(rank) > 400) and (int(rank) <= 600):
            cmd = "{} -w corr -rt -s".format(cmd_basic)  # random TFs, corr based thresholding not enforced
        if (int(rank) > 600) and (int(rank) <= 800):
            cmd = "{} -w corr -rd -en -s".format(cmd_basic)  # random DHSs, corr based thresholding enforced
        if (int(rank) > 800):
            cmd = "{} -w corr -rt -rd -s".format(cmd_basic)  # random DHSs and random TFs, corr based thresholding not enforced
    cmd = "{} -k {}".format(cmd, rank)
    os.system(cmd)
