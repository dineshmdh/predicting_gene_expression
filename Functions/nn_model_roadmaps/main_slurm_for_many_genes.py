# Created on October 30, 2017
import os

# ============ get genes ============

Bgenes_file = os.path.join(os.getcwd(), "../Input_files", "B.treatment.geneNamesOnly.txt")
Cgenes_file = os.path.join(os.getcwd(), "../Input_files", "C.treatment.geneNamesOnly.txt")

handleIn_bgenes = open(Bgenes_file)
bgenes = handleIn_bgenes.readlines()
handleIn_bgenes.close()

handleIn_cgenes = open(Cgenes_file)
cgenes = handleIn_cgenes.readlines()
handleIn_cgenes.close()

bgenes = [x.strip() for x in bgenes]
cgenes = [x.strip() for x in cgenes]
genes = bgenes + cgenes  # len = 100 + 168 = 268

# =====================================

rank = os.environ["SLURM_ARRAY_TASK_ID"]
thisGene = genes[int(rank)]
outputDir = "/data/gordanlab/dinesh/predicting_gex/Output_BCgenes"

cmd_basic = "python main.py {} -t 0.3 -lr 0.5 0.05 0.005 -o {} -m 800 -s".format(thisGene, outputDir)
os.system(cmd_basic)
