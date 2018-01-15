# Created on Jan 12, 2018
import os

# ============ get genes ============

Bgenes_file = os.path.join(os.getcwd(), "../../Input_files", "B.treatment.geneNamesOnly.txt")
Cgenes_file = os.path.join(os.getcwd(), "../../Input_files", "C.treatment.geneNamesOnly.txt")

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
thisGene = genes[int(rank) - 1]
outputDir = "/data/gordanlab/dinesh/pred_gex_v2/Output/BCgenes"

cmd_basic = "python main.py {} -o {} -k {}".format(thisGene, outputDir, rank)
os.system(cmd_basic)
