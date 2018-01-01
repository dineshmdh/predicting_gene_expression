# Created on October 28, 2017

import re
import pandas as pd
import collections as col


class GetLog(object):

    def __init__(self, inFile):

        handleIn = open(inFile)
        lines = handleIn.readlines()
        handleIn.close()

        self.fileName = re.split("/", inFile)[-2]
        self.gene_wCond = self.get_geneName_wCond()
        self.isTainted = False  # True if there is "Traceback" or any error encountered in the text
        self.test_samples = []
        self.tfs_selected = col.OrderedDict()  # {tf: corrVal}
        self.random_tfs_selected = col.OrderedDict()  # {tf: corrVal}

        self.perf_dict = col.OrderedDict()  # will be used to make pandas df later
        self.perf_dict["gene"] = []
        self.perf_dict["mode"] = []
        self.perf_dict["lr"] = []
        self.perf_dict["train_pcc"] = []
        self.perf_dict["train_loss"] = []
        self.perf_dict["test_pcc"] = []
        self.perf_dict["test_loss"] = []

        self.dict_ns = col.OrderedDict()  # smilar to self.perf_dict; ns = node_size
        '''self.dict_ns["gene"] = []
                                self.dict_ns["mode"] = []
                                self.dict_ns["lr"] = []
                                self.dict_ns["layer1_nodes"] = []
                                self.dict_ns["layer1_sizes"] = []
                                self.dict_ns["layer2_sizes"] = []'''

        self.main(lines)
        self.df_perf = self.get_df_perf()
        # self.df_ns = pd.DataFrame.from_dict(self.dict_ns, orient='index')  # Note has keys() as rows and values() as columns
        # self.df_ns.columns = ["values"]

    def get_geneName_wCond(self):
        '''The geneName_and_condition variable will be used to make the perf df later for boxplot.
        Return fileName without "runXX" info. Examples of fileName: SKI_150kb_T1.0_t0.3_cWts_m800_rDHSs_run440_s
        DES_150kb_T1.0_t0.3_rWts_m800_run101'''
        vals = re.split("_", self.fileName)
        vals = [x for x in vals if not x.__contains__("run")]
        return "_".join(vals)

    def get_command_line_arguments(self, aline):
        ''' The line is of this format:
        2017-10-28 04:13:58,274: __main__    : INFO    : Command line arguments: Namespace(distance=150, enforce_corr_thresholding=False, frac_test=0.2, gene='PRDM2', hidden_size=60, init_wts_type='corr', learning_rates=[0.5, 0.05, 0.005], max_iter=800, outputDir='/gpfs/fs0/data/gordanlab/dinesh/predicting_gex/Functions/../Output', slurmrank=1, take_log2_tpm=True, take_this_many_top_fts=20, take_top_fts=True, tf_corr_threshold_high=1.0, tf_corr_threshold_low=0.3, to_seed=True, use_random_DHSs=False, use_random_TFs=False, use_tad_info=True)
        '''
        args = {}
        aline = aline[aline.index("Namespace") + len("Namespace("): -1]
        p1, p2, p3 = re.split("\[|\]", aline)  # p == part
        args["learning_rates"] = p2
        p1, p3 = p1[:-len(", learning_rates=")], p3[2:]
        aline = "{}, {}".format(p1, p3)
        key_values = re.split(", ", aline)
        for akv in key_values:
            k, v = re.split("=", akv)
            args[k] = v
        return args

    def parse_performance(self, aline):
        '''aline is of this format:
        2017-10-28 03:09:05,244: __main__    : INFO    :     gene:PRDM2, mode:dhss, lr:0.005, train_pcc:0.723, train_loss:0.112, test_pcc:0.731, test_loss:0.251
        '''
        aline = aline[aline.index("gene:"):]
        kvs = re.split(", ", aline)  # kvs = keys and values
        for akv in kvs:
            k, v = re.split(":", akv)  # eg. k=mode, v=dhss
            if (k.__contains__("pcc") or (k.__contains__("loss"))):
                self.perf_dict[k].append(float(v))
            else:
                self.perf_dict[k].append(v)

    def parse_node_sizes(self, aline):
        '''aline is of this format:
        2017-10-28 17:21:49,937: __main__    : INFO    :     gene:SKI, mode:dhss, lr:0.05, layer1_node_and_sizes: [('chr1:2047317-2047617', 1.679), ('chr1:2063814-2064114', 2.042), ('chr1:2101672-2101972', 3.071), ('chr1:2112013-2112313', 1.541), ('chr1:2136338-2136638', 3.708), ('chr1:2144196-2144496', 2.994), ('chr1:2174751-2175051', 0.319), ('chr1:2177763-2178063', 11.688), ('chr1:2210701-2211001', 8.421), ('chr1:2222191-2222491', 5.301), ('chr1:2231693-2231993', 2.702), ('chr1:2232351-2232651', 2.363), ('chr1:2243575-2243875', 2.912), ('chr1:2267323-2267623', 1.844), ('chr1:2292256-2292556', 0.422), ('chr1:2304265-2304565', 3.982), ('chr1:2305475-2305775', 0.835), ('chr1:2349674-2349974', 6.961), ('chr1:2379431-2379731', 3.35), ('chr1:2381195-2381495', 5.142)], layer2_node_sizes: [1.884, 2.357, 3.627, 2.09, 0.74, 0.075, 1.188, 0.98, 1.193, 1.883, 1.11, 0.923]

        This function updates self.dict_ns with node sizes and other infos.
        '''
        aline = aline[aline.index("gene"):]
        parts = re.split("\[|\]", aline)

        # will have 5 parts; p4 (0-indexed) is empty string (""); p1 and p3 (0-indexed) have node_sizes
        p1 = ";".join(re.split(", ", parts[0])[:-1])  # eg. parts[0] = 'gene:SKI, mode:dhss, lr:0.05, layer1_node_and_sizes: '
        mode, lr = re.split(";|:", p1)[3], float(re.split("_|:", p1)[-1])
        akey = tuple((mode, lr))
        self.dict_ns[akey] = {}  # node labels and sizes will be saved here for (mode, lr) key

        p2_labels = []
        p2_ns = []  # ns == node sizes
        p2 = re.split("\(|\)", parts[1])  # ["", "'chr1:2047317-2047617', 2.927", ", ", "'chr1:2063814-2064114', 6.067", .., ""]
        p2 = [x for x in p2 if len(x) > 4]  # Note: some TFs might not have a long name (but 4 should be fine)
        for i, ap in enumerate(p2):  # eg. "'chr1:2349674-2349974', 4.271"
            aps = re.split("'|, ", ap)  # eg. ['', 'chr1:2047317-2047617', '', '2.927']
            p2_labels.append(aps[1])
            p2_ns.append(float(aps[3]))
        self.dict_ns[akey]["layer1_nodes"] = p2_labels
        self.dict_ns[akey]["layer1_sizes"] = p2_ns

        p3 = [float(x) for x in re.split(", ", parts[3])]  # eg. of parts[3] == 0.229, 2.268, 1.146, 2.606, 0.514, 0.313, 0.915, 0.499, 0.778, 1.107, 1.681, 1.102
        self.dict_ns[akey]["layer2_sizes"] = p3

    def get_df_perf(self):
        df_perf = pd.DataFrame(self.perf_dict, columns=self.perf_dict.keys())
        df_perf.sort_values(by=["mode", "test_loss"], inplace=True)

        df_perf["test_cbyr"] = df_perf["test_loss"] / df_perf["test_pcc"]
        df_perf["train_cbyr"] = df_perf["train_loss"] / df_perf["train_pcc"]
        df_perf["train_and_test_cbyr"] = (df_perf["train_loss"] / df_perf["train_pcc"]) + (df_perf["test_loss"] / df_perf["test_pcc"])
        return df_perf

    def get_best_perf(self, mode, by="train_and_test_cbyr"):
        '''Return the best performance from self.df_perf that fit the "mode" and "by" parameters.'''
        perfs = self.df_perf[self.df_perf["mode"] == mode][by].tolist()
        if (by in ["train_pcc", "test_pcc"]):
            return max(perfs)
        return min(perfs)

    def main(self, lines):
        for aline in lines:
            if (aline.__contains__("Traceback") or aline.__contains__("Error") or aline.__contains__("error")):
                self.isTainted = True

            if (aline.__contains__("Command line arguments")):
                self.args = self.get_command_line_arguments(aline.strip())

            if (aline.__contains__("Test sample")):  # eg. ..: INFO    :     Test sample 0: MCF-7.rep1
                aline = aline[aline.index("Test sample"):].strip()
                self.test_samples.append(re.split(": ", aline)[1])

            if (aline.__contains__("    TF selected")):  # ..getTF_gexes: INFO    :     TF selected 31, EBF1:0.57
                tf, corr = re.split(":", re.split(", ", aline.strip())[1])
                self.tfs_selected[tf] = float(corr)

            if (aline.__contains__("    random TF selected")):
                tf, corr = re.split(":", re.split(", ", aline.strip())[1])
                self.random_tfs_selected[tf] = float(corr)

            if (aline.__contains__("train_pcc")):
                self.parse_performance(aline.strip())  # updates self.perf_dict

            if (aline.__contains__("layer1_node_and_sizes")):
                self.parse_node_sizes(aline.strip())


if __name__ == "__main__":
    inFile = "/Users/Dinesh/Dropbox/Github/predicting_gex_with_nn_v2/Output/gata2/GATA2_150kb_T1.0_t0.3_rWts_m800_run130/GATA2.log"
    log = GetLog(inFile)

    '''print(log.args)
                print(log.fileName)
                print(log.test_samples)
                print(log.tfs_selected)
                print(log.random_tfs_selected)
                '''
    print(log.dict_ns.keys())
    print(log.dict_ns[tuple(("tfs", 0.050))]["layer1_sizes"])
    print(log.df_perf)
