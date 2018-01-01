# Created on October 29, 2017

import os
import re
from get_log_class import GetLog


class Pool_logs_for_a_gene(object):

    def __init__(self, gene, rank_lower_lim, rank_upper_lim, use_nonRanked, inputDir):

        self.gene = gene
        self.rank_upper_lim = rank_upper_lim  # inclusive
        self.rank_lower_lim = rank_lower_lim  # inclusive
        self.use_nonRanked = use_nonRanked  # if True, any directory w/o "_run" will also included; if we only want to get such directories, set the ranks to < 0.
        self.inputDir = inputDir  # directory containing the results
        self.logFile_dirs = self.get_logFile_dirs()
        self.logs = self.get_log_objects()

    def get_logFile_dirs(self):
        '''Get the right directories in self.inputDir'''
        logFile_dirs = []
        all_dirs = os.listdir(self.inputDir)
        logFile_dirs = []
        for adir in all_dirs:
            if (not adir.__contains__(self.gene)):
                continue
            if (adir.__contains__("_run")):
                vals = re.split("_", adir)
                run_index = -1
                for aval in vals:
                    if (aval.__contains__("run")):
                        run_index = int(aval[3:])
                if (run_index < 0):
                    raise Exception("No run_index found..")
                if (run_index >= self.rank_lower_lim) and (run_index <= self.rank_upper_lim):
                    logFile_dirs.append(os.path.join(self.inputDir, adir, self.gene + ".log"))
            else:
                if (self.use_nonRanked):
                    logFile_dirs.append(os.path.join(self.inputDir, adir, self.gene + ".log"))
        return logFile_dirs

    def get_log_objects(self):
        logs = []
        for alogDir in self.logFile_dirs:
            logFile = os.path.join(alogDir)
            alog = GetLog(logFile)
            if (alog.isTainted):
                print(alog.fileName, "is tainted")
            else:
                logs.append(alog)
        return logs


if __name__ == "__main__":
    pls = Pool_logs_for_a_gene("DES", 101, 110, True, "/Users/Dinesh/Dropbox/Github/predicting_gex_with_nn_v2/Output/des")  # /PRDM2_150kb_T1.0_t0.3_cWts_m800_run")  # pls = pooled logs

    print(pls.logs[0].df_perf)
    print(pls.logs[0].get_best_perf(mode="dhss_tfs"))
    # print(pls[9].df_perf)
