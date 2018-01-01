# Created on August 3, 2017

'''Given a df_enh_thisgene_info - which has fields chrom, enh_ss, enh_es, gene - the objective is to get laplace distribution
weights for the dhss around the TSS - which has weights in 100 bp (by default) distances. i.e. the output weight df has fields
chrom, ss, es, wt (where, ss and es are 100bp apart and overall cover the region of interest - or roi).
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns


class laplace_wts(object):
    def __init__(self, df_enh_thisgene_info, gene_ofInterest_info, gene_ofInterest, dist_lim_in_kb, laplace_scale=10, scale_wt_by=100):
        self.df_enh_thisgene_info = df_enh_thisgene_info
        self.gene_ofInterest = gene_ofInterest
        self.gene_ofInterest_info = gene_ofInterest_info

        self.dist_lim = dist_lim_in_kb
        self.dist_betn_dots = 100  # in bp
        self.num_items = 2 * self.dist_lim * 1000 / self.dist_betn_dots  # number of dots to plot in the scatterplot; times 2 b/c it's 500kb up and 500kb downstream;
        # doing this to ensure that there is at least one dot per dhs site (assuming min(dhs_len) >= 100bp)

        if (self.gene_ofInterest_info["strand_gene"] == "-"):
            self.laplace_mean = int(self.gene_ofInterest_info["es_gene"]) / 1000.  # 0 from the TSS; unit is kb
        else:
            self.laplace_mean = int(self.gene_ofInterest_info["ss_gene"]) / 1000.
        self.laplace_scale = laplace_scale  # looks decent for a 1 or 2mb span
        self.scale_wt_by = scale_wt_by

    def get_laplace_prob(self, mu, scale, x):
        toExp = -1.0 * abs(mu - x) / scale
        return (0.5 / scale) * np.exp(toExp)

    def get_ys_bg_obs_and_comb(self):  # bg == background / laplace prob; obs == observed; comb == combined (with prior laplace prob)
        '''assign background prob'''
        xs = np.linspace(self.laplace_mean - self.dist_lim, self.laplace_mean + self.dist_lim, self.num_items)
        ys_bg = [self.get_laplace_prob(self.laplace_mean, self.laplace_scale, x) for x in xs]

        '''create an observed "prob" array for known enhancer weights'''
        ys_obs = np.ones(len(xs)) * np.percentile(ys_bg, 70)  # this percentile and not lower b/c not all enhancers are known
        for i in range(0, self.df_enh_thisgene_info.shape[0]):
            this_enh = self.df_enh_thisgene_info.iloc[i]
            this_enh_mid = int(0.5 * (this_enh["enh_ss"] + this_enh["enh_es"])) / 1000.  # *0.001 to convert to kb
            xs_sub_enhMid = xs - this_enh_mid
            # get index array to find where the dhs is.
            thisEnh_ind_arr = [1 if (np.sign(xs_sub_enhMid[i]) != np.sign(xs_sub_enhMid[i + 1])) else 0 for i in range(0, len(xs) - 1)]
            if (sum(thisEnh_ind_arr) > 0):
                ys_obs[thisEnh_ind_arr.index(1)] = ys_bg[thisEnh_ind_arr.index(1) + 1] = np.percentile(ys_bg, 99)

        '''merge those two weights to create a combined prob measure'''
        comb = [ys_bg[i] + ys_obs[i] for i in range(0, len(xs))]  # comb = combined
        ys_comb = comb / np.sum(comb)

        return xs, ys_bg, ys_obs, ys_comb

    def plot_wts_prior_and_posterior(self, xs, ys_bg, ys_obs, ys_comb):
        '''plot the weight measures for the dhss relative to the gene tss'''
        plt.rcParams['savefig.dpi'] = 300
        sns.set(font_scale=1.5)

        fig = plt.figure(0, (15, 5))
        sns.set_style("whitegrid")

        plt.subplot(1, 2, 1)
        plt.scatter(xs, ys_bg, color="b", label="Background prob", alpha=0.5, s=3)
        plt.plot(xs, ys_obs, color="r", label="Known enhancers", alpha=0.5)  # ,s=3)
        plt.plot(xs, ys_comb, color="g", label="Combined prob")
        plt.legend()

        plt.xlim(min(xs), max(xs))
        plt.ylim((0, max(max(ys_comb), max(ys_obs), max(ys_bg))))
        plt.title("Gex ~ DHS for " + self.gene_ofInterest)
        plt.xlabel("Dist from TSS (in kb)")
        plt.ylabel("Probability (Gene ~ DHS)")

        plt.subplot(1, 2, 2)
        plt.plot(xs, ys_comb, color="g", label="Combined prob")
        plt.title("Gex ~ DHS for " + self.gene_ofInterest)
        plt.xlabel("Dist from TSS (in kb)")
        plt.ylabel("Probability (Gene ~ DHS)")

    def get_post_nnWts_df(self, xs, ys_comb):
        '''Converting the weights to a bed4 format dataframe that will be used to get the NN weights for the dhss later'''
        xs_bp = [int(x * 1000) for x in xs]
        ys_comb_scaled = [y * self.scale_wt_by for y in ys_comb]  # scaled so that the largest weight (which is a prob measure) is decently "large"

        list_weights = []
        for ax, ay in zip(xs_bp, ys_comb_scaled):
            list_weights.append([self.gene_ofInterest_info["chr_gene"], ax, ax + self.dist_betn_dots, ay])

        df_nnWts_byDistFromTSS = pd.DataFrame.from_records(list_weights, columns=["chrom", "ss", "es", "wt"])
        return df_nnWts_byDistFromTSS
