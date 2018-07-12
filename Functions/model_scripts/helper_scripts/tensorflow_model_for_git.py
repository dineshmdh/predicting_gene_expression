'''
Created 1/1/2018

__author__ = "Dinesh Manandhar"

Notes on implementation:

1. When the running "mode" is "dhss" or "tfs",
the trainX and testX matrices are "extracted" from the
joint df (i.e. df_dhss and df_tfs merged). This is because
the indexes for df_dhss and df_tfs are different before
merging and are processed during merging to have just the
name (i.e. dhs loc or geneName) and "confidence" measure
as a measure of its similarity with the gene expression
for the target gene. (Although this shouldn't matter much
for "dhss" or "tfs" mode, this will also help with plotting
the dataframes later.)
2. The features are expected to be (and are) sorted for df_dhss and
df_tfs that are merged to get the df_joint. This is because
while selecting df_tfs (for mode=="tfs"), top tfs need to
be selected. These dfs are sorted by their similarity measure
(discussed above) for when real features are used. When random
features are selected, no sorting is necessary or done.

3. (To do?) Optionally I could not compute loss_test in the
retraining step. I will be using rmse to assess model performance.
(Note: loss = rmse + regularization).

'''
import pdb
import logging
import re
import os
import tensorflow as tf
import numpy as np
import scipy.stats as stats
import seaborn as sns
from hyperopt import STATUS_OK, STATUS_FAIL
import matplotlib.pyplot as plt
from HPO_helper import LAMDA_BASE, LR_BASE

np.seterr(all='raise')
plt.switch_backend('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # To ignore TF warning in stdout (source: https://github.com/tensorflow/tensorflow/issues/7778)


class Tensorflow_model(object):
    def __init__(self, gv, mp, test_eid_group_index, mode):

        ######################################################
        # set the logging handlers and params
        formatter = logging.Formatter('%(asctime)s: %(name)-12s: %(levelname)-8s: %(message)s')

        file_handler = logging.FileHandler(os.path.join(gv.outputDir, gv.gene_ofInterest + '.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        ######################################################
        ########## set basic model params #########
        self.max_iter = gv.max_iter
        self.pkeep_train = 0.5
        # self.starter_learning_rate = 0.3  # updated by * 0.5 for re-training
        self.decay_at_step = 15  # learning rate is updated after this many training steps
        # self.use_sigmoid_h1 = True
        # self.use_sigmoid_h2 = True
        # self.use_sigmoid_yhat = False
        self.test_eid_group_index = test_eid_group_index
        assert mode in ["dhss", "tfs", "joint"]
        self.mode = mode  # one of "dhss", "tfs" or "joint"

        ########## from model preparation ##########
        self.val_eid_group = "ENCODE2012"  # will be used as validation set; always fixed.
        df_eids, self.train_eid_groups, self.test_eid_group = mp.get_train_and_test_eid_groups(
            mp.csv_eids, self.val_eid_group, test_group_index=self.test_eid_group_index)  # reads the file each time
        self.train_eids = mp.get_eids_for_a_group(df_eids, self.train_eid_groups)
        self.val_eids = mp.get_eids_for_a_group(df_eids, self.val_eid_group)
        self.test_eids = mp.get_eids_for_a_group(df_eids, self.test_eid_group)

        '''Get train and test X matrices (like train_dhss) for the models.
        Note these X matrices (like train_dhss) have fts as rows and samples as cols.
        These will be transposed to get train/test/X/Y matrices later.'''
        self.train_dhss, self.val_dhss, self.test_dhss, self.max_train_dhss, self.min_train_dhss = mp.get_normalized_train_val_test_dfs(gv.df_dhss, self.train_eids, self.val_eids, self.test_eids)
        self.train_tfs, self.val_tfs, self.test_tfs, self.max_train_tfs, self.min_train_tfs = mp.get_normalized_train_val_test_dfs(gv.df_tfs, self.train_eids, self.val_eids, self.test_eids)
        self.train_joint = mp.merge_dhs_and_tf_dfs(self.train_dhss, self.train_tfs, gv)  # for dhs+tf joint model
        self.val_joint = mp.merge_dhs_and_tf_dfs(self.val_dhss, self.val_tfs, gv)
        self.test_joint = mp.merge_dhs_and_tf_dfs(self.test_dhss, self.test_tfs, gv)

        '''Now get the train and test Y vectors.
        Note these Y "vectors" are pandas series objects.'''
        self.train_goi, self.val_goi, self.test_goi = mp.get_normalized_train_val_and_test_goi(gv, self.train_eids, self.val_eids, self.test_eids)
        #############################################

        ########## only consider joint model for now ##########
        if (mode == "joint"):
            self.trainX = np.array(self.train_joint.transpose())
            self.valX = np.array(self.val_joint.transpose())
            self.testX = np.array(self.test_joint.transpose())
        elif (mode == "dhss"):  # see Note num 2 above
            num_dhss = self.train_dhss.shape[0]
            self.trainX = np.array(self.train_joint.iloc[0:num_dhss, :].transpose())
            self.valX = np.array(self.val_joint.iloc[0:num_dhss, :].transpose())
            self.testX = np.array(self.test_joint.iloc[0:num_dhss, :].transpose())
        else:  # see Note num 2 above
            num_dhss = self.train_dhss.shape[0]
            self.trainX = np.array(self.train_joint.iloc[num_dhss:, :].transpose())
            self.valX = np.array(self.val_joint.iloc[num_dhss:, :].transpose())
            self.testX = np.array(self.test_joint.iloc[num_dhss:, :].transpose())

        self.trainY = np.array(self.train_goi.tolist())
        self.trainY = self.trainY.reshape(self.trainY.shape[0], -1)
        self.valY = np.array(self.val_goi.tolist())
        self.valY = self.valY.reshape(self.valY.shape[0], -1)
        self.testY = np.array(self.test_goi.tolist())
        self.testY = self.testY.reshape(self.testY.shape[0], -1)

        assert np.max(self.trainY) <= 1

        if (gv.plot_all):
            self.plot_train_test_XY(gv)
        ############## end of __init__() #############

    def init_nn_updates(self):
        nn_updates = {}
        nn_updates["loss_train"] = []  # loss = rmse + regularization
        nn_updates["loss_val"] = []
        nn_updates["pcc_train"] = []
        nn_updates["pcc_val"] = []
        nn_updates["learning_rate"] = []
        return nn_updates

    def train_tensorflow_nn(self, parameters):

        layer_sizes = [int(n) for n in parameters['layers']['n_units_layer']]
        if (len(layer_sizes) == 1):
            wts = self.get_random_wts(layer_sizes, use_h2=False)
        elif (len(layer_sizes) == 2):
            wts = self.get_random_wts(layer_sizes, use_h2=True)
        else:
            raise Exception()
        lamda = parameters['lamda']
        starter_learning_rate = parameters['starter_learning_rate']
        use_sigmoid_h1 = parameters['use_sigmoid_h1']
        use_sigmoid_h2 = parameters['use_sigmoid_h2']
        use_sigmoid_yhat = parameters['use_sigmoid_yhat']

        # ------ Variables and placeholders ------
        nn_updates = self.init_nn_updates()
        X = tf.placeholder(tf.float32, shape=[None, wts[1].shape[0]], name="X")
        Y = tf.placeholder(tf.float32, [None, 1], name="Y")  # true Ys
        pkeep = tf.placeholder(tf.float32, name="pkeep")

        W1 = tf.Variable(tf.cast(wts[1], tf.float32), name="W1")
        b1 = tf.Variable(tf.zeros([wts[1].shape[1]]), name="H1_bn_offset")  # bn == batch normalization
        g1 = tf.Variable(tf.ones([wts[1].shape[1]]), name="H1_bn_scale")
        W2 = tf.Variable(tf.cast(wts[2], tf.float32), name="W2")
        W3 = None
        b2 = None
        g2 = None
        if not (wts[3] is None):
            W3 = tf.Variable(tf.cast(wts[3], tf.float32), name="W3")
            b2 = tf.Variable(tf.zeros([wts[2].shape[1]]), name="H2_bn_offset")
            g2 = tf.Variable(tf.zeros([wts[2].shape[1]]), name="H2_bn_scale")

        Yhat, rmse, loss = self.tf_model(X, Y, W1, W2, W3, b1, g1, b2, g2, pkeep, lamda,
                                         use_sigmoid_h1, use_sigmoid_h2, use_sigmoid_yhat)  # loss = rmse + regularization
        pcc = tf.contrib.metrics.streaming_pearson_correlation(Yhat, Y, name="pcc")

        # ------ train parameters -----
        global_step = tf.Variable(0, trainable=False, name="global_step")  # like a counter for minimize() function
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   decay_steps=self.decay_at_step, decay_rate=0.96, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(rmse, global_step=global_step)

        # ------ start training ------
        with tf.Session() as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)

            for i in range(self.max_iter):
                nn_updates["learning_rate"].append(learning_rate.eval())  # to be used with Adam opt.
                sess.run(train_step, feed_dict={X: self.trainX, Y: self.trainY, pkeep: self.pkeep_train})

                if (i % 30 == 0):
                    l, p = sess.run([loss, pcc], feed_dict={X: self.trainX, Y: self.trainY, pkeep: 1})
                    nn_updates["loss_train"].append(l)
                    nn_updates["pcc_train"].append(p[0])

                    l, p = sess.run([loss, pcc], feed_dict={X: self.valX, Y: self.valY, pkeep: 1})
                    nn_updates["loss_val"].append(l)
                    nn_updates["pcc_val"].append(p[0])

            ########## Now predict the performance, and update the output dict ########

            y, r, l, p = sess.run([Yhat, rmse, loss, pcc], feed_dict={X: self.valX, Y: self.valY, pkeep: 1})
            nn_updates["yhat_val"] = y
            nn_updates["rmse_val"] = r
            nn_updates["loss_val"].append(l)
            nn_updates["pcc_val"].append(p[0])

            nn_updates["loss"] = nn_updates["rmse_val"]  # 1 - nn_updates["pcc_val"][-1]    # loss_val[-1]  # previously, 1-pcc_val or loss_val
            if (np.any(np.isnan(nn_updates["loss"]))):
                nn_updates["loss"] = np.inf  # https://github.com/hyperopt/hyperopt/pull/176
                nn_updates["status"] = STATUS_FAIL
            else:
                nn_updates["status"] = STATUS_OK

            nn_updates["W1"] = W1.eval()
            nn_updates["W2"] = W2.eval()
            nn_updates["b1"] = b1.eval()
            nn_updates["g1"] = g1.eval()
            if not (wts[3] is None):
                nn_updates["W3"] = W3.eval()
                nn_updates["b2"] = b2.eval()
                nn_updates["g2"] = g2.eval()

            y, r, l, p = sess.run([Yhat, rmse, loss, pcc], feed_dict={X: self.trainX, Y: self.trainY, pkeep: 1})
            nn_updates["yhat_train"] = y
            nn_updates["rmse_train"] = r
            nn_updates["loss_train"].append(l)
            nn_updates["pcc_train"].append(p[0])

            y, r, l, p = sess.run([Yhat, rmse, loss, pcc], feed_dict={X: self.testX, Y: self.testY, pkeep: 1})
            nn_updates["yhat_test"] = y
            nn_updates["rmse_test"] = r
            nn_updates["loss_test"] = l  # saving for the first time
            nn_updates["pcc_test"] = p[0]

            nn_updates["learning_rate"].append(learning_rate.eval())  # will be used in retraining

            print("lamda:{}, layer_sizes:{}, lr:{}, rmse_val:{:.3f}, \
                rmse_test:{:.3f}, use_sigmoid:[{},{},{}], status:{}".format(
                lamda, layer_sizes, starter_learning_rate,
                nn_updates["rmse_val"], nn_updates["rmse_test"],
                use_sigmoid_h1, use_sigmoid_h2, use_sigmoid_yhat,
                nn_updates["status"]))

        return nn_updates

    def tf_model(self, X, Y, W1, W2, W3, b1, g1, b2, g2, pkeep, lamda, use_sigmoid_h1, use_sigmoid_h2, use_sigmoid_yhat):

        # -------- the core model ---------
        H1 = tf.matmul(X, W1, name="h1")
        H1_mean, H1_var = tf.nn.moments(H1, axes=[0], keep_dims=True, name="h1_moments")
        H1_bn = tf.nn.batch_normalization(H1, H1_mean, H1_var, offset=b1, scale=g1,
                                          variance_epsilon=0.000001, name="h1_bn")  # perform batch normalization
        if (use_sigmoid_h1):
            H1_bnt = tf.nn.sigmoid(H1_bn, name="h1_bn_sigmoid")
        else:
            H1_bnt = tf.nn.relu(H1_bn, name="h1_bn_relu")
        H1_bnd = tf.nn.dropout(H1_bnt, pkeep, name="h1_after_bn_transformation_and_dropout")

        if not (W3 is None):
            H2 = tf.matmul(H1_bnd, W2, name="h2")
            H2_mean, H2_var = tf.nn.moments(H2, axes=[0], keep_dims=True, name="h2_moments")
            H2_bn = tf.nn.batch_normalization(H2, H2_mean, H2_var, offset=b2, scale=g2,
                                              variance_epsilon=0.000001, name="h2_bn")
            if (use_sigmoid_h2):
                H2_bnt = tf.nn.sigmoid(H2_bn, name="h2_bn_sigmoid")
            else:
                H2_bnt = tf.nn.sigmoid(H2_bn, name="h2_bn_relu")
            H2_bnd = tf.nn.dropout(H2_bnt, pkeep, name="h2_after_bns_transformation_and_dropout")
            if (use_sigmoid_yhat):
                Yhat = tf.nn.sigmoid(tf.matmul(H2_bnd, W3), name="Yhat_sigmoid_W3_present")
            else:
                Yhat = tf.nn.sigmoid(tf.matmul(H2_bnd, W3), name="Yhat_relu_W3_present")
            regularizer = tf.add(tf.add(tf.nn.l2_loss(W1), tf.nn.l2_loss(W2)), tf.nn.l2_loss(W3), name="regularizer")
        else:  # no W3, W2.shape[1]==1
            if (use_sigmoid_yhat):
                Yhat = tf.nn.sigmoid(tf.matmul(H1_bnd, W2), name="Yhat_sigmoid_noW3")
            else:
                Yhat = tf.nn.relu(tf.matmul(H1_bnd, W2), name="Yhat_relu_noW3")
            regularizer = tf.add(tf.nn.l2_loss(W1), tf.nn.l2_loss(W2), name="regularizer")

        rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(Yhat, Y)), name="loss")
        loss = rmse + lamda * regularizer

        return Yhat, rmse, loss

    def get_random_wts(self, layer_sizes, use_h2=False):
        '''Parameters:

            layer_sizes:    This is either a 1-d or 2-d array.
        '''
        i = self.trainX.shape[1]
        o = 1

        wts = {}
        if (use_h2 is False):
            h1 = layer_sizes[0]
            wts[1] = np.random.randn(i, h1)
            wts[2] = np.random.randn(h1, o)
            wts[3] = None
        else:
            h1, h2 = layer_sizes
            wts[1] = np.random.randn(i, h1)
            wts[2] = np.random.randn(h1, h2)
            wts[3] = np.random.randn(h2, o)
        return wts

    def get_percentage_error(self, real_yhat, predicted_yhat):

        pes = []  # pes = percentage errors
        y_minus_yhat = abs(real_yhat.flatten() - predicted_yhat.flatten())
        for a, b in zip(y_minus_yhat, real_yhat.flatten()):
            if (b == 0):
                b = 0.005
            pes.append(a / b)
        return pes

    def get_log_into_to_save(self, index, trials, best_params, mode):
        '''Parameters:

            index : Represents the index for the best parameter set
                    in the trials.results object.
        '''

        yhat_train = trials.results[index]["yhat_train"].flatten()
        yhat_val = trials.results[index]["yhat_val"].flatten()
        yhat_test = trials.results[index]["yhat_test"].flatten()

        rmse_train = trials.results[index]["rmse_train"]  # rmse's are saved as scalars (in lists)
        rmse_val = trials.results[index]["rmse_val"]
        rmse_test = trials.results[index]["rmse_test"]

        med_pc_train_error = np.median(self.get_percentage_error(self.trainY.flatten(), yhat_train))
        med_pc_val_error = np.median(self.get_percentage_error(self.valY.flatten(), yhat_val))
        pc_test_errors = self.get_percentage_error(self.testY.flatten(), yhat_test)
        med_pc_test_error = np.median(pc_test_errors)

        # get train, val, test pccs
        med_pcc_train = trials.results[index]["pcc_train"][-1]
        med_pcc_val = trials.results[index]["pcc_val"][-1]
        if (len(pc_test_errors) > 2):
            try:
                med_pcc_test = np.corrcoef(self.testY.flatten(), yhat_test)[0, 1]
            except FloatingPointError:
                med_pcc_test = np.nan
        else:
            med_pcc_test = np.nan

        # get train, val and test sccs
        try:
            med_train_scc = stats.spearmanr(self.trainY.flatten(), yhat_train)[0]
            med_val_scc = stats.spearmanr(self.valY.flatten(), yhat_val)[0]
        except:
            med_train_scc = np.nan
            med_val_scc = np.nan
        if (len(pc_test_errors) > 2):  # as with med_pcc_test above
            try:
                med_test_scc = stats.spearmanr(self.testY.flatten(), yhat_test)[0]
            except FloatingPointError:
                med_test_scc = np.nan
        else:
            med_test_scc = np.nan

        to_log = "mode:{};test_group_{}:{};testX.shape:{};median_pc_error:{:.3f},{:.3f},{:.3f};rmse:{:.3f},{:.3f},{:.3f};PCC:{:.3f},{:.3f},{:.3f};SCC:{:.3f},{:.3f},{:.3f};best_params:{};test_pc_errors:{}".format(
            mode, self.test_eid_group_index, self.test_eid_group, self.testX.shape,
            med_pc_train_error, med_pc_val_error, med_pc_test_error,  # median pc errors
            rmse_train, rmse_val, rmse_test,  # rmse measures
            med_pcc_train, med_pcc_val, med_pcc_test,  # all PCCs
            med_train_scc, med_val_scc, med_test_scc,  # all SCCs
            re.sub("\s+", "", str(best_params)),
            ",".join([str(x) for x in pc_test_errors]))

        return to_log

    def plot_scatter_performance(self, mode, index, trials, gv, plot_title):
        '''This function will be called from main.py.

        Parameters:

        plot_title: This is the prefix from the logger performance
                    output line called and saved in main.py. Hence,
                    this already has the mode information.
        index:      Index for trial with the least error
        '''
        plt.figure(figsize=(5, 5))
        sns.regplot(self.trainY.flatten(), trials.results[index]["yhat_train"].flatten(),
                    robust=False, fit_reg=False, scatter_kws={'alpha': 0.45}, color="salmon")
        sns.regplot(self.valY.flatten(), trials.results[index]["yhat_val"].flatten(),
                    robust=False, fit_reg=False, color="mediumvioletred")
        sns.regplot(self.testY.flatten(), trials.results[index]["yhat_test"].flatten(),
                    robust=False, fit_reg=True if len(self.testY.flatten()) > 2 else False, color="steelblue")

        plt.xlim((0, 1.1))
        plt.ylim((0, 1.1))
        plt.plot([[0, 0], [1, 1]], "--")
        plt.ylabel("Predicted RPKM signal")
        plt.title(plot_title)

        xlabel_suffix = "using"
        if (gv.use_random_DHSs):
            xlabel_suffix += " random DHSs"
        else:
            xlabel_suffix += " real DHSs"
        if (gv.use_random_TFs):
            xlabel_suffix += " and random TFs"
        else:
            xlabel_suffix += " and real TFs"
        plt.xlabel("Real RPKM signal normalized - after init training\n{}".format(xlabel_suffix))

        fig_name = "{}_perf_on_{}_after_initTraining_mode_{}".format(gv.gene_ofInterest, self.test_eid_group, mode)
        if (gv.use_random_DHSs is False) and (gv.use_random_TFs is False):
            fig_name += "_realFts.pdf"
        else:
            fig_name += ".pdf"

        plt.savefig(os.path.join(gv.outputDir, fig_name))
        plt.close()

    def get_params_from_best_trial(self, index, trials, best_params):
        '''Parameters:

            index: Index in trials.results for the best params
        '''

        trainX = np.concatenate((self.trainX, self.valX), axis=0)
        trainY = np.concatenate((self.trainY, self.valY), axis=0)

        if (best_params["layers"] == 1):
            layer_sizes = [best_params["n_units_layer_21"], best_params["n_units_layer_22"]]
        else:
            layer_sizes = [best_params["n_units_layer_11"]]

        lamda = LAMDA_BASE * 10**(-1 * best_params['lamda'])  # should match call in get_parameter_space_forHPO()
        use_sigmoid_h1 = best_params["use_sigmoid_h1"]
        use_sigmoid_h2 = best_params["use_sigmoid_h2"]
        use_sigmoid_yhat = best_params["use_sigmoid_yhat"]
        starter_lr = trials.results[index]["learning_rate"][-1]  # LR_BASE * 10**(-1 * best_params['starter_learning_rate'])

        wts = {}
        wts[1] = trials.results[index]["W1"]
        wts[2] = trials.results[index]["W2"]
        b1 = trials.results[index]["b1"]
        g1 = trials.results[index]["g1"]
        if ("W3" in trials.results[index].keys()):  # if W3 is None initially, W3 is not saved later.
            wts[3] = trials.results[index]["W3"]
            b2 = trials.results[index]["b2"]
            g2 = trials.results[index]["g2"]
        else:
            wts[3] = None
            b2 = None
            g2 = None

        print("best params: lamda:{}, starter_lr:{}, layer_sizes:{}, use_sigmoid:[{},{},{}]".format(lamda, starter_lr, layer_sizes,
                                                                                                    use_sigmoid_h1, use_sigmoid_h2, use_sigmoid_yhat))
        print("Also, rmses:[{:.3f},{:.3f},{:.3f}]".format(trials.results[index]["rmse_train"],
                                                          trials.results[index]["rmse_val"],
                                                          trials.results[index]["rmse_test"]))
        return starter_lr, use_sigmoid_h1, use_sigmoid_h2, use_sigmoid_yhat, wts, layer_sizes, lamda, trainX, trainY, b1, g1, b2, g2

    def retrain_tensorflow_nn(self, starter_learning_rate, use_sigmoid_h1, use_sigmoid_h2, use_sigmoid_yhat, wts, layer_sizes, lamda, trainX, trainY, b1, g1, b2, g2):

        print("starting learning rate for retrain:{:.3f}".format(starter_learning_rate))
        updates = {}
        updates["loss_train"] = []  # loss = rmse + regularization
        updates["loss_test"] = []
        updates["rmse_train"] = []
        updates["rmse_test"] = []
        updates["pcc_train"] = []
        updates["pcc_test"] = []
        updates["yhat_train"] = []
        updates["yhat_test"] = []
        updates["learning_rate"] = []

        # ------ Variables and placeholders ------
        X = tf.placeholder(tf.float32, shape=[None, wts[1].shape[0]], name="X")
        Y = tf.placeholder(tf.float32, [None, 1], name="Y")  # true Ys
        pkeep = tf.placeholder(tf.float32, name="pkeep")

        W1 = tf.Variable(tf.cast(wts[1], tf.float32), name="W1")
        b1 = tf.Variable(tf.cast(b1, tf.float32), name="H1_bn_offset")  # bn == batch normalization
        g1 = tf.Variable(tf.cast(g1, tf.float32), name="H1_bn_scale")
        W2 = tf.Variable(tf.cast(wts[2], tf.float32), name="W2")

        if not (wts[3] is None):
            W3 = tf.Variable(tf.cast(wts[3], tf.float32), name="W3")
            b2 = tf.Variable(tf.cast(b2, tf.float32), name="H2_bn_offset")
            g2 = tf.Variable(tf.cast(g2, tf.float32), name="H2_bn_scale")
        else:
            W3 = None
            b2 = None
            g2 = None

        Yhat, rmse, loss = self.tf_model(X, Y, W1, W2, W3, b1, g1, b2, g2, pkeep, lamda,
                                         use_sigmoid_h1, use_sigmoid_h2, use_sigmoid_yhat)
        pcc = tf.contrib.metrics.streaming_pearson_correlation(Yhat, Y, name="pcc")

        # ------ train parameters -----
        global_step = tf.Variable(0, trainable=False, name="global_step")  # like a counter for minimize() function
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   decay_steps=self.decay_at_step, decay_rate=0.96, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(rmse, global_step=global_step)

        # ------ start training ------
        with tf.Session() as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)

            for i in range(self.max_iter):
                updates["learning_rate"].append(learning_rate.eval())  # to be used with Adam opt.

                if (i % 1 == 0):
                    y, r, l, p = sess.run([Yhat, rmse, loss, pcc], feed_dict={X: trainX, Y: trainY, pkeep: 1})
                    updates["yhat_train"].append(y)
                    updates["rmse_train"].append(r)
                    updates["loss_train"].append(l)
                    updates["pcc_train"].append(p[0])
                    y, r, l, p = sess.run([Yhat, rmse, loss, pcc], feed_dict={X: self.testX, Y: self.testY, pkeep: 1})
                    if (i == 0):
                        print("initial test rmse is {:.3f} before retraining".format(r))
                    updates["yhat_test"].append(y)
                    updates["rmse_test"].append(r)
                    updates["loss_test"].append(l)
                    updates["pcc_test"].append(p[0])

                sess.run(train_step, feed_dict={X: trainX, Y: trainY, pkeep: self.pkeep_train})

            ########## Now predict the performance, and update the output dict ########

            updates["W1"] = W1.eval()
            updates["W2"] = W2.eval()
            updates["b1"] = b1.eval()
            updates["g1"] = g1.eval()
            if not (wts[3] is None):
                updates["W3"] = W3.eval()
                updates["b2"] = b2.eval()
                updates["g2"] = g2.eval()
            else:
                updates["W3"] = None
                updates["b2"] = None
                updates["g2"] = None

            y, r, l, p = sess.run([Yhat, rmse, loss, pcc], feed_dict={X: trainX, Y: trainY, pkeep: 1})
            updates["yhat_train"].append(y)  # saving for the first time
            updates["rmse_train"].append(r)
            updates["loss_train"].append(l)
            updates["pcc_train"].append(p[0])

            y, r, l, p = sess.run([Yhat, rmse, loss, pcc], feed_dict={X: self.testX, Y: self.testY, pkeep: 1})
            updates["yhat_test"].append(y)
            updates["rmse_test"].append(r)
            updates["loss_test"].append(l)
            updates["pcc_test"].append(p[0])

        return updates

    def get_log_info_to_save_after_retraining(self, trainY, updates, title_prefix=""):
        '''Parameters:

            trainY:         trainY is updated after retraining, and has valY info as well.
                            Hence, this is different from self.trainY. (testY remains
                            the same and therefore need not be passed in as parameter.)
            title_prefix:   The title prefix will look like this:
                            "mode:joint;test_group_7:Epithelial;testX.shape:(8, 47)".
                            This is obtained from "to_log" line that is sent to main.logger.

        '''
        pc_train_errors = self.get_percentage_error(
            real_yhat=trainY, predicted_yhat=updates["yhat_train"][-1])
        pc_test_errors = self.get_percentage_error(
            real_yhat=self.testY, predicted_yhat=updates["yhat_test"][-1])

        title_line1 = title_prefix
        title_line2 = "median_pc_error:{:.3f},{:.3f};rmse:{:.3f},{:.3f}".format(
            np.median(pc_train_errors), np.median(pc_test_errors),
            updates["rmse_train"][-1], updates["rmse_test"][-1])
        title_line3 = "PCC:{:.3f},{:.3f}".format(updates["pcc_train"][-1], updates["pcc_test"][-1])

        pc_test_errors = ",".join([str(x) for x in pc_test_errors])
        to_log = "{};{};{};test_pc_errors:{}".format(
            title_line1, title_line2, title_line3, pc_test_errors)
        plot_title = "{}\n{}\n{}".format(title_line1, title_line2, title_line3)

        return to_log, plot_title

    def plot_performance_after_retraining(self, mode, gv, updates, trainY, plot_title):
        '''This function will be called from main().
        Parameters:

            mode:       one of "dhss", "tfs" or "joint"

            trainY:     Note the trainX and trainY are updated during retraining
                        but not the testX/Y.
        '''
        plt.figure(figsize=(5, 5))
        sns.regplot(trainY.flatten(), updates["yhat_train"][-1].flatten(),
                    robust=False, fit_reg=False, scatter_kws={'alpha': 0.45}, color="salmon")
        sns.regplot(self.testY.flatten(), updates["yhat_test"][-1].flatten(),
                    robust=False, fit_reg=True if len(self.testY.flatten()) > 2 else False, color="steelblue")
        plt.xlim((0, 1.1))
        plt.ylim((0, 1.1))
        plt.plot([[0, 0], [1, 1]], "--")
        plt.ylabel("Predicted RPKM signal")
        plt.title(plot_title)

        xlabel_suffix = "using"
        if (gv.use_random_DHSs):
            xlabel_suffix += " random DHSs"
        else:
            xlabel_suffix += " real DHSs"
        if (gv.use_random_TFs):
            xlabel_suffix += " and random TFs"
        else:
            xlabel_suffix += " and real TFs"
        plt.xlabel("Real RPKM signal normalized - after retraining\n{}".format(xlabel_suffix))

        fig_name = "{}_perf_on_{}_after_retraining_mode_{}".format(gv.gene_ofInterest, self.test_eid_group, mode)
        if (gv.use_random_DHSs is False) and (gv.use_random_TFs is False):
            fig_name += "_realFts.pdf"
        else:
            fig_name += ".pdf"

        plt.savefig(os.path.join(gv.outputDir, fig_name))
        plt.close()

    def plot_rmse(self, gv, updates, mode):
        plt.figure(figsize=(5, 5))
        plt.plot(updates["rmse_train"], color="salmon", label="rmse_train")
        plt.plot(updates["rmse_test"], color="mediumvioletred", label="rmse_test")
        plt.xlabel("every n iterations")
        plt.ylabel("rmse")
        plt.legend()

        fig_name = "{}_rmse_on_{}_after_retraining_mode_{}".format(gv.gene_ofInterest, self.test_eid_group, mode)
        if (gv.use_random_DHSs is False) and (gv.use_random_TFs is False):
            fig_name += "_realFts.pdf"
        else:
            fig_name += ".pdf"
        plt.savefig(os.path.join(gv.outputDir, fig_name))
        plt.close()

    def plot_train_test_XY(self, gv):
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        sns.heatmap(self.trainX, vmin=0, vmax=1, yticklabels=False)
        plt.title('trainX')
        plt.subplot(2, 2, 2)
        sns.heatmap(self.trainY, vmin=0, vmax=1, yticklabels=False)
        plt.title('trainY')

        plt.subplot(2, 2, 3)
        sns.heatmap(self.testX, vmin=0, vmax=1, yticklabels=False)
        plt.title('testX')

        plt.subplot(2, 2, 4)
        sns.heatmap(self.testY, vmin=0, vmax=1, yticklabels=False)
        plt.title('testY')
        plt.suptitle("{}, test_grp:{}".format(gv.gene_ofInterest, self.test_eid_group))

        plt.savefig(os.path.join(gv.outputDir, "train_test_XY_for_testgrp_{}.pdf".format(self.test_eid_group)))
        plt.close()

    def get_rmse(self, updates, use_sigmoid_h1, use_sigmoid_h2, use_sigmoid_yhat,
                 layer_sizes, lamda, X_, Y_ori):
        # X_ can be original or imputed X. Y is always fixed to Y_ori.

        # ------ Variables and placeholders ------
        pkeep = tf.placeholder(tf.float32, name="pkeep")
        X = tf.placeholder(tf.float32, shape=[None, updates["W1"].shape[0]], name="X")
        Y = tf.placeholder(tf.float32, [None, 1], name="Y")  # true Ys

        W1 = tf.Variable(tf.cast(updates["W1"], tf.float32), name="W1")
        b1 = tf.Variable(tf.cast(updates["b1"], tf.float32), name="H1_bn_offset")  # bn == batch normalization
        g1 = tf.Variable(tf.cast(updates["g1"], tf.float32), name="H1_bn_scale")
        W2 = tf.Variable(tf.cast(updates["W2"], tf.float32), name="W2")

        if not (updates["W3"] is None):
            W3 = tf.Variable(tf.cast(updates["W3"], tf.float32), name="W3")
            b2 = tf.Variable(tf.cast(updates["b2"], tf.float32), name="H2_bn_offset")
            g2 = tf.Variable(tf.cast(updates["g2"], tf.float32), name="H2_bn_scale")
        else:
            W3 = None
            b2 = None
            g2 = None

        Yhat, rmse, loss = self.tf_model(X, Y, W1, W2, W3, b1, g1, b2, g2, pkeep, lamda,
                                         use_sigmoid_h1, use_sigmoid_h2, use_sigmoid_yhat)

        with tf.Session() as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            r = sess.run([rmse], feed_dict={X: X_, Y: Y_ori, pkeep: 1})
        return r[0]  # just returning the rmse

    def get_yhat(self, updates, use_sigmoid_h1, use_sigmoid_h2, use_sigmoid_yhat,
                 layer_sizes, lamda, X_, Y_ori):
        # X_ can be original or imputed X. Y is always fixed to Y_ori.

        # ------ Variables and placeholders ------
        pkeep = tf.placeholder(tf.float32, name="pkeep")
        X = tf.placeholder(tf.float32, shape=[None, updates["W1"].shape[0]], name="X")
        Y = tf.placeholder(tf.float32, [None, 1], name="Y")  # true Ys

        W1 = tf.Variable(tf.cast(updates["W1"], tf.float32), name="W1")
        b1 = tf.Variable(tf.cast(updates["b1"], tf.float32), name="H1_bn_offset")  # bn == batch normalization
        g1 = tf.Variable(tf.cast(updates["g1"], tf.float32), name="H1_bn_scale")
        W2 = tf.Variable(tf.cast(updates["W2"], tf.float32), name="W2")

        if not (updates["W3"] is None):
            W3 = tf.Variable(tf.cast(updates["W3"], tf.float32), name="W3")
            b2 = tf.Variable(tf.cast(updates["b2"], tf.float32), name="H2_bn_offset")
            g2 = tf.Variable(tf.cast(updates["g2"], tf.float32), name="H2_bn_scale")
        else:
            W3 = None
            b2 = None
            g2 = None

        Yhat, rmse, loss = self.tf_model(X, Y, W1, W2, W3, b1, g1, b2, g2, pkeep, lamda,
                                         use_sigmoid_h1, use_sigmoid_h2, use_sigmoid_yhat)

        with tf.Session() as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            y = sess.run([Yhat], feed_dict={X: X_, Y: Y_ori, pkeep: 1})
        return y  # just returning the yhat


if __name__ == "__main__":
    from global_variables_final_for_git import Global_Vars
    from prep_for_model_for_git import Model_preparation

    class Args(object):
        def __init__(self):
            self.gene = "SIX1"
            self.distance = 150
            self.use_tad_info = True
            self.pcc_lowerlimit_to_filter_dhss = 0.2
            self.filter_tfs_by = "zscore"  # or "pcc"
            self.lowerlimit_to_filter_tfs = 6
            self.take_this_many_top_dhss = 6  # all dhss/tfs will already be filtered by pcc(or zscore)
            self.take_this_many_top_tfs = 10
            self.init_wts_type = "corr"
            self.outputDir = "/Users/Dinesh/Dropbox/Github/predicting_gex_with_nn_git/Output/" + self.gene.upper()
            self.use_random_DHSs = True
            self.use_random_TFs = True
            self.max_iter = 500

    args = Args()
    gv = Global_Vars(args, args.outputDir)  # note this takes in new_output_dir as well in .py scripts
    mp = Model_preparation(gv)
    tm = Tensorflow_model(gv, mp, test_eid_group_index=3)
