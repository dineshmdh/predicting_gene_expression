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
from HPO_helper import LAMDA_BASE

np.seterr(all='raise')
plt.switch_backend('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # To ignore TF warning in stdout (source: https://github.com/tensorflow/tensorflow/issues/7778)


class Tensorflow_model(object):
    def __init__(self, gv, mp, test_eid_group_index):

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
        self.pkeep_train = 0.7
        self.starter_learning_rate = 0.03  # set same for both initial training and re-training
        self.decay_at_step = 15
        self.use_sigmoid_h1 = True
        self.use_sigmoid_h2 = True
        self.use_sigmoid_yhat = False
        self.test_eid_group_index = test_eid_group_index

        ########## from model preparation ##########
        self.val_eid_group = "ENCODE2012"  # will be used as validation set; always fixed.
        df_eids, self.train_eid_groups, self.test_eid_group = mp.get_train_and_test_eid_groups(
            mp.csv_eids, self.val_eid_group, test_group_index=self.test_eid_group_index)  # reads the file each time
        self.train_eids = mp.get_eids_for_a_group(df_eids, self.train_eid_groups)
        self.val_eids = mp.get_eids_for_a_group(df_eids, self.val_eid_group)
        self.test_eids = mp.get_eids_for_a_group(df_eids, self.test_eid_group)

        '''Get train and test X matrices for the models.
        Note these X matrices have fts as rows and samples as cols.'''
        self.train_dhss, self.val_dhss, self.test_dhss = mp.get_normalized_train_val_test_dfs(gv.df_dhss, self.train_eids, self.val_eids, self.test_eids)
        self.train_tfs, self.val_tfs, self.test_tfs = mp.get_normalized_train_val_test_dfs(gv.df_tfs, self.train_eids, self.val_eids, self.test_eids)
        self.train_joint = mp.merge_dhs_and_tf_dfs(self.train_dhss, self.train_tfs, gv)  # for dhs+tf joint model
        self.val_joint = mp.merge_dhs_and_tf_dfs(self.val_dhss, self.val_tfs, gv)
        self.test_joint = mp.merge_dhs_and_tf_dfs(self.test_dhss, self.test_tfs, gv)

        '''Now get the train and test Y vectors.
        Note these Y "vectors" are pandas series objects.'''
        self.train_goi, self.val_goi, self.test_goi = mp.get_normalized_train_val_and_test_goi(gv, self.train_eids, self.val_eids, self.test_eids)
        #############################################

        ########## only consider joint model for now ##########
        self.trainX = np.array(self.train_joint.transpose())
        self.valX = np.array(self.val_joint.transpose())
        self.testX = np.array(self.test_joint.transpose())

        self.trainY = np.array(self.train_goi.tolist())
        self.trainY = self.trainY.reshape(self.trainY.shape[0], -1)
        self.valY = np.array(self.val_goi.tolist())
        self.valY = self.valY.reshape(self.valY.shape[0], -1)
        self.testY = np.array(self.test_goi.tolist())
        self.testY = self.testY.reshape(self.testY.shape[0], -1)

        if (np.max(self.trainY) > 1):
            self.logger.error("max(tm.trainY) was > 1..")  # warning only to debug
            pdb.set_trace()

        ############## end of __init__() #############

    def init_nn_updates(self):
        nn_updates = {}
        nn_updates["train_loss"] = []  # loss == cost == rmse
        nn_updates["train_pcc"] = []
        nn_updates["val_loss"] = []
        nn_updates["val_pcc"] = []
        # nn_updates["test_loss"] = []  # not being used; will be of length one for one self/tm model
        # nn_updates["test_pcc"] = []
        nn_updates["learning_rate"] = []
        return nn_updates

    def get_performance_updates(self, sess, loss, pcc, train_data, val_data, nn_updates):
        '''Note: loss == rmse == cost (below)'''
        # success in train?
        l, p = sess.run([loss, pcc], feed_dict=train_data)
        nn_updates["train_loss"].append(l)
        nn_updates["train_pcc"].append(p[0])
        # success in val?
        l_, p_ = sess.run([loss, pcc], feed_dict=val_data)
        nn_updates["val_loss"].append(l_)
        nn_updates["val_pcc"].append(p_[0])

        return nn_updates

    def train_tensorflow_nn(self, parameters):
        '''Arugments:
        - All input arguments with "_" are real arrays to feed in as data.
        - lamda: regularization parameter
        - starter_learning_rate: Initial learning rate for the learning rate decay computation below
        - decay_at_step: learning rate is updated after this many training steps (Easier to think this way)
        '''

        layer_sizes = [int(n) for n in parameters['layers']['n_units_layer']]
        if (len(layer_sizes) == 1):
            wts = self.get_random_wts(layer_sizes, use_h2=False)
        elif (len(layer_sizes) == 2):
            wts = self.get_random_wts(layer_sizes, use_h2=True)
        else:
            raise Exception()
        lamda = parameters['lamda']

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

        Yhat, loss = self.tf_model(X, Y, W1, W2, W3, b1, g1, b2, g2, pkeep, lamda)  # loss == rmse

        # ------ train parameters -----
        global_step = tf.Variable(0, trainable=False, name="global_step")  # like a counter for minimize() function
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step,
                                                   decay_steps=self.decay_at_step, decay_rate=0.96, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # ------ performance metric (besides the loss or rmse) -----
        pcc = tf.contrib.metrics.streaming_pearson_correlation(Yhat, Y, name="pcc")

        # ------ start training ------
        with tf.Session() as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)

            for i in range(self.max_iter):

                train_data = {X: self.trainX, Y: self.trainY, pkeep: self.pkeep_train}
                sess.run(train_step, feed_dict=train_data)
                nn_updates["learning_rate"].append(learning_rate.eval())  # to be used with Adam opt.

                if (i % 10 == 0):
                    val_data = {X: self.valX, Y: self.valY, pkeep: 1}
                    nn_updates = self.get_performance_updates(sess, loss, pcc, train_data, val_data, nn_updates)

            ########## Now predict the performance, and update the output dict ########
            nn_updates["loss"] = nn_updates["val_loss"][-1]  # previously, 1-val_pcc or val_loss
            if (np.any(np.isnan(nn_updates["loss"]))):
                nn_updates["loss"] = np.inf  # https://github.com/hyperopt/hyperopt/pull/176
                nn_updates["status"] = STATUS_FAIL
            else:
                nn_updates["status"] = STATUS_OK

            nn_updates = self.get_performance_updates(sess, loss, pcc, train_data, val_data, nn_updates)
            nn_updates["yhat_train"] = sess.run(Yhat, feed_dict={X: self.trainX, Y: self.trainY, pkeep: 1})
            nn_updates["yhat_val"] = sess.run(Yhat, feed_dict={X: self.valX, Y: self.valY, pkeep: 1})
            nn_updates["yhat_test"] = sess.run(Yhat, feed_dict={X: self.testX, Y: self.testY, pkeep: 1})
            nn_updates["W1"] = W1.eval()
            nn_updates["W2"] = W2.eval()
            nn_updates["b1"] = b1.eval()
            nn_updates["g1"] = g1.eval()
            if not (wts[3] is None):
                nn_updates["W3"] = W3.eval()
                nn_updates["b2"] = b2.eval()
                nn_updates["g2"] = g2.eval()

            print("lamda:{}, layer_sizes:{}, loss:{}, status:{}, yhat_test:{}".format(
                lamda, layer_sizes, nn_updates["loss"], nn_updates["status"], nn_updates["yhat_test"].flatten()))

        return nn_updates

    def tf_model(self, X, Y, W1, W2, W3, b1, g1, b2, g2, pkeep, lamda):

        # -------- the core model ---------
        H1 = tf.matmul(X, W1, name="h1")
        H1_mean, H1_var = tf.nn.moments(H1, axes=[0], keep_dims=True, name="h1_moments")
        H1_bn = tf.nn.batch_normalization(H1, H1_mean, H1_var, offset=b1, scale=g1,
                                          variance_epsilon=0.000001, name="h1_bn")  # perform batch normalization
        if (self.use_sigmoid_h1):
            H1_bnt = tf.nn.sigmoid(H1_bn, name="h1_bn_sigmoid")
        else:
            H1_bnt = tf.nn.relu(H1_bn, name="h1_bn_relu")
        H1_bnd = tf.nn.dropout(H1_bnt, pkeep, name="h1_after_bn_transformation_and_dropout")

        if not (W3 is None):
            H2 = tf.matmul(H1_bnd, W2, name="h2")
            H2_mean, H2_var = tf.nn.moments(H2, axes=[0], keep_dims=True, name="h2_moments")
            H2_bn = tf.nn.batch_normalization(H2, H2_mean, H2_var, offset=b2, scale=g2,
                                              variance_epsilon=0.000001, name="h2_bn")
            if (self.use_sigmoid_h2):
                H2_bnt = tf.nn.sigmoid(H2_bn, name="h2_bn_sigmoid")
            else:
                H2_bnt = tf.nn.sigmoid(H2_bn, name="h2_bn_relu")
            H2_bnd = tf.nn.dropout(H2_bnt, pkeep, name="h2_after_bns_transformation_and_dropout")
            if (self.use_sigmoid_yhat):
                Yhat = tf.nn.sigmoid(tf.matmul(H2_bnd, W3), name="Yhat_sigmoid_W3_present")
            else:
                Yhat = tf.nn.sigmoid(tf.matmul(H2_bnd, W3), name="Yhat_relu_W3_present")
            regularizer = tf.add(tf.add(tf.nn.l2_loss(W1), tf.nn.l2_loss(W2)), tf.nn.l2_loss(W3), name="regularizer")
        else:  # no W3, W2.shape[1]==1
            if (self.use_sigmoid_yhat):
                Yhat = tf.nn.sigmoid(tf.matmul(H1_bnd, W2), name="Yhat_sigmoid_noW3")
            else:
                Yhat = tf.nn.relu(tf.matmul(H1_bnd, W2), name="Yhat_relu_noW3")
            regularizer = tf.add(tf.nn.l2_loss(W1), tf.nn.l2_loss(W2), name="regularizer")

        loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(Yhat, Y)), name="loss")  # cost function to minimize
        loss = loss + lamda * regularizer

        return Yhat, loss

    def get_random_wts(self, layer_sizes, use_h2=False):
        # layer_sizes is either a 1-d or 2-d array
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
            pes.append(a / b)
        return pes

    def get_log_into_to_save(self, trials, best_params):
        index = np.argmin(trials.losses())
        print("Best param index", index)

        yhat_train = trials.results[index]["yhat_train"].flatten()
        yhat_val = trials.results[index]["yhat_val"].flatten()
        yhat_test = trials.results[index]["yhat_test"].flatten()

        med_pc_train_error = np.median(self.get_percentage_error(self.trainY.flatten(), yhat_train))
        med_pc_val_error = np.median(self.get_percentage_error(self.valY.flatten(), yhat_val))
        pc_test_error = self.get_percentage_error(self.testY.flatten(), yhat_test)
        med_pc_test_error = np.median(pc_test_error)

        # get train, val, test pccs
        med_train_pcc = trials.results[index]["train_pcc"][-1]
        med_val_pcc = trials.results[index]["val_pcc"][-1]
        if (len(pc_test_error) > 2):
            try:
                med_test_pcc = np.corrcoef(self.testY.flatten(), yhat_test)[0, 1]
            except FloatingPointError:
                med_test_pcc = np.nan
        else:
            med_test_pcc = np.nan

        # get train, val and test sccs
        try:
            med_train_scc = stats.spearmanr(self.trainY.flatten(), yhat_train)[0]
            med_val_scc = stats.spearmanr(self.valY.flatten(), yhat_val)[0]
        except:
            med_train_scc = np.nan
            med_val_scc = np.nan
        if (len(pc_test_error) > 2):
            try:
                med_test_scc = stats.spearmanr(self.testY.flatten(), yhat_test)[0]
            except FloatingPointError:
                med_test_scc = np.nan

        pc_test_error = ",".join([str(x) for x in pc_test_error])
        to_log = "test_group_{}:{};testX.shape:{};median_pc_error:{:.3f},{:.3f},{:.3f};PCC:{:.3f},{:.3f},{:.3f};SCC:{:.3f},{:.3f},{:.3f};best_params:{};test_pc_errors:{}".format(
            self.test_eid_group_index, self.test_eid_group, self.testX.shape,
            med_pc_train_error, med_pc_val_error, med_pc_test_error,  # median pc errors
            med_train_pcc, med_val_pcc, med_test_pcc,  # all PCCs
            med_train_scc, med_val_scc, med_test_scc,  # all SCCs
            re.sub("\s+", "", str(best_params)), pc_test_error
        )
        return to_log

    def plot_scatter_performance(self, trials, gv, plot_title):
        '''This function will be called from main.py'''
        index = np.argmin(trials.losses())  # trial with least error
        plt.figure(figsize=(5, 5))
        sns.regplot(self.trainY.flatten(), trials.results[index]["yhat_train"].flatten(), robust=False, fit_reg=False, scatter_kws={'alpha': 0.45}, color="salmon")
        sns.regplot(self.valY.flatten(), trials.results[index]["yhat_val"].flatten(), robust=False, fit_reg=False, color="mediumvioletred")
        sns.regplot(self.testY.flatten(), trials.results[index]["yhat_test"].flatten(), robust=False, fit_reg=False, color="steelblue")
        plt.xlim((0, 1.1))
        plt.ylim((0, 1.1))
        plt.plot([[0, 0], [1, 1]], "--")

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

        plt.ylabel("Predicted RPKM signal")
        plt.title(plot_title)
        fig_name = "{}_perf_on_{}_after_init_training.pdf".format(gv.gene_ofInterest, self.test_eid_group)
        plt.savefig(os.path.join(gv.outputDir, fig_name))
        plt.close()

    def get_params_from_best_trial(self, index, trials, best_params):
        '''trials and best_params are obtained from HPO search.
        index is the index in trials.results for the best params'''

        nn_updates = {}
        nn_updates["train_loss"] = []  # loss == cost == rmse
        nn_updates["train_pcc"] = []
        nn_updates["test_loss"] = []
        nn_updates["test_pcc"] = []
        nn_updates["learning_rate"] = []

        trainX = np.concatenate((self.trainX, self.valX), axis=0)
        trainY = np.concatenate((self.trainY, self.valY), axis=0)

        if (best_params["layers"] == 1):
            layer_sizes = [best_params["n_units_layer_21"], best_params["n_units_layer_22"]]
        else:
            layer_sizes = [best_params["n_units_layer_11"]]

        lamda = LAMDA_BASE * 10**(-1 * best_params['lamda'])  # should match call in get_parameter_space_forHPO()

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

        return wts, layer_sizes, lamda, trainX, trainY, nn_updates, b1, g1, b2, g2

    def retrain_tensorflow_nn(self, wts, layer_sizes, lamda, trainX, trainY, nn_updates, b1, g1, b2, g2):

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

        Yhat, loss = self.tf_model(X, Y, W1, W2, W3, b1, g1, b2, g2, pkeep, lamda)  # loss == rmse
        pcc = tf.contrib.metrics.streaming_pearson_correlation(Yhat, Y, name="pcc")

        # ------ train parameters -----
        global_step = tf.Variable(0, trainable=False, name="global_step")  # like a counter for minimize() function
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step,
                                                   decay_steps=self.decay_at_step, decay_rate=0.96, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # ------ start training ------
        with tf.Session() as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            print("done initializing the session")

            train_data = {X: trainX, Y: trainY, pkeep: self.pkeep_train}
            test_data = {X: self.testX, Y: self.testY, pkeep: 1}
            print("done setting up train and test data")

            for i in range(self.max_iter):
                sess.run(train_step, feed_dict=train_data)
                nn_updates["learning_rate"].append(learning_rate.eval())  # to be used with Adam opt.
                if (i % 10 == 0):
                    l, p = sess.run([loss, pcc], feed_dict=train_data)
                    nn_updates["train_loss"].append(l)
                    nn_updates["train_pcc"].append(p[0])
                    l_, p_ = sess.run([loss, pcc], feed_dict=test_data)
                    nn_updates["test_loss"].append(l_)
                    nn_updates["test_pcc"].append(p_[0])

            ########## Now predict the performance, and update the output dict ########
            nn_updates["yhat_train"] = sess.run(Yhat, feed_dict={X: trainX, Y: trainY, pkeep: 1})
            nn_updates["yhat_test"] = sess.run(Yhat, feed_dict={X: self.testX, Y: self.testY, pkeep: 1})
            nn_updates["W1"] = W1.eval()
            nn_updates["W2"] = W2.eval()
            if not (wts[3] is None):
                nn_updates["W3"] = W3.eval()

        return nn_updates

    def plot_performance_after_retraining(self, gv, updates, trainY, title_prefix=""):
        '''This function will be called from main().
        Note the trainX and trainY are updated during retraining but not the testX/Y.
        The title prefix will look like this: "test_group_7:Epithelial;testX.shape:(8, 47)",
        and will be obtained from "to_log" line that is sent to main.logger.'''
        plt.figure(figsize=(5, 5))
        sns.regplot(trainY.flatten(), updates["yhat_train"].flatten(), robust=False, fit_reg=False, scatter_kws={'alpha': 0.45}, color="salmon")
        sns.regplot(self.testY.flatten(), updates["yhat_test"].flatten(), robust=False, fit_reg=False, color="steelblue")
        plt.xlim((0, 1.1))
        plt.ylim((0, 1.1))
        plt.plot([[0, 0], [1, 1]], "--")
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
        plt.ylabel("Predicted RPKM signal")
        pc_test_errors = self.get_percentage_error(real_yhat=self.testY, predicted_yhat=updates["yhat_test"].flatten())
        plt.title("{}\nmed_test_pc_error:{:.3f};test_pcc:{:.3f}".format(title_prefix, np.median(pc_test_errors), updates["test_pcc"][-1]))
        fig_name = "{}_perf_on_{}_after_retraining.pdf".format(gv.gene_ofInterest, self.test_eid_group)
        plt.savefig(os.path.join(fig_name))
        plt.close()


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
            self.take_this_many_top_dhss = 4  # all dhss/tfs will already be filtered by pcc(or zscore)
            self.take_this_many_top_tfs = 6
            self.init_wts_type = "corr"
            self.outputDir = "/Users/Dinesh/Dropbox/Github/predicting_gex_with_nn_git/Output/" + self.gene.upper()
            self.use_random_DHSs = True
            self.use_random_TFs = True
            self.max_iter = 500

    args = Args()
    gv = Global_Vars(args, args.outputDir)  # note this takes in new_output_dir as well in .py scripts
    mp = Model_preparation(gv)
    tm = Tensorflow_model(gv, mp, test_eid_group_index=3)
