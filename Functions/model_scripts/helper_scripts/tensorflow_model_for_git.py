import pdb
import logging
import os
import tensorflow as tf
import numpy as np
import seaborn as sns
from hyperopt import STATUS_OK
import matplotlib.pyplot as plt
plt.switch_backend('agg')


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
        ########## set basic params #########
        self.max_iter = gv.max_iter
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
        print("lamda:{}, layer_sizes:{}".format(lamda, layer_sizes))
        pkeep_train = 0.7
        starter_learning_rate = 0.7
        decay_at_step = 15
        use_sigmoid_h1 = True
        use_sigmoid_h2 = True
        use_sigmoid_yhat = False

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
        global_step = tf.Variable(0, trainable=False, name="global_step")  # this is like a counter when passed to minimize() function
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   decay_steps=decay_at_step, decay_rate=0.96, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # ------ performance metric (besides the loss or rmse) -----
        pcc = tf.contrib.metrics.streaming_pearson_correlation(Yhat, Y, name="pcc")

        # ------ start training ------
        with tf.Session() as sess:
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)

            for i in range(self.max_iter):

                train_data = {X: self.trainX, Y: self.trainY, pkeep: pkeep_train}
                sess.run(train_step, feed_dict=train_data)
                nn_updates["learning_rate"].append(learning_rate.eval())  # to be used with Adam opt.

                if (i % 20 == 0):
                    val_data = {X: self.valX, Y: self.valY, pkeep: 1}
                    nn_updates = self.get_performance_updates(sess, loss, pcc, train_data, val_data, nn_updates)

            ########## Now predict the performance, and update the output dict ########
            nn_updates = self.get_performance_updates(sess, loss, pcc, train_data, val_data, nn_updates)
            nn_updates["yhat_train"] = sess.run(Yhat, feed_dict={X: self.trainX, Y: self.trainY, pkeep: 1})  # model prediction on the training set
            nn_updates["yhat_val"] = sess.run(Yhat, feed_dict={X: self.valX, Y: self.valY, pkeep: 1})
            if (np.isnan(nn_updates["yhat_val"]).any()):
                self.logger.warning("yhat_val has nan..")
                nn_updates["status"] = "fail"
                return nn_updates
            nn_updates["yhat_test"] = sess.run(Yhat, feed_dict={X: self.testX, Y: self.testY, pkeep: 1})
            nn_updates["W1"] = W1.eval()
            nn_updates["W2"] = W2.eval()
            if not (wts[3] is None):
                nn_updates["W3"] = W3.eval()
            nn_updates["loss"] = nn_updates["val_loss"][-1]  # / np.sqrt(nn_updates["val_pcc"][-1] + 0.0001)
            nn_updates["status"] = STATUS_OK
        return nn_updates

    def tf_model(self, X, Y, W1, W2, W3, b1, g1, b2, g2, pkeep, lamda,
                 use_sigmoid_h1=True, use_sigmoid_h2=True, use_sigmoid_yhat=False):

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

    def plot_scatter_performance(self, trials, gv, index=None):
        if (index is None):
            index = np.argmin(trials.losses())  # trial with least error
        plt.figure(figsize=(5, 5))
        sns.regplot(self.trainY.flatten(), trials.results[index]["yhat_train"].flatten(), robust=False, fit_reg=False, scatter_kws={'alpha': 0.45}, color="salmon")
        sns.regplot(self.valY.flatten(), trials.results[index]["yhat_val"].flatten(), robust=False, fit_reg=False, color="mediumvioletred")
        sns.regplot(self.testY.flatten(), trials.results[index]["yhat_test"].flatten(), robust=False, fit_reg=False, color="steelblue")
        plt.xlim((0, 1.1))
        plt.ylim((0, 1.1))
        plt.plot([[0, 0], [1, 1]], "--")
        plt.xlabel("Real RPKM signal normalized")
        plt.ylabel("Predicted RPKM signal")
        med_pc_test_error = np.median(self.get_percentage_error(self.testY, trials.results[index]["yhat_test"].flatten()))
        med_pc_val_error = np.median(self.get_percentage_error(self.valY, trials.results[index]["yhat_val"].flatten()))
        plt.title("{}, Test Group:{},\nMedian (val, test) % error: ({}, {})".format(gv.gene_ofInterest, self.test_eid_group,
                                                                                    round(med_pc_test_error, 2), round(med_pc_val_error, 2)))

        fig_name = "{}_perf_on_{}.pdf".format(gv.gene_ofInterest, self.test_eid_group)
        plt.savefig(os.path.join(gv.outputDir, fig_name))
        return med_pc_test_error, med_pc_val_error


if __name__ == "__main__":
    from global_variables_final_for_git import Global_Vars
    from prep_for_model_for_git import Model_preparation

    class Args(object):
        def __init__(self):
            self.gene = "NANOG"
            self.distance = 200
            self.use_tad_info = True
            self.pcc_lowerlimit_to_filter_dhss = 0.25
            self.take_log2_tpm = True
            self.filter_tfs_by = "zscore"  # or "pcc"
            self.lowerlimit_to_filter_tfs = 4.75
            self.take_this_many_top_fts = 15  # all dhss/tfs will already be filtered by pcc(or zscore)
            self.init_wts_type = "corr"
            self.outputDir = "/Users/Dinesh/Dropbox/Github/predicting_gex_with_nn_git/Output/testing"
            self.use_random_DHSs = False
            self.use_random_TFs = False
            self.max_iter = 300

    args = Args()
    gv = Global_Vars(args, args.outputDir)  # note this takes in new_output_dir as well in .py scripts
    mp = Model_preparation(gv)
    tm = Tensorflow_model(gv, mp, test_eid_group_index=3)
