'''
Created on August 23, 2017

For some functions below, input parameter m is the "Model_preparation" object.

'''
import pdb
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import tensorflow as tf
import copy

np.seterr(divide='ignore', invalid='ignore')  # see README.md


def get_batches(m):
    # return list indices for batch1 and batch2 to load trainX and trainY in batches
    b1 = np.random.choice(len(m.train_celllines), int(len(m.train_celllines) / 2))
    b2 = np.array(list(set(np.arange(len(m.train_celllines))).difference(set(b1))))
    return b1, b2


def init_updates():
    updates = {}
    updates["train_loss"] = []  # loss == cost == rmse
    updates["train_pcc"] = []
    updates["test_loss"] = []
    updates["test_pcc"] = []
    updates["learning_rate"] = []
    return updates


def get_performance_updates(sess, loss, pcc, train_data, test_data, updates):
    '''loss == rmse == cost (below)'''
    # success in train?
    l, p = sess.run([loss, pcc], feed_dict=train_data)
    updates["train_loss"].append(l)
    updates["train_pcc"].append(p)

    # success in test?
    l_, p_ = sess.run([loss, pcc], feed_dict=test_data)
    updates["test_loss"].append(l_)
    updates["test_pcc"].append(p_)

    return updates


def plot_costs_and_performance(updates, m, logger, mode, learning_rate, plot_name_with_suffix):
    '''Arguments:
       - m is the Model_Model_preparation object
       - mode is one of {dhss, tfs, dhss_wtfs}
    '''
    fig = plt.figure(0, (13.5, 3.5))
    sns.set(font_scale=1.2)
    # sns.set_style("whitegrid")
    plt.suptitle(plot_name_with_suffix[:-4], fontsize=14)  # :-4 because this has .pdf at the end

    plt.subplot(1, 4, 1)
    plt.plot(np.arange(len(updates["train_loss"])), updates["train_loss"], label="Training set", color="salmon")
    plt.plot(np.arange(len(updates["test_loss"])), updates["test_loss"], label="Test set", color="steelblue")
    plt.legend()
    plt.xlabel("Iteration (step_size=30)")
    plt.ylabel("Cost")

    plt.subplot(1, 4, 2)
    plt.plot(np.arange(len(updates["yhat_test"])), updates["yhat_test"], label="Predicted (Test set)", color="steelblue")  # model prediction on test data
    plt.plot(np.arange(len(updates["yhat_test"])), m.testY, label="True (Test set)", color="steelblue", ls="--")
    # plt.plot(np.arange(len(updates["yhat_train"])), updates["yhat_train"], "--r", label="predicted_on_Train")  # model prediction on training data
    # plt.plot(np.arange(len(updates["yhat_train"])), m.trainY, "--r", label="actual")
    plt.legend()
    plt.xlabel("Cell line Index")
    plt.ylabel("TPM (scaled)")
    plt.ylim(0, 1.3)

    plt.subplot(1, 4, 3)
    try:
        sns.regplot(m.testY.flatten(), updates["yhat_test"].flatten(), robust=False, color="steelblue")
        sns.regplot(m.trainY.flatten(), updates["yhat_train"].flatten(), robust=False, fit_reg=False, scatter_kws={'alpha': 0.45}, color="salmon")
    except ValueError:
        logger.warning("ValueError encountered in trying to plot the regplot..")
        logger.warning("m.testY: {}".format(m.testY.flatten()))
        logger.warning("updates['yhat_test']: {}".format(updates["yhat_test"].flatten()))
        logger.warning("m.trainY: {}".format(m.trainY.flatten()))
        logger.warning("updates['yhat_train']: {}".format(updates["yhat_train"].flatten()))

    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    plt.xlim(0, max(1.1, np.max(m.testY.flatten()) + 0.1))
    plt.ylim(0, max(1.1, np.max(updates["yhat_test"].flatten()) + 0.1))
    plt.xlabel("True TPM (scaled)")
    plt.ylabel("Predicted TPM (scaled)")
    train_pcc = updates["train_pcc"][-1][0]
    train_loss = updates["train_loss"][-1]
    test_pcc = updates["test_pcc"][-1][0]
    test_loss = updates["test_loss"][-1]
    plt.title("PCC:{:.2f}; loss:{:.2f} (Test)".format(test_pcc, test_loss))
    logger.info("    gene:{0}, mode:{1}, lr:{2:.3f}, train_pcc:{3:.3f}, train_loss:{4:.3f}, test_pcc:{5:.3f}, test_loss:{6:.3f}".format(m.gene_ofInterest, mode, learning_rate, train_pcc, train_loss, test_pcc, test_loss))

    plt.subplot(1, 4, 4)
    # import pdb
    # pdb.set_trace()
    try:
        sns.kdeplot(m.testY.flatten(), updates["yhat_test"].flatten(), shade=True, cmap="Blues")  # can give some division by zero error
    except:
        logger.warning("Division by zero encountered in plotting the density performance plot..")

    logger.debug("    Done with making the performance plot")
    plt.title("Density plot for test set")
    plt.xlabel("True TPM (scaled)")
    plt.ylabel("Predicted TPM (scaled)")

    plt.tight_layout()
    plt.savefig(m.outputDir + "/" + plot_name_with_suffix)
    plt.close()
    return fig


def train_tensorflow_nn(logger, w1_, w2_, trainX_, trainY_, testX_, testY_,
                        max_iter=500, pkeep_train=0.7, lamda=0.0001,
                        starter_learning_rate=0.3, decay_at_step=30):
    '''Arugments:
    - All input arguments with "_" are real arrays to feed in as data.
    - lamda: regularization parameter
    - starter_learning_rate: Initial learning rate for the learning rate decay computation below
    - decay_at_step: learning rate is updated after this many training steps (Easier to think this way)
    '''

    # ------ Variables and placeholders ------
    updates = init_updates()

    lamda = 0.00005
    X = tf.placeholder(tf.float32, shape=[None, w1_.shape[0]], name="X")
    Y = tf.placeholder(tf.float32, [None, 1], name="Y")  # true Ys
    pkeep = tf.placeholder(tf.float32, name="pkeep")

    # # lamda = tf.Variable(tf.cast(lamda, tf.float32), name="lamda")  # to be used in loss function (currently not in use)
    W1 = tf.Variable(tf.cast(w1_, tf.float32), name="W1")
    b1 = tf.Variable(tf.zeros([w1_.shape[1]]), name="H1_bn_offset")  # bn == batch normalization
    g1 = tf.Variable(tf.ones([w1_.shape[1]]), name="H1_bn_scale")
    W2 = tf.Variable(tf.cast(w2_, tf.float32), name="W2")
    # b2 = tf.Variable(tf.zeros([w2_.shape[1]]), name="H1_bn_offset")

    Yhat, loss = tf_model(X, Y, W1, W2, b1, g1, pkeep, lamda)  # loss == rmse

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

        for i in range(max_iter):

            train_data = {X: trainX_, Y: trainY_, pkeep: pkeep_train}
            sess.run(train_step, feed_dict=train_data)
            updates["learning_rate"].append(learning_rate.eval())  # to be used with Adam opt.

            if (i % 10 == 0):
                test_data = {X: testX_, Y: testY_, pkeep: 1}
                updates = get_performance_updates(sess, loss, pcc, train_data, test_data, updates)

        # Now predict the performance, and get the weights
        updates["yhat_train"] = sess.run(Yhat, feed_dict={X: trainX_, Y: trainY_, pkeep: 1})  # model prediction on the training set
        updates["yhat_test"] = sess.run(Yhat, feed_dict={X: testX_, Y: testY_, pkeep: 1})
        if (np.isnan(updates["yhat_test"]).any()):
            logger.warning("ValueError enroute -- np.nan present in yhat_test")
            pdb.set_trace()
        updates["W1"] = W1.eval()
        updates["W2"] = W2.eval()

        # Get the layer1 and layer2 node sizes
        data_to_feed = {X: trainX_, Y: trainY_, pkeep: 1}
        sizes_layer1, sizes_layer2 = get_layer1_layer2_sizes_tfmodel(sess, updates, data_to_feed, X, Y, W1, W2, b1, g1, pkeep, lamda)
        sizes = {}
        sizes["layer1"] = sizes_layer1
        sizes["layer2"] = sizes_layer2
    return updates, sizes


def tf_model(X, Y, W1, W2, b1, g1, pkeep, lamda):
    # -------- the core model ---------
    H1 = tf.matmul(X, W1, name="H1")
    H1_mean, H1_var = tf.nn.moments(H1, axes=[0], keep_dims=True, name="H1_moments")
    H1_bn = tf.nn.batch_normalization(H1, H1_mean, H1_var, offset=b1,
                                      scale=g1, variance_epsilon=0.000001, name="H1_bn")  # perform batch normalization
    H1_bns = tf.nn.sigmoid(H1_bn, name="H1_bn_sigmoid")
    H1_bnd = tf.nn.dropout(H1_bns, pkeep, name="H1_after_bns_and_dropout")
    Yhat = tf.nn.sigmoid(tf.matmul(H1_bnd, W2), name="Yhat")
    loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(Yhat, Y)), name="loss")  # cost function to minimize
    regularizer = tf.add(tf.nn.l2_loss(W1), tf.nn.l2_loss(W2), name="regularizer")
    loss = loss + lamda * regularizer
    # loss = tf.reduce_mean((Y - Yhat)**2) + tf.multiply(lamda, tf.add(tf.reduce_sum(tf.square(W1)), tf.reduce_sum(tf.square(W2))))  # this is not working (yields negative loss); default norm is euclidean
    return Yhat, loss


def get_layer1_layer2_sizes_tfmodel(sess, updates, data_to_feed,
                                    X_tfp, Y_tfp, W1_tfp, W2_tfp, b1_tfp, g1_tfp, pkeep_tfp, lamda):
    '''
    This function is an adaptation of get_layer1_layer2_sizes(NN, X, Y) function in the helper_scripts.py.
    Compute the node sizes for the tensorflow(tf) NN model. The size of a node is computed by
    seeing how much removing that node affects the overall RMSE.

    Arguments:
        - tfp in the filename is short for tensorflow parameter
        - data_to_feed is a dict of this format {X: trainX, Y: testY, pkeep: 1} that is fed to tf model.
            Note that X and Y keys have values pertaining to the whole of training examples
            since we want the node sizes of the final model, and pkeep is 1.
        - updates is a dict with parameter values evaluated.
    '''

    # Set up the base Yhat and loss values to compare against.
    Yhat_, loss_ = tf_model(X_tfp, Y_tfp, W1_tfp, W2_tfp, b1_tfp, g1_tfp, pkeep_tfp, lamda)  # note the _ in the var names
    Yhat_base, loss_base = sess.run([Yhat_, loss_], feed_dict=data_to_feed)
    # rmse_base = np.sqrt(sum([x**2 for x in Yhat_base.flatten() - data_to_feed["Y"].flatten()])/data_to_feed["Y"].flatten().shape[0])

    # First get the weights for the first layer
    sizes_layer1 = np.zeros(updates["W1"].shape[0])
    for i in range(0, updates["W1"].shape[0]):
        w1 = copy.deepcopy(updates["W1"])
        w1[i, :] = 0
        assert not np.array_equal(updates["W1"], w1)

        W1_ = tf.Variable(tf.cast(w1, tf.float32), name="W1_")
        sess.run(tf.variables_initializer([W1_], name="W1_init"))  # need to initialize this new variable
        Yhat_, loss_ = tf_model(X_tfp, Y_tfp, W1_, W2_tfp, b1_tfp, g1_tfp, pkeep_tfp, lamda)  # only W1 is updated
        Yhat_shuf, loss_shuf = sess.run([Yhat_, loss_], feed_dict=data_to_feed)
        sizes_layer1[i] = (loss_shuf - loss_base) * 100
        # if (loss_shuf < loss_base):
        #    print("for layer 1: loss_shuf < loss_base, i", i, loss_shuf, loss_base, loss_shuf - loss_base)

    '''Similarly get the weights for the second layer'''
    sizes_layer2 = np.zeros(updates["W2"].shape[0])
    for i in range(0, updates["W2"].shape[0]):  # shape is (#hidden layers, 1)
        w2 = copy.deepcopy(updates["W2"])
        w2[i, 0] = 0
        assert not np.array_equal(updates["W2"], w2)

        W2_ = tf.Variable(tf.cast(w2, tf.float32), name="W2_")
        sess.run(tf.variables_initializer([W2_], name="W2_init"))  # need to initialize this new variable
        Yhat_, loss_ = tf_model(X_tfp, Y_tfp, W1_tfp, W2_, b1_tfp, g1_tfp, pkeep_tfp, lamda)  # only W2 is updated
        Yhat_shuf, loss_shuf = sess.run([Yhat_, loss_], feed_dict=data_to_feed)
        sizes_layer2[i] = (loss_shuf - loss_base) * 100
        # if (loss_shuf < loss_base):
        #    print("for layer 2: loss_shuf < loss_base, i", i, loss_shuf, loss_base, loss_shuf - loss_base)

    return sizes_layer1, sizes_layer2


def train_tensorflow_nn_scipy_needsWork(w1_, w2_, trainX_, trainY_, testX_, testY_,
                                        max_iter=250, pkeep_train=0.7,
                                        starter_learning_rate=0.3, decay_at_step=30):
    ''' this function is not working .. it seems to be stuck. callback is not implemented. needs work.
    Arugments:
    - All input arguments with "_" are real arrays to feed in as data.
    - starter_learning_rate: Initial learning rate for the learning rate decay computation below
    - decay_at_step: learning rate is updated after this many training steps (Easier to think this way)
    '''

    # ------ Variables and placeholders ------
    updates = init_updates()

    X = tf.placeholder(tf.float32, shape=[None, w1_.shape[0]], name="X")
    Y = tf.placeholder(tf.float32, [None, 1], name="Y")  # true Ys
    pkeep = tf.placeholder(tf.float32, name="pkeep")

    W1 = tf.Variable(tf.cast(w1_, tf.float32), name="W1")
    b1 = tf.Variable(tf.zeros([w1_.shape[1]]), name="H1_bn_offset")  # bn == batch normalization
    g1 = tf.Variable(tf.ones([w1_.shape[1]]), name="H1_bn_scale")
    W2 = tf.Variable(tf.cast(w2_, tf.float32), name="W2")
    # b2 = tf.Variable(tf.zeros([w2_.shape[1]]), name="H1_bn_offset")

    # -------- model ---------
    H1 = tf.matmul(X, W1, name="H1")
    H1_mean, H1_var = tf.nn.moments(H1, axes=[0], keep_dims=True, name="H1_moments")
    H1_bn = tf.nn.batch_normalization(H1, H1_mean, H1_var, offset=b1, scale=g1, variance_epsilon=0.000001, name="H1_bn")  # perform batch normalization
    H1_bns = tf.nn.sigmoid(H1_bn, name="H1_bn_sigmoid")
    H1_bnd = tf.nn.dropout(H1_bns, pkeep, name="H1_after_bns_and_dropout")
    Yhat = tf.nn.sigmoid(tf.matmul(H1_bnd, W2), name="Yhat")
    sse = tf.reduce_mean((Y - Yhat)**2, name="sse")  # cost function to minimize
    # sse = tf.sum(tf.divide(tf.square(Y - Yhat), w1_.shape[0]),

    # ------ train parameters -----
    '''
    global_step = tf.Variable(0, trainable=False, name="global_step")  # this is like a counter when passed to minimize() function
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               decay_steps=decay_at_step, decay_rate=0.96, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(sse, global_step=global_step)
    '''

    # option 2
    train_step = tf.contrib.opt.ScipyOptimizerInterface(sse, method="L-BFGS-B",
                                                        options={'maxiter': max_iter, 'gtol': 1e-8, 'disp': True})

    # ------ performance metric -----
    # pcc = tf.contrib.metrics.streaming_pearson_correlation(Yhat, Y, name="pcc")

    # ------ start training ------
    with tf.Session() as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        train_data = {X: trainX_, Y: trainY_, pkeep: pkeep_train}
        train_step.minimize(sess, feed_dict=train_data)
        # sess.run(train_step, feed_dict=train_data)  # to be used with Adam Opt
        # updates["learning_rate"].append(learning_rate.eval())  # to be used with Adam opt.

        '''
        # if (i % 30 == 0):
        test_data = {X: testX_, Y: testY_, pkeep: 1}
        updates = get_performance_updates(sess, sse, pcc, rmse, train_data, test_data, updates)

        # Now predict the performance, and get the weights
        updates["yhat_train"] = sess.run(Yhat, feed_dict={X: trainX_, Y: trainY_, pkeep: 1})  # model prediction on the training set
        updates["yhat_test"] = sess.run(Yhat, feed_dict={X: testX_, Y: testY_, pkeep: 1})
        updates["W1"] = W1.eval()
        updates["W2"] = W2.eval()
        '''
    return updates
