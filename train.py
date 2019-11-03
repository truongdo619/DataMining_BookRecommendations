import tensorflow as tf
import numpy as np
import os
import time
import datetime
from NCF import NCF
from tensorflow.contrib import learn
import yaml
import math
from utils import *


# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("mf_dim", 16, "Dimensionality of character embedding (default: 16)")
tf.flags.DEFINE_string("mlp_layer_sizes", "32,16,8", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("limit_early_stop", 10, "Early stop parameter")
tf.flags.DEFINE_integer("timestamp", 1, "Time Stamp")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("decay_coefficient", 2.5, "Decay coefficient (default: 2.5)")

FLAGS = tf.flags.FLAGS

data_training = data
def run_train(_timestamp):
    # Load data
    print("Loading data...")

    _data = data
    np.random.shuffle(_data)
    num_dev = int(FLAGS.dev_sample_percentage * len(data_training))
    print(data)
    data_train = _data[num_dev:]
    data_dev = _data[:num_dev]

    x_user_train, x_item_train = list(zip(*data_train))
    x_user_train = [user_id - 1 for user_id in x_user_train]
    x_item_train = [item_id - 1 for item_id in x_item_train]
    
    x_user_dev, x_item_dev = list(zip(*data_dev))
    x_user_dev = [user_id - 1 for user_id in x_user_dev]
    x_item_dev = [item_id - 1 for item_id in x_item_dev]
    
    y_train = [[1] for _ in range(0, len(data_train))]
    y_dev = [[1] for _ in range(0, len(data_dev))]


    print("Number of users: ", nb_user)
    print("Number of items: ", nb_item)
    print("Number of interactions for train: ", len(data_train))
    print("Number of interactions for dev: ", len(data_dev))

    

    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            ncf = NCF(
                nb_users = nb_user,
                nb_items = nb_item,
                mlp_layer_sizes = [int(x) for x in FLAGS.mlp_layer_sizes.split(',')],
                mf_dim = FLAGS.mf_dim)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0,
                                                    rho=0.95,
                                                    epsilon=1e-06)
            # optimizer = tf.train.AdamOptimizer(ncf.learning_rate)
            grads_and_vars = optimizer.compute_gradients(ncf.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(_timestamp)
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "modelsNCF", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", ncf.loss)
            

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            
            def train_step(x_users, x_items, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    ncf.input_user: x_users,
                    ncf.input_item: x_items,
                    ncf.input_y: y_batch,
                    ncf.dropout_keep_prob: FLAGS.dropout_keep_prob,
                }
                _, step, summaries, loss = sess.run(
                    [train_op, global_step, train_summary_op, ncf.loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}"
                    .format(time_str, step, loss))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(batches, writer=None):
                """
                Evaluates model on a dev set
                """
                sum_loss = 0.0
                avg_count = 0
                for batch in batches:
                    x_users, x_items, y_batch = zip(*batch)
                    feed_dict = {
                        ncf.input_user: x_users,
                        ncf.input_item: x_items,
                        ncf.input_y : y_batch,
                        ncf.dropout_keep_prob: 1.0
                    }
                    step, summaries, loss = sess.run(
                        [global_step, dev_summary_op, ncf.loss],
                        feed_dict)
                    sum_loss += loss
                    avg_count += 1
                    if writer:
                        writer.add_summary(summaries, step)
                loss = sum_loss / avg_count
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {}".format(time_str, step, loss))
                return loss

            # Generate batches
            batches = batch_iter(
                list(zip(x_user_train, x_item_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            batches_dev = batch_iter(
                list(zip(x_user_dev, x_item_dev, y_dev)), FLAGS.batch_size, 1)
            
            # Training loop. For each batch...
            counter = 0
            best_loss_so_far = 1000.0
            early_stop_count = 0
            for batch in batches:
                counter += 1
                x_batch_user, x_batch_item, y_batch = zip(*batch)
                # print(ed_batch)
                train_step(x_batch_user, x_batch_item, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    loss = dev_step(batches_dev, writer=dev_summary_writer)
                    if loss < best_loss_so_far:
                        best_loss_so_far = loss
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                        early_stop_count = 0
                    else:
                        early_stop_count += 1
                        if early_stop_count == FLAGS.limit_early_stop:
                            print("Early stop at loss: ", best_loss_so_far)
                            break
                    print("")

run_train('test')