
import tensorflow as tf
import numpy as np


class NCF(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, nb_users, nb_items, mlp_layer_sizes, mf_dim, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_user = tf.placeholder(tf.int32, [None], name="input_user")
        self.input_item = tf.placeholder(tf.int32, [None], name="input_item")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        embedding_size = mf_dim + mlp_layer_sizes[0] // 2
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W_users = tf.get_variable(
                # tf.random_uniform([nb_items, embedding_size], -0.1, 0.1),
                initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01),
                shape = [nb_items, embedding_size],
                name="W_user",
                trainable = True)
            self.W_items = tf.get_variable(
                # tf.random_uniform([nb_users, embedding_size], -0.1, 0.1),
                initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01),
                shape = [nb_users, embedding_size],
                name="W_item",
                trainable = True)
            # Matrix Factorization Embedding
            self.embedded_mf_user = tf.nn.embedding_lookup(self.W_users[:, :mf_dim], self.input_user, partition_strategy='div')
            self.embedded_mf_item = tf.nn.embedding_lookup(self.W_items[:, :mf_dim], self.input_item, partition_strategy='div')
            # MLP Embedding
            self.embedded_mlp_user = tf.nn.embedding_lookup(self.W_users[:, mf_dim:], self.input_user, partition_strategy='div')
            self.embedded_mlp_item = tf.nn.embedding_lookup(self.W_items[:, mf_dim:], self.input_item, partition_strategy='div')


        # MF Layer
        mf_output = tf.math.multiply(self.embedded_mf_item, self.embedded_mf_user)

        # MLP layer
        mlp_input = tf.concat((self.embedded_mlp_item, self.embedded_mlp_user), 1)

        for i in range(1, len(mlp_layer_sizes)):
            prev_size = mlp_layer_sizes[i - 1]
            current_size = mlp_layer_sizes[i]
            W_cur = tf.get_variable(
                name = "W_cur" + str(i),
                shape=[prev_size, current_size],
                initializer=tf.contrib.layers.xavier_initializer())
            bias_cur = tf.Variable(tf.constant(0.1, shape=[current_size]), name = "bias" + str(i))
            h = tf.add(tf.matmul(mlp_input, W_cur), bias_cur)
            mlp_input = tf.nn.relu(h)
            mlp_input = tf.nn.dropout(mlp_input, self.dropout_keep_prob)
        
        mlp_output = mlp_input
        
        # Final fully connected 
        logits = tf.concat((mf_output, mlp_output), 1)
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[mlp_layer_sizes[-1] + mf_dim, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            self.scores = tf.nn.xw_plus_b(logits, W, b, name="scores")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
