import tensorflow as tf 
import numpy as np 
import os, re
import time, csv
import datetime, yaml
from tensorflow.contrib import learn 
from utils import *


with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

class EvaluationNCF():

    def __init__(self, top_n):
        self.model_path = cfg['NCF_model_path']
        self.checkpoint_dir = self.model_path + "/checkpoints"
        self.checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)
        self.graph = tf.Graph()
        self.top_n = top_n
        self.batch_size = 64
        with self.graph.as_default():
            session_conf = tf.ConfigProto(
                    allow_soft_placement = True,
                    log_device_placement = False
            )
            self.sess = tf.Session(config = session_conf)
            with self.sess.as_default():
                    self.saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                    self.saver.restore(self.sess, self.checkpoint_file)
    
    def query(self, inputs):
        all_predictions = []
        input_users, input_items = [], []
        user_id = inputs[0]
        book_ids, _ = list(zip(*inputs[2]))
        for book_id in book_ids:
            input_users.append(user_id - 1)
            input_items.append(book_id - 1)
            
        input_user = self.graph.get_operation_by_name("input_user").outputs[0]
        input_item = self.graph.get_operation_by_name("input_item").outputs[0]
        dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        scores = self.graph.get_operation_by_name("output/scores").outputs[0]
        batches = batch_iter(list(zip(input_users, input_items)), self.batch_size, 1, shuffle = False)
        for batch in batches:
            x_user, x_item = zip(*batch)
            batche_prediction = self.sess.run(scores, 
                    {input_user: x_user,
                    input_item: x_item,
                    dropout_keep_prob: 1.0}
            )

            for i in range(len(batche_prediction)):
                sfm = batche_prediction[i]
                all_predictions.append((sfm[0], x_item[i]))
        
        all_predictions = sorted(all_predictions, reverse = True)
        top_ids = [x[1] + 1 for x in all_predictions[:self.top_n]]
        return top_ids