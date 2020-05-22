#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import util
import logging

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def read_doc_keys(fname):
    keys = set()
    with open(fname) as f:
        for line in f:
            keys.add(line.strip())
    return keys

if __name__ == "__main__":
  # Eval dev
  # config = util.initialize_from_env()
  # model = util.get_model(config)
  # saver = tf.train.Saver()
  # with tf.Session() as session:
  #   model.restore(session)
  #   # Make sure eval mode is True if you want official conll results
  #   model.evaluate(session, official_stdout=True, eval_mode=True, visualize=False)

  # Eval test
  config = util.initialize_from_env(name_suffix='May14_06-10-51')
  model = util.get_model(config)
  saver = tf.train.Saver()
  with tf.Session() as session:
      model.restore(session)
      # Make sure eval mode is True if you want official conll results
      model.evaluate_test(session, official_stdout=True, eval_mode=True, visualize=False)
