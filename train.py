from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import util
import logging
import numpy as np


format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
  config = util.initialize_from_env()

  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]

  model = util.get_model(config)
  saver = tf.train.Saver()

  log_dir = config["log_dir"]
  max_steps = config['num_epochs'] * config['num_docs']
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)

  max_f1 = 0
  mode = 'w'

  tf_conf = tf.ConfigProto()
  tf_conf.gpu_options.allow_growth = True
  with tf.Session(config=tf_conf) as session:
    session.run(tf.global_variables_initializer())
    model.start_enqueue_thread(session)
    accumulated_loss = 0.0

    # ckpt = tf.train.get_checkpoint_state(log_dir)
    ckpt = None
    if ckpt and ckpt.model_checkpoint_path:
      print("Restoring from: {}".format(ckpt.model_checkpoint_path))
      saver.restore(session, ckpt.model_checkpoint_path)
      mode = 'a'
    fh = logging.FileHandler(os.path.join(log_dir, 'stdout.log'), mode=mode)
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)

    initial_time = time.time()
    while True:
      tf_loss, tf_global_step, _, tf_marginalized, tf_log_norm, tf_bert_lr, tf_task_lr = session.run([model.loss, model.global_step, model.train_op,
                                                model.marginalized_gold_scores, model.log_norm, model.bert_lr, model.task_lr])
      # print('{}\t{}'.format(tf_marginalized, tf_log_norm))
      # print(np.sum(tf_log_norm).item())
      # print(np.sum(tf_marginalized).item())
      # print(tf_loss)
      # print('-------')
      accumulated_loss += tf_loss
      # print('tf global_step', tf_global_step)

      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        logger.info("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
        # logger.info('bert_lr / task_lr: {} / {}'.format(tf_bert_lr, tf_task_lr))
        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0

      if tf_global_step  > 0 and tf_global_step % eval_frequency == 0:
        saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        eval_summary, eval_f1 = model.evaluate(session, tf_global_step)

        if eval_f1 > max_f1:
          max_f1 = eval_f1
          util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

        writer.add_summary(eval_summary, tf_global_step)
        writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

        logger.info("[{}] evaL_f1={:.4f}, max_f1={:.4f}".format(tf_global_step, eval_f1, max_f1))
        if tf_global_step > max_steps:
          break
