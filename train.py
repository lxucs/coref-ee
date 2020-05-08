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
logger = logging.getLogger()
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
      # tf_loss, tf_global_step, _, tf_mention_doc, tf_input_ids, tf_candidate_starts, tf_candidate_ends, tf_candidate_cluster_ids, tf_candidate_mention_scores, tf_candidate_span_emb, tf_span_start_emb, tf_span_end_emb, tf_span_width_index, tf_span_width_emb, tf_mention_word_scores, tf_head_attn_reps, tf_span_width_embeddings, tf_top_span_indices, tf_top_span_ends, tf_top_span_emb, tf_top_span_mention_scores, tf_top_span_speaker_ids, tf_genre_emb, tf_top_antecedents_idx, tf_top_antecedents_mask, tf_top_fast_antecedent_scores, tf_top_antecedent_offsets, tf_antecedent_distance_scores, tf_fast_antecedent_scores, tf_segment_distance, tf_antecedent_distance_buckets, tf_top_antecedent_emb, tf_top_antecedent_scores, tf_top_antecedent_cluster_ids, tf_same_cluster_indicator, tf_non_dummy_indicator, tf_pairwise_labels, tf_top_antecedent_labels, tf_final_loss, tf_marginalized, tf_log_norm, tf_bert_lr, tf_task_lr, tf_span, tf_gold = \
      #   session.run([model.loss, model.global_step, model.train_op, model.mention_doc, model.input_ids,
      #     model.candidate_starts, model.candidate_ends, model.candidate_cluster_ids, model.candidate_mention_scores, model.candidate_span_emb,
      #     model.span_start_emb, model.span_end_emb, model.span_width_index, model.span_width_emb, model.mention_word_scores, model.head_attn_reps, model.span_width_embeddings,
      #     model.top_span_indices, model.top_span_ends, model.top_span_emb, model.top_span_mention_scores, model.top_span_speaker_ids, model.genre_emb,
      #     model.top_antecedents_idx, model.top_antecedents_mask, model.top_fast_antecedent_scores, model.top_antecedent_offsets, model.antecedent_distance_scores, model.fast_antecedent_scores,
      #     model.segment_distance, model.antecedent_distance_buckets, model.top_antecedent_emb, model.top_antecedent_scores, model.top_antecedent_cluster_ids, model.same_cluster_indicator, model.non_dummy_indicator, model.pairwise_labels, model.top_antecedent_labels, model.final_loss,
      #     model.marginalized_gold_scores, model.log_norm, model.bert_lr, model.task_lr,
      #     model.num_span, model.gold_span])
      tf_loss, tf_global_step, _, tf_marginalized, tf_log_norm, tf_bert_lr, tf_task_lr, tf_span, tf_gold = \
        session.run([model.loss, model.global_step, model.train_op,
                     model.marginalized_gold_scores, model.log_norm, model.bert_lr, model.task_lr,
                     model.num_span, model.gold_span])
      if tf_global_step % 20 == 0:
        logger.info('---------debug step: %d---------' % tf_global_step)
        logger.info('spans/gold: %d/%d; ratio: %.2f' % (tf_span.item(), tf_gold.sum().item(), tf_gold.sum().item()/tf_span.item()))
        logger.info('norm/gold: %.4f/%.4f' % (np.sum(tf_log_norm).item(), np.sum(tf_marginalized).item()))
        # logger.info(tf_top_fast_antecedent_scores[6][:5])
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

      if tf_global_step > 0 and tf_global_step % eval_frequency == 0:
        if tf_global_step > 40000:
          saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        eval_summary, eval_f1 = model.evaluate(session, tf_global_step)

        if eval_f1 > max_f1:
          max_f1 = eval_f1
          if tf_global_step > 40000:
            util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

        writer.add_summary(eval_summary, tf_global_step)
        writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

        logger.info("[{}] evaL_f1={:.4f}, max_f1={:.4f}".format(tf_global_step, eval_f1, max_f1))
        if tf_global_step > max_steps:
          break
