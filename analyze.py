import util
import json
import pickle
from os.path import join
import os
from collections import defaultdict
import tensorflow as tf

singular_pronouns = ['i', 'me', 'my', 'mine', 'myself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'yourself']
plural_pronouns = ['they', 'them', 'their', 'theirs', 'themselves', 'we', 'us', 'our', 'ours', 'ourselves', 'yourselves']
ambiguous_pronouns = ['you', 'your', 'yours']
valid_pronouns = singular_pronouns + plural_pronouns + ambiguous_pronouns


def get_prediction_path(config, config_name, saved_suffix):
    dir_analysis = join(config['data_dir'], 'analysis')
    os.makedirs(dir_analysis, exist_ok=True)

    name = f'pred_{config_name}_{saved_suffix}.bin'
    path = join(dir_analysis, name)
    return path


def get_prediction(config_name, saved_suffix):
    conf = util.initialize_from_env_2(config_name, saved_suffix)

    path = get_prediction_path(conf, config_name, saved_suffix)
    if os.path.exists(path):
        # Load if saved
        with open(path, 'rb') as f:
            prediction = pickle.load(f)
        print('Loaded prediction from %s' % path)
    else:
        # Get prediction
        model = util.get_model(conf)
        with tf.Session() as session:
            model.restore(session)
            predicted_clusters, predicted_spans, predicted_antecedents = model.predict_test(session)
        tf.reset_default_graph()
        prediction = (predicted_clusters, predicted_spans, predicted_antecedents)

        # Save
        with open(path, 'wb') as f:
            pickle.dump(prediction, f)
        print('Prediction saved in %s' % path)

    return prediction


def get_original_samples(config):
    samples = []
    paths = {
        'trn': join(config['data_dir'], f'train.english.{config["max_segment_len"]}.jsonlines'),
        'dev': join(config['data_dir'], f'dev.english.{config["max_segment_len"]}.jsonlines'),
        'tst': join(config['data_dir'], f'test.english.{config["max_segment_len"]}.jsonlines')
    }
    with open(paths['tst']) as fin:
        for line in fin.readlines():
            data = json.loads(line)
            samples.append(data)
    return samples


def check_singular_plural_cluster(cluster):
    """ Cluster with text """
    singular, plural, contain_ambiguous = False, False, False
    for m in cluster:
        if singular and plural:
            break
        m = m.lower()
        if not singular:
            singular = (m in singular_pronouns)
        if not plural:
            plural = (m in plural_pronouns)
    for m in cluster:
        m = m.lower()
        if m in ambiguous_pronouns:
            contain_ambiguous = True
            break
    return singular, plural, contain_ambiguous


def analyze(config_name, saved_suffix):
    conf = util.initialize_from_env_2(config_name, saved_suffix)

    example_list = get_original_samples(conf)

    # Get prediction
    predicted_clusters, predicted_spans, predicted_antecedents = get_prediction(config_name, saved_suffix)

    # Get cluster text
    cluster_list = []
    subtoken_list = []
    for i, example in enumerate(example_list):
        subtokens = util.flatten(example['sentences'])
        subtoken_list.append(subtokens)
        cluster_list.append([[' '.join(subtokens[m[0]: m[1] + 1]) for m in c] for c in predicted_clusters[i]])

    # Get cluster stats
    num_clusters, num_singular_clusters, num_plural_clusters, num_mixed_clusters, num_mixed_ambiguous = 0, 0, 0, 0, 0
    for clusters in cluster_list:
        # print(clusters)
        for c in clusters:
            singular, plural, contain_ambiguous = check_singular_plural_cluster(c)
            num_clusters += 1
            if singular and plural:
                num_mixed_clusters += 1
                if contain_ambiguous:
                    num_mixed_ambiguous += 1
            if singular:
                num_singular_clusters += 1
            if plural:
                num_plural_clusters += 1

    # Get gold clusters
    gold_to_cluster_id = []  # 0 means not in cluster
    non_anaphoric = []  # Firstly appeared mention in a cluster
    for i, example in enumerate(example_list):
        gold_to_cluster_id.append(defaultdict(int))
        non_anaphoric.append(set())

        clusters = example['clusters']
        clusters = [sorted(cluster) for cluster in clusters]  # Sort mention
        for c_i, c in enumerate(clusters):
            non_anaphoric[i].add(tuple(c[0]))
            for m in c:
                gold_to_cluster_id[i][tuple(m)] = c_i + 1

    # Get antecedent stats
    fl, fn, wl, correct = 0, 0, 0, 0  # False Link, False New, Wrong Link
    s_to_p, p_to_s = 0, 0
    num_non_gold, num_total_spans = 0, 0
    for i, antecedents in enumerate(predicted_antecedents):
        antecedents = [(-1, -1) if a == -1 else predicted_spans[i][a] for a in antecedents]
        for j, antecedent in enumerate(antecedents):
            span = predicted_spans[i][j]
            span_cluster_id = gold_to_cluster_id[i][span]
            num_total_spans += 1

            if antecedent == (-1, -1):
                continue

            # Only look at stats of pronouns
            span_text = ' '.join(subtoken_list[i][span[0]: span[1] + 1]).lower()
            antecedent_text = ' '.join(subtoken_list[i][antecedent[0]: antecedent[1] + 1]).lower()
            if span_text not in valid_pronouns or antecedent_text not in valid_pronouns:
                continue

            if span_text in singular_pronouns and antecedent_text in plural_pronouns:
                s_to_p += 1
            elif span_text in plural_pronouns and antecedent_text in singular_pronouns:
                p_to_s += 1

            if span_cluster_id == 0:  # Non-gold span
                num_non_gold += 1
                if antecedent == (-1, -1):
                    correct += 1
                else:
                    fl += 1
            elif span in non_anaphoric[i]:  # Non-anaphoric span
                if antecedent == (-1, -1):
                    correct += 1
                else:
                    fl += 1
            else:
                if antecedent == (-1, -1):
                    fn += 1
                elif span_cluster_id != gold_to_cluster_id[i][antecedent]:
                    wl += 1
                else:
                    correct += 1

    return num_clusters, num_singular_clusters, num_plural_clusters, num_mixed_clusters, num_mixed_ambiguous, fl, fn, wl, correct, \
           num_non_gold, num_total_spans, s_to_p, p_to_s


if __name__ == '__main__':
    gpu_id = 6
    util.set_gpus(gpu_id)

    experiments = [('train_spanbert_large_ee', 'May14_06-02-15'),
                   ('train_spanbert_large_ee', 'May14_06-05-42'),
                   ('train_spanbert_large_lr2e-4_ee', 'May14_06-03-24'),
                   ('train_spanbert_large_lr2e-4_ee', 'May14_06-10-51')]

    results_final = None
    for experiment in experiments:
        results = analyze(*experiment)
        if results_final is None:
            results_final = results
        else:
            results_final = [r + results[i] for i, r in enumerate(results_final)]

        # print('%s_%s: # clusters: %d; # singular clusters: %d; # plural clusters: %d; # mixed clusters: %d; '
        #       'FL %d; FN: %d; WL: %d; CORRECT %d; # gold spans: %d; # total spans: %d' % (*experiment, *results))

    results_final = [r / len(experiments) for r in results_final]
    print('Avg: # clusters: %.3f; # singular clusters: %.3f; # plural clusters: %.3f; # mixed clusters: %.3f; # mixed with ambiguous: %.3f; '
          'FL %.3f; FN: %.3f; WL: %.3f; CORRECT %.3f; # gold spans: %.3f; # total spans: %.3f; # S to P: %.3f; # P to S: %.3f' % (*results_final,))
