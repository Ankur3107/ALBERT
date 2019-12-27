### Step are:

1.  Convert data into model format:
    
        export SQUAD_DIR=SQuAD
        export SQUAD_VERSION=v1.1
        export ALBERT_DIR=large
        export OUTPUT_DIR=squad_out_${SQUAD_VERSION}
        mkdir $OUTPUT_DIR
        
        export MODEL_DIR=xxlarge
        mkdir ${MODEL_DIR}
        tar -xvzf model.tar.gz --directory=${MODEL_DIR}
        #Converting weights to TF 2.0
        CUDA_VISIBLE_DEVICES=1 python converter.py --tf_hub_path=${MODEL_DIR}/ --model_type=albert_encoder --version=2 --model=xxlarge
        #Copy albert_config.json to config.json
        cp ${MODEL_DIR}/assets/albert_config.json ${MODEL_DIR}/config.json
        #Rename assets to vocab
        mv ${MODEL_DIR}/assets/ ${MODEL_DIR}/vocab
        #Delete unwanted files
        rm -rf ${MODEL_DIR}/saved_model.pb ${MODEL_DIR}/variables/ ${MODEL_DIR}/saved_model.pb ${MODEL_DIR}/tfhub_module.pb

2.  Train model on multi gpu:

        CUDA_VISIBLE_DEVICES=1,2,3 python run_squad.py \
        --mode=train_and_predict \
        --input_meta_data_path=${OUTPUT_DIR}/squad_${SQUAD_VERSION}_meta_data \
        --train_data_path=${OUTPUT_DIR}/squad_${SQUAD_VERSION}_train.tf_record \
        --predict_file=${SQUAD_DIR}/dev-${SQUAD_VERSION}.json \
        --albert_config_file=${ALBERT_DIR}/config.json \
        --init_checkpoint=${ALBERT_DIR}/tf2_model.h5 \
        --spm_model_file=${ALBERT_DIR}/vocab/30k-clean.model \
        --train_batch_size=24 \
        --predict_batch_size=24 \
        --learning_rate=1e-5 \
        --num_train_epochs=3 \
        --model_dir=${OUTPUT_DIR} \
        --strategy_type=mirror

3.  Test data:

        CUDA_VISIBLE_DEVICES=1,2,3 python run_squad.py --mode=predict \
        --albert_config_file=${ALBERT_DIR}/config.json \
        --model_dir=${OUTPUT_DIR} \
        --input_meta_data_path=${OUTPUT_DIR}/squad_v1.1_meta_data \
        --predict_file=${SQUAD_DIR}/dev-v1.1.json \
        --spm_model_file=${ALBERT_DIR}/vocab/30k-clean.model \
        --strategy_type=mirror

4. Evaluate Accuracy:

        from __future__ import print_function
        from collections import Counter
        import string
        import re
        import argparse
        import json
        import sys


        def normalize_answer(s):
            """Lower text and remove punctuation, articles and extra whitespace."""
            def remove_articles(text):
                return re.sub(r'\b(a|an|the)\b', ' ', text)

            def white_space_fix(text):
                return ' '.join(text.split())

            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return white_space_fix(remove_articles(remove_punc(lower(s))))


        def f1_score(prediction, ground_truth):
            prediction_tokens = normalize_answer(prediction).split()
            ground_truth_tokens = normalize_answer(ground_truth).split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1


        def exact_match_score(prediction, ground_truth):
            return (normalize_answer(prediction) == normalize_answer(ground_truth))


        def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
            scores_for_ground_truths = []
            for ground_truth in ground_truths:
                score = metric_fn(prediction, ground_truth)
                scores_for_ground_truths.append(score)
            return max(scores_for_ground_truths)


        def evaluate(dataset, predictions):
            f1 = exact_match = total = 0
            for article in dataset:
                for paragraph in article['paragraphs']:
                    for qa in paragraph['qas']:
                        total += 1
                        if qa['id'] not in predictions:
                            message = 'Unanswered question ' + qa['id'] + \
                                    ' will receive score 0.'
                            print(message, file=sys.stderr)
                            continue
                        ground_truths = list(map(lambda x: x['text'], qa['answers']))
                        prediction = predictions[qa['id']]
                        exact_match += metric_max_over_ground_truths(
                            exact_match_score, prediction, ground_truths)
                        f1 += metric_max_over_ground_truths(
                            f1_score, prediction, ground_truths)

            exact_match = 100.0 * exact_match / total
            f1 = 100.0 * f1 / total

            return {'exact_match': exact_match, 'f1': f1}

        if True:
            expected_version = '1.1'
            dataset_file = 'dev-v1.1.json'
            prediction_file = 'predictions.json'
            
            with open(dataset_file) as dataset_file:
                dataset_json = json.load(dataset_file)
                dataset = dataset_json['data']
            with open(prediction_file) as prediction_file:
                predictions = json.load(prediction_file)
            print(json.dumps(evaluate(dataset, predictions)))