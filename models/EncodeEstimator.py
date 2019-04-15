import os

import numpy as np
import tensorflow as tf

from models.Encoder import EncoderFactory
from models.Losses import LossesFactory


class EncodeEstimator:
    def __init__(self, params):
        self.max_length = params['max_length']
        self.term_size = params['term_size']
        self.doc_vec_length = params['doc_vec_length']
        self.vocab_size = params['embeddings'].shape[0]
        self.padding_length = params['padding_length']
        self.num_epochs = params['num_epochs']
        self.batch_size = params['batch_size']

        # instance attribute declaration
        self.n = None
        self.num_steps_per_epoch = None

        self.w_embed = None

        # create estimator
        self.estimator = tf.estimator.Estimator(
            self.model_fn,
            config=tf.estimator.RunConfig(
                model_dir=params['model_dir'],
                tf_random_seed=24,
                save_checkpoints_steps=200,
                keep_checkpoint_max=3),
            params=params)

        # create encoder
        self.encoder = EncoderFactory.get_encoder(params)

        # create loss_fn
        self.loss_fn = LossesFactory.get_loss_fn(params)

    def train_input_fn(self, docs, labels_lm, labels):
        dataset = self.eval_input_fn(docs, labels_lm, labels)
        dataset = dataset.shuffle(self.n).repeat(self.num_epochs)
        return dataset

    def eval_input_fn(self, docs, labels_lm, labels):
        # todo: add unknown words to word vectors

        # truncates docs
        docs = [i[:self.max_length] for i in docs]

        docs_len = [len(doc) for doc in docs]

        dataset_docs = tf.data.Dataset.from_generator(lambda: docs, tf.int32)
        dataset_len = tf.data.Dataset.from_generator(lambda: docs_len, tf.int32)
        dataset_labels_lm = tf.data.Dataset.from_generator(lambda: labels_lm, tf.int32)
        dataset_labels = tf.data.Dataset.from_generator(lambda: labels, tf.int32)

        padding_value = self.vocab_size - 1

        # "dynamic padding": each batch might have different
        # sequence lengths but the sequence lengths within
        # one batch are the same the data type of padding_values
        # should match that of the dataset! both tf.int32 in this case
        dataset_docs = dataset_docs.padded_batch(
            self.batch_size,
            [self.padding_length],
            padding_values=padding_value)

        dataset_len = dataset_len.batch(self.batch_size)

        dataset_labels_lm = dataset_labels_lm.padded_batch(
            self.batch_size, [-1], padding_values=padding_value)

        dataset_labels = dataset_labels.padded_batch(
            self.batch_size, [-1], padding_values=-1)

        dataset_labels = self.loss_fn.get_sparse_representation(
            dataset_labels, self.term_size)

        # zip up dataset
        dataset_feature = tf.data.Dataset.zip(
            (dataset_docs, dataset_len, dataset_labels_lm))

        dataset_feature_dict = dataset_feature.map(
            lambda sen, sen_len, lab_lm: {
                'sentence': sen, 'length': sen_len, 'labels_lm': lab_lm})

        dataset = tf.data.Dataset.zip((dataset_feature_dict, dataset_labels))
        return dataset

    def predict_input_fn(self, docs):
        # truncates docs
        docs = [i[:self.max_length] for i in docs]

        docs_len = [len(doc) for doc in docs]

        dataset_docs = tf.data.Dataset.from_generator(lambda: docs, tf.int32)

        dataset_len = tf.data.Dataset.from_generator(lambda: docs_len, tf.int32)

        # pad and batch
        dataset_docs = dataset_docs.padded_batch(
            self.batch_size, [self.padding_length],
            padding_values=self.vocab_size-1)

        dataset_len = dataset_len.batch(self.batch_size)

        # zip dataset
        dataset_feature = tf.data.Dataset.zip(
            (dataset_docs, dataset_len))

        dataset_feature_dict = dataset_feature.map(
            lambda sen, sen_len: {'sentence': sen, 'length': sen_len})

        return dataset_feature_dict

    def model_fn(self, features, labels, mode, params):
        """ Model function used in the estimator.
        Args:
            features (Tensor): Input features to the model.
            labels (Tensor): Labels tensor for training and evaluation.
            mode (ModeKeys): Specifies if training, evaluation or prediction.
            params (Dict):
        Returns:
            (EstimatorSpec): Model to be run by Estimator.
        """

        self.w_embed = tf.get_variable(
            'w_embed',
            shape=[self.vocab_size, params['embeddings_dim']],
            initializer=tf.constant_initializer(params['embeddings'], verify_shape=True))

        del params['embeddings']

        with tf.name_scope("embedding"):
            x = tf.nn.embedding_lookup(self.w_embed, features['sentence'], name='embed')

        doc_vecs, logits = self.encoder.encode(x, features['length'])

        with tf.name_scope("get_predicted_labels"):
            pred_labels_one_hot, pred_labels = self.loss_fn.get_pred_labels(logits, self.term_size)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'doc_vecs': doc_vecs, 'pred_labels': pred_labels}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        with tf.name_scope("embedding_lm"):
            labels_lm = tf.nn.embedding_lookup(self.w_embed, features['labels_lm'], name='embed_lm')

        with tf.name_scope("cross_entropy"):
            loss = self.loss_fn.get_loss(
                logits, labels_lm, labels, pred_labels_one_hot, doc_vecs, params['alpha'])

        with tf.name_scope("metrics"):
            acc_op = tf.metrics.accuracy(labels, pred_labels_one_hot)
        #     labels = tf.cast(labels, tf.int64)
        #     pred_labels_one_hot = tf.cast(pred_labels_one_hot, tf.int64)
        #     recall = tf.metrics.recall(labels, pred_labels_one_hot)
        #     precision = tf.metrics.precision(labels, pred_labels_one_hot)
        #     p_at_1 = tf.metrics.precision_at_k(labels, pred_labels_one_hot, 1)
        #     p_at_3 = tf.metrics.precision_at_k(labels, pred_labels_one_hot, 3)
        #     p_at_5 = tf.metrics.precision_at_k(labels, pred_labels_one_hot, 5)
        #     r_at_1 = tf.metrics.recall_at_k(labels, pred_labels_one_hot, 1)
        #     r_at_3 = tf.metrics.recall_at_k(labels, pred_labels_one_hot, 3)
        #     r_at_5 = tf.metrics.recall_at_k(labels, pred_labels_one_hot, 5)

        global_step = tf.train.get_global_step()

        metrics = {
            'accuracy': acc_op
            # 'precision': precision,
            # 'recall': recall,
            # 'precision_at_1': p_at_1,
            # 'precision_at_3': p_at_3,
            # 'precision_at_5': p_at_5,
            # 'recall_at_1': r_at_1,
            # 'recall_at_3': r_at_3,
            # 'recall_at_5': r_at_5,
        }

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        # training mode
        assert mode == tf.estimator.ModeKeys.TRAIN

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(
                params['lr'], params['beta1'],
                params['beta2'], params['epsilon'])
            train_op = optimizer.minimize(loss, global_step=global_step)

        train_log_hook = tf.train.LoggingTensorHook(
            {
                'epoch': global_step // self.num_steps_per_epoch,
                'acc': acc_op[1]
            },
            every_n_iter=100)

        return tf.estimator.EstimatorSpec(
            mode, predictions=logits, loss=loss, train_op=train_op,
            training_hooks=[train_log_hook])

    def train(self, docs, labels_lm, labels, max_step=None):
        self.n = len(labels)
        self.num_steps_per_epoch = (self.n - 1) // self.batch_size + 1

        self.estimator.train(lambda: self.train_input_fn(docs, labels_lm, labels),
                             max_steps=max_step)

    def predict(self, docs):
        pred_labels = []
        doc_vecs = np.zeros((len(docs), self.doc_vec_length))

        self.encoder.dropout = 0.0  # change dropout rate to 0 for predict

        predictions = self.estimator.predict(
            lambda: self.predict_input_fn(docs), yield_single_examples=False)
        for i, pred_dict in enumerate(predictions):
            if i % 100 == 0:
                print("The {}th batch:".format(i))
            start = i * self.batch_size
            end = start + self.batch_size
            doc_vecs[start: end] = pred_dict['doc_vecs']
            pred_labels.extend(pred_dict['pred_labels'])
        return doc_vecs, pred_labels

    def train_and_eval(self, docs_train, labels_lm_train, labels_train,
                       docs_eval, labels_lm_eval, labels_eval, max_step=None):
        self.n = len(labels_train)
        self.num_steps_per_epoch = (self.n - 1) // self.batch_size + 1

        eval_spec = tf.estimator.EvalSpec(
            lambda: self.eval_input_fn(docs_eval, labels_lm_eval, labels_eval),
            steps=None,
            throttle_secs=150
        )

        os.makedirs(os.path.join(self.estimator.model_dir, 'eval'), exist_ok=True)

        early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
            self.estimator,
            metric_name='loss',
            max_steps_without_decrease=self.num_steps_per_epoch,
            min_steps=1000,
            run_every_secs=None,
            run_every_steps=self.num_steps_per_epoch
        )

        train_spec = tf.estimator.TrainSpec(
            lambda: self.train_input_fn(docs_train, labels_lm_train, labels_train),
            hooks=[early_stopping],
            max_steps=max_step
        )

        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)
