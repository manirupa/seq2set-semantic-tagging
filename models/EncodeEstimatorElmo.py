import numpy as np
import tensorflow as tf

from models.Encoder import EncoderFactory
from models.Losses import LossesFactory


class EncodeEstimator:
    def __init__(self, params):
        self.max_length = params['max_length']
        self.term_size = params['term_size']
        self.doc_vec_length = params['doc_vec_length']
        self.embedding_dim = params['embeddings_dim']
        self.pred_data_size = params['pred_data_size']
        self.num_epochs = params['num_epochs']
        self.batch_size = params['batch_size']

        # instance attribute declaration
        self.n = None
        self.num_steps_per_epoch = None

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

    def train_input_fn(self, sen_gen, len_gen, labels):
        dataset = self.eval_input_fn(sen_gen, len_gen, labels)
        dataset = dataset.shuffle(self.n).repeat(self.num_epochs)
        return dataset

    def eval_input_fn(self, sen_gen, len_gen, labels):
        # todo: add unknown words to word vectors

        dataset_docs = tf.data.Dataset.from_generator(sen_gen, tf.float32)
        dataset_len = tf.data.Dataset.from_generator(len_gen, tf.int32)
        dataset_labels = tf.data.Dataset.from_generator(lambda: labels, tf.int32)

        # "dynamic padding": each batch might have different
        # sequence lengths but the sequence lengths within
        # one batch are the same the data type of padding_values
        # should match that of the dataset! both tf.int32 in this case
        dataset_docs = dataset_docs.padded_batch(
            self.batch_size,
            tf.TensorShape([None, 100]))

        dataset_len = dataset_len.batch(self.batch_size)

        dataset_labels = dataset_labels.padded_batch(
            self.batch_size, [-1], padding_values=-1)

        dataset_labels = self.loss_fn.get_sparse_representation(
            dataset_labels, self.term_size)

        # zip up dataset
        dataset_feature = tf.data.Dataset.zip(
            (dataset_docs, dataset_len))

        dataset_feature_dict = dataset_feature.map(
            lambda sen, sen_len: {'sentence': sen, 'length': sen_len})

        dataset = tf.data.Dataset.zip((dataset_feature_dict, dataset_labels))
        return dataset

    def predict_input_fn(self, sen_gen, len_gen):
        dataset_docs = tf.data.Dataset.from_generator(sen_gen, tf.float32)
        dataset_len = tf.data.Dataset.from_generator(len_gen, tf.int32)

        # pad and batch
        dataset_docs = dataset_docs.padded_batch(
            self.batch_size, tf.TensorShape([None, 100]))
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

        x = features['sentence']
        doc_vecs, logits = self.encoder.encode(x, features['length'])

        with tf.name_scope("get_predicted_labels"):
            pred_labels_one_hot, pred_labels = self.loss_fn.get_pred_labels(logits, self.term_size)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'doc_vecs': doc_vecs, 'pred_labels': pred_labels}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        with tf.name_scope("cross_entropy"):
            loss = self.loss_fn.get_loss(
                logits, labels, pred_labels_one_hot)

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

    def train(self, sen_gen, len_gen, labels, max_step=None):
        self.n = len(labels)
        self.num_steps_per_epoch = (self.n - 1) // self.batch_size + 1

        self.estimator.train(lambda: self.train_input_fn(sen_gen, len_gen, labels),
                             max_steps=max_step)

    def predict(self, sen_gen, len_gen):
        pred_labels = []
        doc_vecs = np.zeros((self.pred_data_size, self.doc_vec_length))

        self.encoder.dropout = 0.0  # change dropout rate to 0 for predict

        predictions = self.estimator.predict(
            lambda: self.predict_input_fn(sen_gen, len_gen), yield_single_examples=False)
        for i, pred_dict in enumerate(predictions):
            print("The {}th batch:".format(i))
            doc_vecs[i * self.batch_size:(i + 1) * self.batch_size] = pred_dict['doc_vecs']
            pred_labels.extend(pred_dict['pred_labels'])
        return doc_vecs, pred_labels
