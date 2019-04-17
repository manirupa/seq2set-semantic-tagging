import tensorflow as tf
import numpy as np


def _get_sparse_representation(dataset, term_size):
    return dataset.map(
        lambda y_batched: tf.reduce_sum(
            tf.one_hot(y_batched, term_size), axis=1))


def _get_sparse_representation_softmax_skewed(dataset, term_size):
    def convert_to_skewed_labels(y_batched):
        """ helper function for get_sparse_representation """

        y_batched_new = np.zeros(
            (len(y_batched), term_size), dtype=np.float32)
        count = 0
        for i, yi in enumerate(y_batched):
            for y_ij in yi:
                if y_ij > -1:
                    count += 1
                else:
                    break
            if count == 1:
                y_batched_new[i][y_batched[0]] = 1
            elif count == 2:
                y_batched_new[i][y_batched[0]] = 0.45
                y_batched_new[i][y_batched[1]] = 0.55
            else:
                y_batched_new[i][y_batched[0]] = 0.325
                y_batched_new[i][y_batched[1]] = 0.33
                y_batched_new[i][y_batched[2]] = 0.345
        return y_batched_new

    return dataset.map(
        lambda y_batched: tf.py_func(
            convert_to_skewed_labels, [y_batched], tf.float32))


def _get_pred_labels_sigmoid(logits):
    prob = tf.nn.sigmoid(logits)
    pred_labels_one_hot = tf.round(prob)  # 0 if < 0.5, else 1
    _, pred_labels = tf.nn.top_k(prob, k=5)
    return pred_labels_one_hot, pred_labels


def _get_pred_labels_softmax(logits, term_size):
    prob = tf.nn.softmax(logits)
    _, pred_labels = tf.nn.top_k(prob, k=5)
    pred_labels_one_hot = tf.reduce_sum(
        tf.one_hot(pred_labels, term_size), axis=1)
    return pred_labels_one_hot, pred_labels


def _get_loss_sigmoid(logits, labels):
    loss_op = tf.losses.sigmoid_cross_entropy(
        labels, logits, reduction=tf.losses.Reduction.MEAN)
    return loss_op


def _get_loss_sigmoid_constraint(logits, labels, pred_labels_one_hot):
    output_sum = tf.count_nonzero(
        pred_labels_one_hot, 1, dtype=tf.float32)
    loss_output_sum = tf.square(tf.subtract(output_sum, 3.0))
    cross_entropy = tf.losses.sigmoid_cross_entropy(labels, logits)
    losses = cross_entropy + loss_output_sum
    loss_op = tf.reduce_mean(losses)
    return loss_op


def _get_loss_lm(labels_lm, doc_vecs):
    """
    :param labels_lm: a tensor of shape [batch_size, max_num_labels, embeddings_dim]
    :param doc_vecs: a tensor of shape
    :return: a scalar tensor
    """

    max_num_labels = tf.shape(labels_lm)[1]

    # expand dim of doc to [batch_size, 1, doc_vec_size]
    b1 = tf.expand_dims(doc_vecs, 1)

    # repeat the second axis max_number_of_labels times
    # get shape [batch_size, max_number_of_labels, embeddings_dim]
    b2 = tf.tile(b1, [1, max_num_labels, 1])

    product = tf.multiply(labels_lm, b2)  # todo mask out the padded zeros

    sum_prod = tf.reduce_sum(product, 2)  # get the scalar product of labels per doc with doc

    sig = tf.sigmoid(sum_prod)  # transform to sigmoid

    log = -1.0 * tf.log(sig)  # log sigmoid

    log_sum = tf.reduce_sum(log, 1)

    mean_loss = tf.reduce_mean(log_sum, 0)

    return mean_loss


def _get_loss_softmax(logits, labels):
    divisor = tf.reduce_sum(labels, 1)
    divisor_reshaped = tf.reshape(divisor, [-1, 1])
    normalized_labels = labels / divisor_reshaped
    loss_op = tf.losses.softmax_cross_entropy(
        normalized_labels, logits,
        reduction=tf.losses.Reduction.MEAN)
    return loss_op


def _get_loss_softmax_skewed(logits, labels):
    loss_op = tf.losses.softmax_cross_entropy(
        labels, logits, reduction=tf.losses.Reduction.MEAN)
    return loss_op


class LossesFactory:
    @staticmethod
    def get_loss_fn(params):
        if params['loss_fn'] == "sigmoid":
            return _Sigmoid()
        elif params['loss_fn'] == "lm":
            return _SigmoidLM()
        elif params['loss_fn'] == "sigmoid_with_constraint":
            return _SigmoidConstraint()
        elif params['loss_fn'] == "softmax_uniform":
            return _Softmax()
        elif params['loss_fn'] == "softmax_skewed_labels":
            return _SoftmaxSkewed()
        else:
            raise ValueError('Invalid loss function name')


class _LossFn:
    @NotImplementedError
    def get_sparse_representation(self, dataset, term_size):
        pass

    @NotImplementedError
    def get_pred_labels(self, logits, term_size):
        pass

    @NotImplementedError
    def get_loss(self, logits, labels_lm, labels, pred_labels_one_hot, doc_vecs, alpha):
        pass


class _Sigmoid(_LossFn):
    def get_sparse_representation(self, dataset, term_size):
        return _get_sparse_representation(dataset, term_size)

    def get_pred_labels(self, logits, term_size):
        return _get_pred_labels_sigmoid(logits)

    def get_loss(self, logits, labels_lm, labels, pred_labels_one_hot, doc_vecs, alpha):
        return _get_loss_sigmoid(logits, labels)


class _SigmoidLM(_Sigmoid):
    def get_loss(self, logits, labels_lm, labels, pred_labels_one_hot, doc_vecs, alpha):
        return alpha * _get_loss_lm(labels_lm, doc_vecs) + _get_loss_sigmoid(logits, labels)


class _SigmoidConstraint(_Sigmoid):
    def get_loss(self, logits, labels_lm, labels, pred_labels_one_hot, doc_vecs, alpha):
        return _get_loss_sigmoid_constraint(logits, labels, pred_labels_one_hot)


class _Softmax(_LossFn):
    def get_sparse_representation(self, dataset, term_size):
        return _get_sparse_representation(dataset, term_size)

    def get_pred_labels(self, logits, term_size):
        return _get_pred_labels_softmax(logits, term_size)

    def get_loss(self, logits, labels_lm, labels, pred_labels_one_hot, doc_vecs, alpha):
        return _get_loss_softmax(logits, labels)


class _SoftmaxSkewed(_Softmax):
    def get_sparse_representation(self, dataset, term_size):
        return _get_sparse_representation_softmax_skewed(dataset, term_size)

    def get_loss(self, logits, labels_lm, labels, pred_labels_one_hot, doc_vecs, alpha):
        return _get_loss_softmax_skewed(logits, labels)
