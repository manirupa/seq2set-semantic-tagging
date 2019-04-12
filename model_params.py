"""Defines model parameters."""


def get_params(model):
    base_params = {
        # dims of hidden layers, where the last dim equals term_size.
        # There must exists one layer whose dim == doc_vec_length
        'mlp_layer_dims': [50, 1000, 2500, 5000, 9956],
        'doc_vec_length': 50,

        # optimizer params
        'lr': 0.0001,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-9,

        # non-model params
        'max_length': 256,     # used for truncating
        'padding_length': 256  # used for padding
    }

    model_params = {
        'LSTM': {
            **base_params,
            'model': 'LSTM',
            'hidden_size': 200,
            'num_layers': 2
        },

        'GRU': {
            **base_params,
            'model': 'GRU',
            'hidden_size': 200,
            'num_layers': 2
        },

        'BiLSTM': {
            **base_params,
            'model': 'BiLSTM',
            'hidden_size': 200,
            'num_layers': 4
        },

        'BiGRU': {
            **base_params,
            'model': 'BiLSTM',
            'hidden_size': 200,
            'num_layers': 4
        },

        'BiLSTMATT': {
            **base_params,
            'model': 'BiLSTMATT',
            'hidden_size': 200,
            'num_layers': 2,
            'attention_size': 200
        },

        'DAN': {
            **base_params,
            'model': 'DAN'
        },

        'Transformer': {
            **base_params,
            'model': 'Transformer',
            'num_heads': 5,
            'num_layers': 2
        }
    }

    return model_params[model]


# wv settings
settings = {'sg': 1, 'size': 50, 'window': 4, 'min_count': 1, 'negative': 5, 'iter': 15}
