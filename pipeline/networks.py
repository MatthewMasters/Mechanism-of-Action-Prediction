import numpy as np
import torch
from torch import nn
from os.path import join

from pipeline.utils import SETTINGS


def recalibrate_layer(layer):
    if torch.isnan(layer.weight_v).sum() > 0:
        print('recalibrate layer.weight_v')
        layer.weight_v = torch.nn.Parameter(
            torch.where(torch.isnan(layer.weight_v), torch.zeros_like(layer.weight_v), layer.weight_v))
        layer.weight_v = torch.nn.Parameter(layer.weight_v + 1e-7)

    if torch.isnan(layer.weight).sum() > 0:
        print('recalibrate layer.weight')
        layer.weight = torch.where(torch.isnan(layer.weight), torch.zeros_like(layer.weight), layer.weight)
        layer.weight += 1e-7


def initialize_weights(layers, label='all'):
    weights = layers.state_dict()
    last_layer_id = next(reversed(weights)).split('.')[0]
    label_probs = np.load(join(SETTINGS['RAW_DATA_DIR'], 'label_probs.npy'))
    if label != 'all':
        idx = TARGET_LABELS.index(label)
        label_probs = np.expand_dims(label_probs[idx], 0)
    weights[last_layer_id + '.dense.bias'] = torch.tensor(label_probs)
    # weights[last_layer_id + '.weight'] = torch.ones(weights[last_layer_id + '.weight'].shape) * 0.01
    layers.load_state_dict(weights)
    return layers


class DenseModule(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, normalization=None, activation=None, weight_norm=True):
        super(DenseModule, self).__init__()

        self.normalization = None
        if normalization is not None:
            self.normalization = normalization(input_dim)

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(input_dim, output_dim)
        if weight_norm:
            self.dense = nn.utils.weight_norm(self.dense)

        self.activation = None
        if activation is not None:
            self.activation = activation()

    def forward(self, x):
        if self.normalization is not None:
            x = self.normalization(x)

        x = self.dropout(x)

        recalibrate_layer(self.dense)
        x = self.dense(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class MoaDenseNet(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_hidden_layer=2,
                 hidden_layer_size=512,
                 dropout=0.5,
                 activation='prelu',
                 normalization='batch'):
        super(MoaDenseNet, self).__init__()

        if normalization == 'batch':
            normalization = nn.BatchNorm1d
        else:
            Exception('Normalization not supported.')

        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'prelu':
            activation = nn.PReLU
        else:
            Exception('Activation not supported.')

        layers = []
        layers.append(DenseModule(input_dim, hidden_layer_size, dropout/2, normalization, activation))

        for _ in range(n_hidden_layer):
            layers.append(DenseModule(hidden_layer_size, hidden_layer_size, dropout, normalization, activation))

        # Classification layer
        layers.append(DenseModule(hidden_layer_size, output_dim, dropout, normalization, activation=None))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return dict(prediction=self.layers(x['features']))