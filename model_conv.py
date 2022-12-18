import numpy as np
import nussl
import os
import tempfile
import torch

from nussl.ml.train.closures import Closure
from nussl.ml.train import BackwardsEvents
from tqdm import tqdm

# Model modules
num_features = 101  
num_sources = 4  # how many sources to estimate
mask_activation = 'sigmoid'  # activation function for masks
num_audio_channels = 1  # number of audio channels

modules = {
    'mix_magnitude': {},
    'my_norm': {
        'class': 'BatchNorm',
    },
    'my_conv': {
        'class': 'ConvolutionalStack2D',
        'args': {
            'in_channels': 1,
            'channels': [3, 1],
            'dilations': [1, 1],
            'filter_shapes': [3, 3],
            'residuals': [True, True]
        }
    },
    'my_mask': {
        'class': 'Embedding',
        'args': {
            'num_features': num_features,
            'hidden_size': num_features,
            'embedding_size': num_sources,
            'activation': mask_activation,
            'num_audio_channels': num_audio_channels,
            # embed the frequency dimension (2) for all audio channels (3)
            'dim_to_embed': [2, 3]
        }
    },
    'my_estimates': {
        'class': 'Mask',
    },
}

# Model connections
connections = [
    ['my_conv',        ['mix_magnitude', ]],
    ['my_norm',        ['my_conv', ]],
    ['my_mask',        ['my_norm', ]],
    ['my_estimates',   ['my_mask', 'mix_magnitude']]
]

# Outputs
output = ['my_estimates', 'my_mask']

# Config
config = {
    'modules': modules,
    'connections': connections,
    'output': output,
    'name': 'model_conv'
}

# Create model
model = nussl.ml.SeparationModel(config)

# batch size = 1, frames = 400, frequencies = 129, audio channels = 1
mix_magnitude = torch.rand(1, 400, 101, 1)
data = {'mix_magnitude': mix_magnitude}
output = model(data)

# Save model
#with tempfile.NamedTemporaryFile(suffix='.pth', delete=True) as f:
#    loc = model.save(f.name)
#    reloaded_dict = torch.load(f.name)
#
#    print(reloaded_dict.keys())
#
#    new_model = nussl.ml.SeparationModel(reloaded_dict['config'])
#    new_model.load_state_dict(reloaded_dict['state_dict'])
#
#    print(new_model)

class TrainClosure(Closure):
    """
    This closure takes an optimization step on a SeparationModel object given a
    loss.

    Args:
        loss_dictionary (dict): Dictionary containing loss functions and specification.
        optimizer (torch Optimizer): Optimizer to use to train the model.
        model (SeparationModel): The model to be trained.
    """

    def __init__(self, loss_dictionary, optimizer, model):
        super().__init__(loss_dictionary)
        self.optimizer = optimizer
        self.model = model

    def __call__(self, engine, data):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(data)

        loss_ = self.compute_loss(output, data)
        loss_['loss'].backward()
        engine.fire_event(BackwardsEvents.BACKWARDS_COMPLETED)
        self.optimizer.step()
        loss_ = {key: loss_[key].item() for key in loss_}

        return loss_

def format_np(arr):
    return torch.from_numpy(np.expand_dims(np.expand_dims(arr.T, 0), 3)).float()
        

loss_dictionary = {
    'MSELoss': {
        'weight': 1,
        'keys': {
            'my_estimates': 'input',
            'source_magnitudes': 'target',
        }
    }
}

closure = nussl.ml.train.closures.Closure(loss_dictionary)

optimizer = torch.optim.Adam(model.parameters(), lr=.001)

train_closure = nussl.ml.train.closures.TrainClosure(
    loss_dictionary, optimizer, model
)

load_model = True
target_model = './model_570'

if load_model:
    print("Loading model {}".format(target_model))
    reloaded_dict = torch.load(target_model)

    model = nussl.ml.SeparationModel(reloaded_dict['config'])
    model.load_state_dict(reloaded_dict['state_dict'])

num_train = 100
num_test = 50

epochs = 3000
training_loss, test_loss = 0, 0
for epoch in (pbar := tqdm(range(epochs))):
    for i in range(num_train):

        data = format_np(np.load('./data/train_np/{}_0.npy'.format(i)))
        track1 = format_np(np.load('./data/train_np/{}_1.npy'.format(i)))
        track2 = format_np(np.load('./data/train_np/{}_2.npy'.format(i)))
        track3 = format_np(np.load('./data/train_np/{}_3.npy'.format(i)))
        track4 = format_np(np.load('./data/train_np/{}_4.npy'.format(i)))

        source = torch.stack([track1, track2, track3, track4], dim=4)

        item = {'mix_magnitude': data,
        	'source_magnitudes': source}
        loss_output = train_closure(None, item)

        training_loss += loss_output['loss']

    if (epoch + 1) % 100 == 0:
        for j in range(num_test):
            data = format_np(np.load('./data/test_np/{}_0.npy'.format(j)))
            track1 = format_np(np.load('./data/test_np/{}_1.npy'.format(j)))
            track2 = format_np(np.load('./data/test_np/{}_2.npy'.format(j)))
            track3 = format_np(np.load('./data/test_np/{}_3.npy'.format(j)))
            track4 = format_np(np.load('./data/test_np/{}_4.npy'.format(j)))
            source = torch.stack([track1, track2, track3, track4], dim=4)
            item = {'mix_magnitude': data,
        	    'source_magnitudes': source}
            output = model(item)
            loss_output = closure.compute_loss(output, item)

            test_loss += loss_output['loss']
        print("Train Loss: {} Test Loss: {}".format(training_loss / num_train / 100, test_loss / num_test))

        training_loss, test_loss = 0, 0

        model.save('./model_{}'.format(epoch+1))
