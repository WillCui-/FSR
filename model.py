import nussl
import torch

# Model modules
num_features = 129  # number of frequency bins in STFT
num_sources = 3  # how many sources to estimate
mask_activation = 'sigmoid'  # activation function for masks
num_audio_channels = 1  # number of audio channels

modules = {
    'mix_magnitude': {},
    'my_log_spec': {
        'class': 'AmplitudeToDB'
    },
    'my_norm': {
        'class': 'BatchNorm',
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
    ['my_log_spec',    ['mix_magnitude', ]],
    ['my_norm',        ['my_log_spec', ]],
    ['my_mask',        ['my_norm', ]],
    ['my_estimates',   ['my_mask', 'mix_magnitude']]
]

# Outputs
output = ['my_estimates', 'my_mask']

# Config
config = {
    'modules': modules,
    'connections': connections,
    'output': output
}

model = nussl.ml.SeparationModel(config)
print(model)

# batch size = 1, frames = 400, frequencies = 129, audio channels = 1
mix_magnitude = torch.rand(1, 400, 129, 1)
data = {'mix_magnitude': mix_magnitude}
output = model(data)
