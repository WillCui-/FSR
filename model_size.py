# wyc2112

'''
Script to print the architecture of a model, as well as the size
'''

import nussl
import torch

model_name = './model_100'

reloaded_dict = torch.load(model_name)

model = nussl.ml.SeparationModel(reloaded_dict['config'])
model.load_state_dict(reloaded_dict['state_dict'])

print(model)
