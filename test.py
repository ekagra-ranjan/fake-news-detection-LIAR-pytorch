import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import DataSample, dataset_to_variable, test_data_prepare
import numpy as np

num_to_label_6_way_classification = [
    'pants-fire',
    'false',
    'barely-true',
    'half-true',
    'mostly-true',
    'true'
]

num_to_label_2_way_classification = ['false', 'true']

def test(test_samples, test_output, model, classification_type, use_cuda = False):

    model.eval()

    test_samples = dataset_to_variable(test_samples, use_cuda)
    out = open(test_output, 'w', buffering=1)
    acc = 0
    
    for sample in test_samples:
        prediction = model(sample)
        prediction = int(np.argmax(prediction.cpu().data.numpy()))
        #---choose 6 way or binary classification 
        if classification_type == 2:
            out.write(num_to_label_2_way_classification[prediction]+'\n')
        else:
            out.write(num_to_label_6_way_classification[prediction]+'\n')

        if prediction == sample.label:
            acc += 1
    acc /= len(test_samples)
    print('  Test Accuracy: {:.3f}'.format(acc))
    out.close()

    return acc

