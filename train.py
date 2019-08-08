import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
from model import Net
from data import dataset_to_variable

def train(train_samples,
          valid_samples,
          word2num,
          lr,
          epoch,
          num_classes,
          use_cuda):

    print('Training...')

    # Prepare training data
    print('  Preparing training data...')
    statement_word2num = word2num[0]
    subject_word2num = word2num[1]
    speaker_word2num = word2num[2]
    speaker_pos_word2num = word2num[3]
    state_word2num = word2num[4]
    party_word2num = word2num[5]
    context_word2num = word2num[6]

    train_data = train_samples
    train_data = dataset_to_variable(train_data, use_cuda)
    valid_data = valid_samples
    valid_data = dataset_to_variable(valid_data, use_cuda)

    # Construct model instance
    print('  Constructing network model...')
    model = Net(len(statement_word2num),
                len(subject_word2num),
                len(speaker_word2num),
                len(speaker_pos_word2num),
                len(state_word2num),
                len(party_word2num),
                len(context_word2num),
                num_classes)
    if use_cuda: device = torch.device('cuda')
    else: device = torch.device('cpu')
    model.to(device)

    # Start training
    print('  Start training')

    optimizer = optim.Adam(model.parameters(), lr = lr)
    model.train()

    step = 0
    display_interval = 500
    tick = time.time()

    for epoch_ in range(epoch):
        random.shuffle(train_data)
        total_loss = 0
        for sample in train_data:

            optimizer.zero_grad()

            # import pdb; pdb.set_trace()
            prediction = model(sample)
            label = Variable(torch.LongTensor([sample.label])).to(device)
            # loss = F.cross_entropy(prediction, label)
            # print("prediction:", prediction, " label:", label)
            loss = F.nll_loss(prediction, label)
            loss.backward()
            optimizer.step()

            step += 1
            if step % display_interval == 0:
                print('  [INFO] - Epoch '+ str(epoch_+1) + '/'+ str(epoch) + ' Step:: '+str(step)+' Loss: {:.3f}'.format(loss.data.item()))

            total_loss += loss.cpu().data.numpy()

        print('  [INFO] --- Epoch '+str(epoch_+1)+' complete. Avg. Loss: {:.3f}'.format(total_loss/len(train_data)) + '  Time taken: {:.3f}' .format(time.time()-tick) )

        valid(valid_data, word2num, model)

    return model







def valid(valid_samples, word2num, model):

    model.eval()
    
    acc = 0
    for sample in valid_samples:
        prediction = model(sample)
        # import pdb; pdb.set_trace()
        prediction = int(np.argmax(prediction.cpu().data.numpy()))
        if prediction == sample.label:
            acc += 1
    acc /= len(valid_samples)
    print('  Validation Accuracy: {:.3f}'.format(acc))
