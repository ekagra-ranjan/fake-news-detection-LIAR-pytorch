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
          lr,
          epoch,
          model,
          num_classes,
          use_cuda,
          word2num,
          hyper,
          nnArchitecture,
          timestampLaunch):


    train_data = train_samples
    train_data = dataset_to_variable(train_data, use_cuda)
    valid_data = valid_samples
    valid_data = dataset_to_variable(valid_data, use_cuda)

    # model cuda
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    model.to(device)

    # Start training
    print('\n  Start training')

    optimizer = optim.Adam(model.parameters(), lr = lr)

    step = 0
    val_acc = 0
    display_interval = 500
    tick = time.time()

    for epoch_ in range(epoch):
        
        model.train()
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

            # if step %100 == 0:
            #     break

        print('  [INFO] --- Epoch '+str(epoch_+1)+' complete. Avg. Loss: {:.3f}'.format(total_loss/len(train_data)) + '  Time taken: {:.3f}' .format(time.time()-tick) )
        val_acc = valid(valid_data, model)

        modelName = 'm-' + nnArchitecture + '-num_classes-'+ str(num_classes) + '-' + str(timestampLaunch) + '-epoch-' + str(epoch_) + '-val_acc-{:.3f}'.format(val_acc) + '.pth.tar'
        torch.save({'state_dict': model.state_dict(), 'word2num': word2num, 'hyper': hyper}, './models/' + modelName)
        print("Saved: ", modelName)
        


    return model, val_acc







def valid(valid_samples, model):

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

    return acc
