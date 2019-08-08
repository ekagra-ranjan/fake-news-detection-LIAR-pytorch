import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Net(nn.Module):

    def __init__(self,

                 statement_vocab_dim,
                 subject_vocab_dim,
                 speaker_vocab_dim,
                 speaker_pos_vocab_dim,
                 state_vocab_dim,
                 party_vocab_dim,
                 context_vocab_dim,
                 num_classes,

                 statement_embed_dim = 100,
                 statement_kernel_num = 14,
                 statement_kernel_size = [3, 4, 5],

                 subject_embed_dim = 5,
                 subject_hidden_dim = 5,
                 subject_lstm_nlayers = 2,
                 subject_lstm_bidirectional = True,

                 speaker_embed_dim = 5,

                 speaker_pos_embed_dim = 10,
                 speaker_pos_hidden_dim = 5,
                 speaker_pos_lstm_nlayers = 2,
                 speaker_pos_lstm_bidirectional = True,

                 state_embed_dim = 5,

                 party_embed_dim = 5,

                 context_embed_dim = 20,
                 context_hidden_dim = 6,
                 context_lstm_nlayers = 2,
                 context_lstm_bidirectional = True,
                 dropout = 0.5):

        # Statement CNN
        super(Net, self).__init__()

        # import pdb; pdb.set_trace()
        self.num_classes = num_classes

        self.statement_vocab_dim = statement_vocab_dim
        self.statement_embed_dim = statement_embed_dim
        self.statement_kernel_num = statement_kernel_num
        self.statement_kernel_size = statement_kernel_size

        self.statement_embedding = nn.Embedding(self.statement_vocab_dim, self.statement_embed_dim)
        self.statement_convs = nn.ModuleList()
        # conv layer with in_channel = 1, out_channel = statement_kernel_num 
        # and kernel spatial size (rectangle) = (kernel_, statement_embed_dim)
        for kernel_ in self.statement_kernel_size:
            self.statement_convs.append(nn.Conv2d(1, self.statement_kernel_num, (kernel_, self.statement_embed_dim)))

        # Subject
        self.subject_vocab_dim = subject_vocab_dim
        self.subject_embed_dim = subject_embed_dim
        self.subject_lstm_nlayers = subject_lstm_nlayers
        self.subject_lstm_num_direction = 2 if subject_lstm_bidirectional else 1
        self.subject_hidden_dim = subject_hidden_dim

        self.subject_embedding = nn.Embedding(self.subject_vocab_dim, self.subject_embed_dim)
        self.subject_lstm = nn.LSTM(
            input_size = self.subject_embed_dim,
            hidden_size = self.subject_hidden_dim,
            num_layers = self.subject_lstm_nlayers,
            batch_first = True,
            bidirectional = subject_lstm_bidirectional
        )

        # Speaker
        self.speaker_vocab_dim = speaker_vocab_dim
        self.speaker_embed_dim = speaker_embed_dim

        self.speaker_embedding = nn.Embedding(self.speaker_vocab_dim, self.speaker_embed_dim)

        # Speaker Position
        self.speaker_pos_vocab_dim = speaker_pos_vocab_dim
        self.speaker_pos_embed_dim = speaker_pos_embed_dim
        self.speaker_pos_lstm_nlayers = speaker_pos_lstm_nlayers
        self.speaker_pos_lstm_num_direction = 2 if speaker_pos_lstm_bidirectional else 1
        self.speaker_pos_hidden_dim = speaker_pos_hidden_dim

        self.speaker_pos_embedding = nn.Embedding(self.speaker_pos_vocab_dim, self.speaker_pos_embed_dim)
        self.speaker_pos_lstm = nn.LSTM(
            input_size = self.speaker_pos_embed_dim,
            hidden_size = self.speaker_pos_hidden_dim,
            num_layers = self.speaker_pos_lstm_nlayers,
            batch_first = True,
            bidirectional = speaker_pos_lstm_bidirectional
        )

        # State
        self.state_vocab_dim = state_vocab_dim
        self.state_embed_dim = state_embed_dim

        self.state_embedding = nn.Embedding(self.state_vocab_dim, self.state_embed_dim)

        # Party
        self.party_vocab_dim = party_vocab_dim
        self.party_embed_dim = party_embed_dim

        self.party_embedding = nn.Embedding(self.party_vocab_dim, self.party_embed_dim)

        # Context
        self.context_vocab_dim = context_vocab_dim
        self.context_embed_dim = context_embed_dim
        self.context_lstm_nlayers = context_lstm_nlayers
        self.context_lstm_num_direction = 2 if context_lstm_bidirectional else 1
        self.context_hidden_dim = context_hidden_dim

        self.context_embedding = nn.Embedding(self.context_vocab_dim, self.context_embed_dim)
        self.context_lstm = nn.LSTM(
            input_size = self.context_embed_dim,
            hidden_size = self.context_hidden_dim,
            num_layers = self.context_lstm_nlayers,
            batch_first = True,
            bidirectional = context_lstm_bidirectional
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(self.statement_kernel_size) * self.statement_kernel_num
                            + self.subject_lstm_nlayers * self.subject_lstm_num_direction
                            + self.speaker_embed_dim
                            + self.speaker_pos_lstm_nlayers * self.speaker_pos_lstm_num_direction
                            + self.state_embed_dim
                            + self.party_embed_dim
                            + self.context_lstm_nlayers * self.context_lstm_num_direction,
                            self.num_classes)

    def forward(self, sample):

        
        statement = Variable(sample.statement).unsqueeze(0)
        subject = Variable(sample.subject).unsqueeze(0)
        speaker = Variable(sample.speaker).unsqueeze(0)
        speaker_pos = Variable(sample.speaker_pos).unsqueeze(0)
        state = Variable(sample.state).unsqueeze(0)
        party = Variable(sample.party).unsqueeze(0)
        context = Variable(sample.context).unsqueeze(0)

        batch = 1 # Current support one sample per time
                  # TODO: Increase batch number

        # Statement
        import pdb; pdb.set_trace()
        statement_ = self.statement_embedding(statement).unsqueeze(0) # 1*W*D -> 1*1*W*D
        statement_ = [F.relu(conv(statement_)).squeeze(3) for conv in self.statement_convs] # 1*1*W*1 -> 1*Conv-filters*(W-1) x len(convs)
        statement_ = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in statement_] # 1*Co*1 -> 1*Conv-filters x len(convs)
        statement_ = torch.cat(statement_, 1)  # 1*len(convs)

        # Subject
        subject_ = self.subject_embedding(subject) # 1*W*D
        _, (subject_, _) = self.subject_lstm(subject_) # (layer x dir) * batch * hidden
        subject_ = F.max_pool1d(subject_, self.subject_hidden_dim).view(1, -1) # (layer x dir) * batch * 1 -> 1*(layer x dir)

        # Speaker
        speaker_ = self.speaker_embedding(speaker).squeeze(0) # 1*1*D -> 1*D

        # Speaker Position
        speaker_pos_ = self.speaker_pos_embedding(speaker_pos)
        _, (speaker_pos_, _) = self.speaker_pos_lstm(speaker_pos_)
        speaker_pos_ = F.max_pool1d(speaker_pos_, self.speaker_pos_hidden_dim).view(1, -1)

        # State
        state_ = self.state_embedding(state).squeeze(0)

        # Party
        party_ = self.party_embedding(party).squeeze(0)

        # Context
        context_ = self.context_embedding(context)
        _, (context_, _) = self.context_lstm(context_)
        context_ = F.max_pool1d(context_, self.context_hidden_dim).view(1, -1)

        # Concatenate
        features = torch.cat((statement_, subject_, speaker_, speaker_pos_, state_, party_, context_), 1)
        features = self.dropout(features)
        out = self.fc(features)
        out = F.log_softmax(out, dim=-1)

        return out
