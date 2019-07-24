import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import hyperparams as hp
from text.symbols import symbols
import numpy as np
import copy
from collections import OrderedDict


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x


class EncoderPrenet(nn.Module):
    """
    Pre-network for Encoder consists of convolution networks.
    """

    def __init__(self, embedding_size, num_hidden):
        super(EncoderPrenet, self).__init__()
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(len(symbols), embedding_size, padding_idx=0)

        self.conv1 = Conv(in_channels=embedding_size,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')

        self.conv2 = Conv(in_channels=num_hidden,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')

        self.conv3 = Conv(in_channels=num_hidden,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')

        self.batch_norm1 = nn.BatchNorm1d(num_hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_hidden)
        self.batch_norm3 = nn.BatchNorm1d(num_hidden)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.projection = Linear(num_hidden, num_hidden)

    def forward(self, input_):
        input_ = self.embed(input_)
        input_ = input_.transpose(1, 2)
        input_ = self.dropout1(self.batch_norm1(
            torch.relu(self.conv1(input_))))
        input_ = self.dropout2(self.batch_norm2(
            torch.relu(self.conv2(input_))))
        input_ = self.dropout3(self.batch_norm3(
            torch.relu(self.conv3(input_))))
        input_ = input_.transpose(1, 2)
        input_ = self.projection(input_)

        return input_


class FFN(nn.Module):
    """
    Positionwise Feed-Forward Network
    """

    def __init__(self, num_hidden):
        """
        :param num_hidden: dimension of hidden 
        """
        super(FFN, self).__init__()
        self.w_1 = Conv(num_hidden, num_hidden * 4,
                        kernel_size=1, w_init='relu')
        self.w_2 = Conv(num_hidden * 4, num_hidden, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, input_):
        # FFN Network
        x = input_.transpose(1, 2)
        x = self.w_2(torch.relu(self.w_1(x)))
        x = x.transpose(1, 2)

        # residual connection
        x = x + input_

        # dropout
        x = self.dropout(x)

        # layer normalization
        x = self.layer_norm(x)

        return x


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self,
                 n_mel_channels=80,
                 postnet_embedding_dim=512,
                 postnet_kernel_size=5,
                 postnet_n_convolutions=5):

        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                Conv(n_mel_channels,
                     postnet_embedding_dim,
                     kernel_size=postnet_kernel_size,
                     stride=1,
                     padding=int((postnet_kernel_size - 1) / 2),
                     dilation=1,
                     w_init='tanh'),

                nn.BatchNorm1d(postnet_embedding_dim))
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    Conv(postnet_embedding_dim,
                         postnet_embedding_dim,
                         kernel_size=postnet_kernel_size,
                         stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1,
                         w_init='tanh'),

                    nn.BatchNorm1d(postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                Conv(postnet_embedding_dim,
                     n_mel_channels,
                     kernel_size=postnet_kernel_size,
                     stride=1,
                     padding=int((postnet_kernel_size - 1) / 2),
                     dilation=1,
                     w_init='linear'),

                nn.BatchNorm1d(n_mel_channels))
        )

    def forward(self, x):
        # x = x.contiguous().transpose(1, 2)

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(
                self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        # x = x.contiguous().transpose(1, 2)
        return x


class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """

    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden 
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query, mask=None, query_mask=None):
        # Get attention score
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        # Masking to ignore padding (key side)
        if mask is not None:
            attn = attn.masked_fill(mask, -2 ** 32 + 1)
            attn = torch.softmax(attn, dim=-1)
        else:
            attn = torch.softmax(attn, dim=-1)

        # Masking to ignore padding (query side)
        if query_mask is not None:
            attn = attn * query_mask

        # Dropout
        attn = self.attn_dropout(attn)

        # Get Context Vector
        result = torch.bmm(attn, value)

        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads 
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm_1 = nn.LayerNorm(num_hidden)

    def forward(self, memory, decoder_input, mask=None, query_mask=None):

        batch_size = memory.size(0)
        seq_k = memory.size(1)
        seq_q = decoder_input.size(1)

        # Repeat masks h times
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(-1).repeat(1, 1, seq_k)
            query_mask = query_mask.repeat(self.h, 1, 1)
        if mask is not None:
            mask = mask.repeat(self.h, 1, 1)

        # Make multihead
        key = self.key(memory).view(batch_size,
                                    seq_k,
                                    self.h,
                                    self.num_hidden_per_attn)
        value = self.value(memory).view(batch_size,
                                        seq_k,
                                        self.h,
                                        self.num_hidden_per_attn)
        query = self.query(decoder_input).view(batch_size,
                                               seq_q,
                                               self.h,
                                               self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1,
                                                        seq_k,
                                                        self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1,
                                                            seq_k,
                                                            self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1,
                                                            seq_q,
                                                            self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(
            key, value, query, mask=mask, query_mask=query_mask)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q,
                             self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(
            batch_size, seq_q, -1)

        # Concatenate context vector with input (most important)
        result = torch.cat([decoder_input, result], dim=-1)

        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + decoder_input

        # Layer normalization
        result = self.layer_norm_1(result)

        return result, attns


class Prenet(nn.Module):
    """
    Prenet before passing through the network
    """

    def __init__(self, input_size, hidden_size, output_size, p=0.5):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(Prenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
            ('fc1', Linear(self.input_size, self.hidden_size)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p)),
            ('fc2', Linear(self.hidden_size, self.output_size)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p)),
        ]))

    def forward(self, input_):

        out = self.layer(input_)

        return out
