from logging import Logger

import torch
import torch.nn as nn
import copy

from NeuralSolvers.models.waveact import WaveAct



def set_seed(seed=2342):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

'''
The following code belongs to the PINNsFormer repository:
https://github.com/AdityaLab/pinnsformer
'''

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__()
        self.linear = nn.Sequential(*[
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model)
        ])

    def forward(self, x):
        return self.linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x):
        x2 = self.act1(x)
        # pdb.set_trace()
        x = x + self.attn(x2, x2, x2)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x, e_outputs):
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x, e_outputs):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.act(x)

class PINNsFormer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden,  lb, ub, nhead=4, d_model=2, activation=None, normalize=True, device='cpu', dropout=True):
        set_seed(2342)
        super(PINNsFormer, self).__init__()

        if(activation is not None):
            Logger.warning("Warning: Activation is not None. PINNsFormer uses own activation "
                           "WaveAct which is instantiated at every layer.")

        self.linear_emb = nn.Linear(input_size, d_model)

        self.encoder = Encoder(d_model, num_hidden, nhead)
        self.decoder = Decoder(d_model, num_hidden, nhead)
        self.linear_out = nn.Sequential(*[
            nn.Linear(d_model, hidden_size),
            WaveAct(),
            nn.Linear(hidden_size, hidden_size),
            WaveAct(),
            nn.Linear(hidden_size, output_size)
        ])

        '''
        self.init_layers(input_size, output_size, hidden_size,num_hidden)
        self.lb = torch.Tensor(lb).float().to(device)
        self.ub = torch.Tensor(ub).float().to(device)
        self.linear_layers.to(device)
        self.normalize = normalize
        self.device = device
        '''

    def forward(self, x):
        src = self.linear_emb(x)
        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        output = self.linear_out(d_output)
        # pdb.set_trace()
        # raise Exception('stop')
        return output