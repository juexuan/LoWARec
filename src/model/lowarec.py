import copy
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward, MultiHeadAttention, AttentionLayer
from einops import repeat

class LoWARec(SequentialRecModel):
    def __init__(self, args):
        super(LoWARec, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = LoWARecEncoder(args)
        self.apply(self.init_weights)
        

    def forward(self, input_ids, user_ids=None, all_sequence_output=False): # 注意修改为了True
        # 256 * 1 * 50 * 50
        extended_attention_mask = self.get_attention_mask(input_ids)
        # (batch, max_len, hidden) 256 * 50 * 64
        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )               
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]


        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]

        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)
        return loss

class LoWARecEncoder(nn.Module):
    def __init__(self, args):
        super(LoWARecEncoder, self).__init__()
        self.args = args
        block = LoWARecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers

class LoWARecBlock(nn.Module):
    def __init__(self, args):
        super(LoWARecBlock, self).__init__()
        self.layer = LoWARecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class LoWARecLayer(nn.Module):
    def __init__(self, args):
        super(LoWARecLayer, self).__init__()
        self.args = args

        self.filter_layer = FourierWaveletFusionLayer(args)

        self.alpha = args.alpha
        self.projected_layer = DualStageAttention(seq_len = 1, dim_proj = args.dim_proj, d_model = 64, n_heads = args.proj_n_heads, d_ff=None, dropout=args.proj_dropout)

    def forward(self, input_tensor, attention_mask):
        # (batch, max_len, hidden)
        dsp = self.filter_layer(input_tensor)
        pal = self.projected_layer(input_tensor)
        # gsp = self.attention_layer(input_tensor, attention_mask)
        hidden_states = self.alpha * dsp + ( 1 - self.alpha ) * pal

        return hidden_states



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

# Frequency-Aware Bias
class FourierWaveletFusionLayer(nn.Module):
    def __init__(self, args, wavelet_kernel_size=3):
        super(FourierWaveletFusionLayer, self).__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)

        self.c = args.c // 2 + 1
        self.d_model = args.hidden_size


        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.alpha_act = nn.Sigmoid()

        self.wavelet_conv = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=wavelet_kernel_size,
            padding=wavelet_kernel_size // 2,
            groups=self.d_model,
            bias=False
        )

        with torch.no_grad():
            for i in range(self.d_model):
                self.wavelet_conv.weight[i][0] = torch.tensor([1, -1, 0.0][:wavelet_kernel_size])

    def forward(self, input_tensor):
        # input_tensor: [batch, seq_len, d_model]
        batch, seq_len, hidden = input_tensor.shape


        x_fft = torch.fft.rfft(input_tensor, dim=1, norm='ortho')  # [B, F, H]
        low_fft = x_fft.clone()
        low_fft[:, self.c:, :] = 0
        low_part = torch.fft.irfft(low_fft, n=seq_len, dim=1, norm='ortho')  # [B, L, H]


        x_wavelet = input_tensor.permute(0, 2, 1)  # [B, H, L]
        high_part = self.wavelet_conv(x_wavelet).permute(0, 2, 1)  # [B, L, H]


        alpha = self.alpha_act(self.alpha)
        fused = alpha * low_part + (1 - alpha) * high_part


        output = self.out_dropout(fused)
        output = self.LayerNorm(output + input_tensor)

        return output


# Dual-Stage Attention
class DualStageAttention(nn.Module):
    """
    Temporal projected attention layer
    """
    def __init__(self, seq_len, dim_proj, d_model, n_heads, d_ff=None, dropout=0.1):
        super(DualStageAttention, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.out_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.in_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.projector = nn.Parameter(torch.randn(dim_proj, d_model))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))

        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, x):
        # x: [b s n d]:[batch, seq_len, num_items, hidden_dim]
        new_x_shape = x.size()[:-1] + (1, self.d_model)
        x = x.view(*new_x_shape)
        x = x.permute(0, 2, 1, 3)
        # print("x.shape:{}".format(x.shape))
        batch = x.shape[0]
        projector = repeat(self.projector, 'dim_proj d_model -> repeat seq_len dim_proj d_model',
                              repeat=batch, seq_len=self.seq_len)  # [b, s, c, d]

        # Item-to-Interest Aggregation Attention
        message_out = self.out_attn(projector, x, x)  # [b, s, c, d] <-> [b s n d] -> [b s c d]
        # Interest-Context Interaction Attention
        message_in = self.in_attn(x, projector, message_out)  # [b s n d] <-> [b, s, c, d] -> [b s n d]

        message = x + self.dropout(message_in)
        message = message.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = message.size()[:-2] + (self.d_model,)
        # context_layer.shape:(batch, max_len, hidden)
        message = message.view(*new_context_layer_shape)
        # print("message.shape:{}".format(message.shape))
        message = self.norm1(message)
        message = message + self.dropout(self.MLP(message))
        message = self.norm2(message)

        return message




