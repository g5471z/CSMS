"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from c2nl.modules.util_class import LayerNorm
from c2nl.modules.multi_head_attn import MultiHeadedAttention
from c2nl.modules.position_ffn import PositionwiseFeedForward
from c2nl.encoders.encoder import EncoderBase
from c2nl.utils.misc import sequence_mask


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self,
                 d_model,
                 heads,
                 d_ff,
                 d_k,
                 d_v,
                 dropout,
                 max_relative_positions=0,
                 use_neg_dist=True):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = MultiHeadedAttention(heads,
                                              d_model,
                                              d_k,
                                              d_v,
                                              dropout=dropout,
                                              max_relative_positions=max_relative_positions,
                                              use_neg_dist=use_neg_dist)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        
#         self.attention_1 = MultiHeadedAttention(heads,
#                                               d_model,
#                                               d_k,
#                                               d_v,
#                                               dropout=dropout,
#                                               max_relative_positions=max_relative_positions,
#                                               use_neg_dist=use_neg_dist)

#         self.dropout_1 = nn.Dropout(dropout)
#         self.layer_norm_1 = LayerNorm(d_model)
        
#         self.code_attention = MultiHeadedAttention(heads,
#                                               d_model,
#                                               d_k,
#                                               d_v,
#                                               dropout=dropout,
#                                               max_relative_positions=max_relative_positions,
#                                               use_neg_dist=use_neg_dist)

#         self.code_dropout = nn.Dropout(dropout)
#         self.code_layer_norm = LayerNorm(d_model)
        
#         self.ast_attention = MultiHeadedAttention(heads,
#                                               d_model,
#                                               d_k,
#                                               d_v,
#                                               dropout=dropout,
#                                               max_relative_positions=max_relative_positions,
#                                               use_neg_dist=use_neg_dist)

#         self.ast_dropout = nn.Dropout(dropout)
#         self.ast_layer_norm = LayerNorm(d_model)
        
        self.code_feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
#         self.ast_feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, mask, inputs1=None):
        """
        Transformer Encoder Layer definition.
        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        # self-code
        if inputs1 is None:
            context, attn_per_head, _ = self.attention(inputs, inputs, inputs,
                                                       mask=mask, attn_type="self")
            out = self.layer_norm(self.dropout(context) + inputs)
        else:
            context, attn_per_head, _ = self.attention(inputs, inputs, inputs1,
                                                       mask=mask, attn_type="self")
            out = self.layer_norm(self.dropout(context) + inputs1)
        # self-ast
#         ast, ast_attn_per_head, _ = self.attention(ast_inputs, ast_inputs, ast_inputs,
#                                                    mask=ast_mask, attn_type="self")
#         ast_out = self.layer_norm(self.dropout(ast) + ast_inputs)
        
#         code_temp = out
#         ast_temp = ast_out
        
#         # code-ast
#         context1, attn_per_head1, _ = self.code_attention(ast_temp, ast_temp, code_temp,
#                                                    mask=ast_mask, attn_type="self")
#         out1 = self.code_layer_norm(self.code_dropout(context1) + code_temp)
        
#         # ast-code
#         ast1, ast_attn_per_head1, _ = self.ast_attention(code_temp, code_temp, ast_temp,
#                                                    mask=mask, attn_type="self")
#         ast_out1 = self.ast_layer_norm(self.ast_dropout(ast1) + ast_temp)
        
#         return self.code_feed_forward(out1), attn_per_head1, self.ast_feed_forward(ast_out1), ast_attn_per_head1
        return self.code_feed_forward(out), attn_per_head


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".
    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    Returns:
        (`FloatTensor`, `FloatTensor`):
        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self,
                 num_layers,
                 d_model=512,
                 heads=8,
                 d_k=64,
                 d_v=64,
                 d_ff=2048,
                 dropout=0.2,
                 max_relative_positions=0,
                 use_neg_dist=True):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        if isinstance(max_relative_positions, int):
            max_relative_positions = [max_relative_positions] * self.num_layers
        assert len(max_relative_positions) == self.num_layers

        self.layer = nn.ModuleList(
            [TransformerEncoderLayer(d_model,
                                     heads,
                                     d_ff,
                                     d_k,
                                     d_v,
                                     dropout,
                                     max_relative_positions=max_relative_positions[i],
                                     use_neg_dist=use_neg_dist)
             for i in range(num_layers)])

    def count_parameters(self):
        params = list(self.layer.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(self, src, lengths=None, src1=None):
        """
        Args:
            src (`FloatTensor`): `[batch_size x src_len x model_dim]`
            lengths (`LongTensor`): length of each sequence `[batch]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        self._check_args(src, lengths)
#         self._check_args(ast, ast_len)

        out = src
#         ast_out = ast
        mask = None if lengths is None else \
            ~sequence_mask(lengths, out.shape[1]).unsqueeze(1)
#         ast_mask = None if ast_len is None else \
#             ~sequence_mask(ast_len, ast_out.shape[1]).unsqueeze(1)
        # Run the forward pass of every layer of the tranformer.
        if src1 is not None:
            out1 = src1
            mask1 = None if lengths is None else \
            ~sequence_mask(lengths, out1.shape[1]).unsqueeze(1)
        
        representations = []
        attention_scores = []
#         ast_representations = []
#         ast_attention_scores = []
        for i in range(self.num_layers):
            if src1 is None:
                out, attn_per_head = self.layer[i](out, mask)
                representations.append(out)
                attention_scores.append(attn_per_head)
            else:
                out, attn_per_head = self.layer[i](out1, mask1, out)
                representations.append(out)
                attention_scores.append(attn_per_head)
#             ast_representations.append(ast_out)
#             ast_attention_scores.append(ast_attn_per_head)
#         print("out:", out.shape)
        return representations, attention_scores
