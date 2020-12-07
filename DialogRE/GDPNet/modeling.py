

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from soft_dtw_cuda import SoftDTW




def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)            

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTSelfAttention(nn.Module):
    def __init__(self, config):
        super(BERTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BERTSelfOutput(nn.Module):
    def __init__(self, config):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTAttention(nn.Module):
    def __init__(self, config):
        super(BERTAttention, self).__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BERTIntermediate(nn.Module):
    def __init__(self, config):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    def __init__(self, config):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BERTEncoder(nn.Module):
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        layer = BERTLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])    

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BERTPooler(nn.Module):
    def __init__(self, config):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        #extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return all_encoder_layers, pooled_output


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, args, mem_dim, layers):
        super(GraphConvLayer, self).__init__()
        self.args = args
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.linear_output = self.linear_output.cuda()

    def forward(self, adj, gcn_inputs, src_mask):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):

            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs

        out = self.linear_output(gcn_outputs)

        return adj,out, src_mask

class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, args, mem_dim, layers, heads):
        super(MultiGraphConvLayer, self).__init__()
        self.args = args
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs, src_mask):

        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l

                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)

                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return adj_list,out, src_mask

class GCN_Pool_for_Single(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, args, mem_dim, layers):
        super(GCN_Pool_for_Single, self).__init__()
        self.args = args
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, 1)

        # dcgcn block
        self.weight_list = nn.ModuleList()

        self.weight_list.append(nn.Linear(self.mem_dim, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.linear_output = self.linear_output.cuda()

    def forward(self, adj, gcn_inputs):

        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs


        for l in range(self.layers):

            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            outputs = self.gcn_drop(gAxW)

        gcn_outputs = outputs
        out = self.linear_output(gcn_outputs)

        return out



def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -1e12)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn





def Top_K(score, ratio):
    #batch_size = score.size(0)
    node_sum = score.size(1)
    score = score.view(-1,node_sum)
    K = int(ratio*node_sum)+1
    Top_K_values, Top_K_indices =  score.topk(K, largest=False, sorted=False)
    return Top_K_values, Top_K_indices


class SAGPool_Single(torch.nn.Module):
    def __init__(self,args, ratio=0.5,non_linearity=torch.tanh):
        super(SAGPool_Single,self).__init__()
        #self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = GCN_Pool_for_Single(args, args.graph_hidden_size,1)
        self.non_linearity = non_linearity
    def forward(self, adj, x , src_mask):
        '''if batch is None:
            batch = edge_index.new_zeros(x.size(0))'''

        score = self.score_layer(adj, x)

        _, idx = Top_K(score, self.ratio)

        for i in range(src_mask.size(0)):
            for j in range(idx.size(1)):
                src_mask[i][0][idx[i][j]] = False

        return adj, x, src_mask



class SAGPool_Multi(torch.nn.Module):
    def __init__(self,args, ratio=0.5,non_linearity=torch.tanh, heads = 3):
        super(SAGPool_Multi,self).__init__()
        #self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = GCN_Pool_for_Single(args, args.graph_hidden_size,1)
        self.linear_transform = nn.Linear(args.graph_hidden_size, args.graph_hidden_size//heads)
        self.non_linearity = non_linearity

    def forward(self, adj_list, x, src_mask):
        '''if batch is None:
            batch = edge_index.new_zeros(x.size(0))'''
        # x = x.unsqueeze(-1) if x.dim() == 1 else x
        adj_list_new = []
        src_mask_list = []
        #src_mask_source = src_mask
        x_list = []
        x_select_list = []
        for adj in adj_list:

            score = self.score_layer(adj, x)

            _, idx = Top_K(score, self.ratio)

            x_selected = self.linear_transform(x)
            x_select_list.append(x_selected)



            for i in range(src_mask.size(0)):
                for j in range(idx.size(1)):
                    src_mask[i][0][idx[i][j]] = False
            src_mask_list.append(src_mask)
            adj_list_new.append(adj)
        src_mask_out = torch.zeros_like(src_mask_list[0]).cuda()

        x = torch.cat(x_select_list, dim=2)
        for src_mask_i in src_mask_list:
            src_mask_out = src_mask_out + src_mask_i

        return adj_list_new, x, src_mask_out




class PoolGCN(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.in_dim = config.hidden_size
        self.mem_dim = args.graph_hidden_size
        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)
        self.in_drop = nn.Dropout(args.input_dropout)
        self.num_layers = args.num_graph_layers
        self.layers = nn.ModuleList()
        self.heads = args.heads
        self.sublayer_first = args.sublayer_first
        self.sublayer_second = args.sublayer_second



        # gcn layer
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(MultiGraphConvLayer(args, self.mem_dim, self.sublayer_first, self.heads))
                self.layers.append(SAGPool_Multi(args, ratio=args.pooling_ratio, heads= self.heads))
                self.layers.append(MultiGraphConvLayer(args, self.mem_dim, self.sublayer_second, self.heads))
                self.layers.append(SAGPool_Multi(args, ratio=args.pooling_ratio, heads = self.heads))
            else:
                self.layers.append(MultiGraphConvLayer(args, self.mem_dim, self.sublayer_first, self.heads))
                self.layers.append(SAGPool_Multi(args, ratio=args.pooling_ratio, heads= self.heads))
                self.layers.append(MultiGraphConvLayer(args, self.mem_dim, self.sublayer_second, self.heads))
                self.layers.append(SAGPool_Multi(args, ratio=args.pooling_ratio, heads = self.heads))
        self.agg_nodes_num = int(len(self.layers)//2 * self.mem_dim )
        self.aggregate_W = nn.Linear(self.agg_nodes_num, self.mem_dim)

        self.attn = MultiHeadAttention(self.heads, self.mem_dim)

        self.GGG = GGG(config, args)



    def forward(self, adj, inputs, input_id):

        src_mask = (input_id != 0).unsqueeze(-2)
        src_mask = src_mask[:,:,:adj.size(2)]
        embs = self.in_drop(inputs)
        gcn_inputs = embs
        gcn_inputs = self.input_W_G(gcn_inputs)

        layer_list = []



        gcn_inputs, attn_adj_list = self.GGG(gcn_inputs,adj)
        outputs = gcn_inputs

        for i in range(len(self.layers)):
            if i < 4:

                attn_adj_list, outputs, src_mask = self.layers[i](attn_adj_list, outputs, src_mask)
                if i==0:
                    src_mask_input = src_mask
                if i%2 !=0:
                    layer_list.append(outputs)

            else:
                attn_tensor = self.attn(outputs, outputs, src_mask)
                attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
                attn_adj_list, outputs, src_mask = self.layers[i](attn_adj_list, outputs, src_mask)

                if i%2 !=0:
                    layer_list.append(outputs)


        aggregate_out = torch.cat(layer_list, dim=2)

        dcgcn_output = self.aggregate_W(aggregate_out)

        mask_out = src_mask.reshape([src_mask.size(0),src_mask.size(2),src_mask.size(1)])




        return dcgcn_output, mask_out, layer_list, src_mask_input

def kl_div_gauss(mean_1, mean_2, std_1, std_2):
    kld_element = 0.5*(2*torch.log(std_2) - 2*torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2))/std_2.pow(2) -1)
    return kld_element



class GGG(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.num_heads = args.heads
        self.hid_dim = args.graph_hidden_size
        self.R = self.hid_dim // self.num_heads
        self.max_len_left = args.max_offset
        self.max_len_right = args.max_offset


        self.transform = nn.Linear(self.hid_dim*2, self.hid_dim)
        self.gauss = nn.Linear(self.hid_dim, self.num_heads * 2)

        self.pool_X = nn.MaxPool1d(args.max_offset*2 + 1, stride=1, padding=args.max_offset)
        self.dropout = nn.Dropout(args.input_dropout)


    def forward(self, x, adj, mask = None):
        B, T, C = x.size()
        H = self.num_heads



        x_pooled = self.pool_X(x)
        x_new = torch.cat([x_pooled,x],dim=2)
        x_new = self.transform(x_new)
        x_new = self.dropout(x_new)



        gauss_parameters = self.gauss(x_new)
        gauss_mean, gauss_std = gauss_parameters[:,:,:H], F.softplus(gauss_parameters[:,:,H:])



        kl_div = kl_div_gauss(gauss_mean.unsqueeze(1).repeat(1,T,1,1),gauss_mean.unsqueeze(2).repeat(1,1,T,1),gauss_std.unsqueeze(1).repeat(1,T,1,1),gauss_std.unsqueeze(2).repeat(1,1,T,1))
        adj_multi = kl_div

        attn_adj_list = [attn_adj.squeeze(3) for attn_adj in torch.split(adj_multi, 1, dim=3)]



        return x_new, attn_adj_list


class BertForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels, args):
        super(BertForSequenceClassification, self).__init__()
        self.args = args
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gcn = PoolGCN(config, args)

        self.classifier = nn.Linear(config.hidden_size + args.graph_hidden_size, num_labels * 36)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)
        self.DTW_criterion = SoftDTW(use_cuda=True, gamma=0.1)



    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, n_class=1):
        seq_length = input_ids.size(2)
        attention_mask_ = attention_mask.view(-1,seq_length)

        l = (attention_mask_.data.cpu().numpy() != 0).astype(np.int64).sum(1)

        real_length = max(l)
        word_embedding, pooled_output = self.bert(input_ids.view(-1,seq_length),
                                     token_type_ids.view(-1,seq_length),
                                     attention_mask.view(-1,seq_length))
        adj = torch.ones(input_ids.size(0),real_length,real_length).cuda()
        word_embedding = word_embedding[-1]
        word_embedding = word_embedding[:,:real_length]




        h, pool_mask,layer_list, src_mask_input = self.gcn(adj, word_embedding,input_ids.view(-1,seq_length))

        h_out = pool(h, pool_mask, type="max")

        output = self.dropout(torch.cat([pooled_output,h_out],dim=1))
        #print(output.size())
        logits = self.classifier(output)
        logits = logits.view(-1, 36)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()

            src_mask_input = src_mask_input.reshape([src_mask_input.size(0),src_mask_input.size(2),src_mask_input.size(1)])

            src_mask_input_index = src_mask_input.squeeze(2).sum(0).nonzero()

            src_mask_output_index = pool_mask.squeeze(2).sum(0).nonzero()

            DTW_hidden_0 = torch.index_select(layer_list[0],1,src_mask_input_index.view(-1))
            DTW_hidden_1 = torch.index_select(layer_list[len(layer_list)-1], 1, src_mask_output_index.view(-1))

            loss_dtw = self.DTW_criterion(DTW_hidden_0, DTW_hidden_1).mean()

            labels = labels.view(-1, 36)

            loss = loss_fct(logits, labels) + self.args.lamada*loss_dtw
            return loss, logits
        else:
            return logits

