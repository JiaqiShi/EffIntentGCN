import torch
import torch.nn as nn
import torch.nn.functional as F

from model.graph import Graph
from model.attention import Attention_Layer

class IntentGCN(nn.Module):
    def __init__(self,
                 input_shape,
                 num_class,
                 graph_args=None,
                 edge_importance_weighting=True,
                 **kwargs):
        super().__init__()

        if type(input_shape) == list:
            input_shape = input_shape[0]
        max_T, num_joints, in_channels = input_shape
        self.in_channels = in_channels

        if graph_args is not None:
            self.graph = Graph(num_joints, **graph_args)
        else:
            self.graph = Graph(num_joints)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        self.kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.bn = nn.BatchNorm1d(in_channels * A.size(1))

        self._get_gcn_layers(**kwargs)

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for i in self.blocks])
        else:
            self.edge_importance = [1] * len(self.blocks)

        self.fcn = nn.Linear(self.output_channels[-1], num_class)

    def _get_gcn_layers(self,
                        layer_num,
                        channels,
                        stride_2_layer_index,
                        dropout=0,
                        attention_start_layer=1,
                        **kw):
        def str2intlist(inp_str):
            return [int(s) for s in inp_str.split(',')]

        def get_output_channels(layer_num, channels_list):
            step_num = len(channels_list)
            layer_steps = [
                int(layer_num / step_num) +
                1 if i < layer_num % step_num else int(layer_num / step_num)
                for i in range(step_num)
            ]
            return [
                channels_list[step] for step, layer in enumerate(layer_steps)
                for _ in range(layer)
            ]

        channels_list = str2intlist(channels)
        if layer_num < len(channels_list):
            raise ValueError(
                f'Too many channels given. Expected length larger than {len(channels)}, but got {layer_num}.'
            )
        stride_2_layers = str2intlist(stride_2_layer_index)
        output_channels = get_output_channels(layer_num, channels_list)
        self.output_channels = output_channels
        
        gcn_layer_list = []
        for i in range(layer_num):
            if i in stride_2_layers:
                stride = 2
            else:
                stride = 1
            
            kw0 = {k:(False if k == 'attention' else v) for k,v in kw.items()} if i < attention_start_layer else kw
            if i == 0:
                gcn_layer_list.append(
                    GraphConvBlock(self.in_channels, output_channels[i],
                                   self.kernel_size, stride, **kw0))
            else:
                gcn_layer_list.append(
                    GraphConvBlock(output_channels[i - 1], output_channels[i],
                                   self.kernel_size, stride, dropout, **kw0))
        self.blocks = nn.ModuleList(gcn_layer_list)

    def get_model_name(self):
        return self.__class__.__name__

    def forward(self, x):

        if type(x) == list:
            x, lengths = x
        x = x.float()

        # N, T, V, C -> N, C, T, V
        x = x.permute(0, 3, 1, 2).contiguous()

        N, C, T, V = x.size()

        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        for block, importance in zip(self.blocks, self.edge_importance):
            x = block(x, self.A * importance)

        hiddens = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)

        x = self.fcn(hiddens)

        return x
    
class GraphConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 adap_graph=True,
                 weighted_sum=True,
                 n_head=4,
                 d_kc=0.25,
                 d_vc=0.25,
                 attention=False,
                 att_type='stja',
                 act='relu',
                 use_depthwise_separable_conv_spatial=False,
                 use_low_rank_spatial=False,
                 rank_factor_spatial=0.5,
                 use_expansion_conv_spatial=False,
                 expansion_factor_spatial=2.0,
                 use_low_rank_adap=False,
                 rank_factor_adap=0.5,
                 use_depthwise_separable_conv_t=False,
                 use_low_rank_t=False,
                 rank_factor_t=0.5,
                 use_expansion_conv_t=False,
                 expansion_factor_t=2.0,
                 ):
        super().__init__()

        self.scn = SpatialGraphConv(in_channels,
                                    out_channels,
                                    kernel_size[1],
                                    adap_graph=adap_graph,
                                    weighted_sum=weighted_sum,
                                    n_head=n_head,
                                    d_kc=d_kc,
                                    d_vc=d_vc,
                                    use_depthwise_separable_conv=use_depthwise_separable_conv_spatial,
                                    use_low_rank=use_low_rank_spatial,
                                    rank_factor=rank_factor_spatial,
                                    use_expansion_conv=use_expansion_conv_spatial,
                                    expansion_factor=expansion_factor_spatial,
                                    use_low_rank_adap=use_low_rank_adap,
                                    rank_factor_adap=rank_factor_adap
                                    )

        self.use_depthwise_separable_conv_t = use_depthwise_separable_conv_t
        self.use_low_rank_t = use_low_rank_t

        self.use_expansion_conv_t = use_expansion_conv_t
        self.expansion_factor_t = expansion_factor_t

        expanded_channels = int(out_channels * expansion_factor_t) if self.use_expansion_conv_t else out_channels

        if self.use_depthwise_separable_conv_t and self.use_low_rank_t:
            rank = max(int(expanded_channels * rank_factor_t), 2)
            conv_layers = [
                nn.Conv2d(expanded_channels, rank, (1, 1), stride=(1, 1)),
                nn.Conv2d(rank, rank, (kernel_size[0], 1), stride=(stride, 1), groups=rank, 
                          padding=((kernel_size[0] - 1) // 2, 0)),
                nn.Conv2d(rank, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            ]
        elif self.use_depthwise_separable_conv_t:
            conv_layers = [
                nn.Conv2d(expanded_channels, expanded_channels, (kernel_size[0], 1),
                          stride=(stride, 1), groups=expanded_channels, 
                          padding=((kernel_size[0] - 1) // 2, 0)),
                nn.Conv2d(expanded_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            ]
        elif self.use_low_rank_t:
            rank = max(int(expanded_channels * rank_factor_t), 2)
            conv_layers = [
                nn.Conv2d(expanded_channels, rank, (1, 1), stride=(1, 1)),
                nn.Conv2d(rank, out_channels, (kernel_size[0], 1), 
                          stride=(stride, 1), padding=((kernel_size[0] - 1) // 2, 0)),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            ]
        else:
            conv_layers = [
                nn.Conv2d(expanded_channels, out_channels, (kernel_size[0], 1),
                          (stride, 1), padding=((kernel_size[0] - 1) // 2, 0)),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            ]
        
        if self.use_expansion_conv_t:
            conv_layers.insert(0, nn.Conv2d(out_channels, expanded_channels, 1))
            conv_layers.insert(1, nn.BatchNorm2d(expanded_channels))
            conv_layers.insert(2, nn.ReLU(inplace=True))

        self.tcn = nn.Sequential(*conv_layers)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        if act is None:
            act = 'relu'
        __activations = {
            'relu': nn.ReLU(inplace=True),
            'relu6': nn.ReLU6(inplace=True),
            'hswish': HardSwish(inplace=True),
            'swish': Swish(inplace=True),
        }
        self.act = __activations[act]

        self.attention = attention
        if self.attention:
            self.att = Attention_Layer(out_channels, att_type, self.act)

    def forward(self, x, A, length=None):

        res = self.residual(x)
        x = self.scn(x, A)
        x = self.tcn(x) + res

        # return self.relu(x)

        if self.attention:
            return self.att(x)
        else:
            return self.act(x)
        
class SpatialGraphConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True,
                 adap_graph=True,
                 weighted_sum=True,
                 use_depthwise_separable_conv=False,
                 use_low_rank=False,
                 rank_factor=0.5,
                 use_expansion_conv=False,
                 expansion_factor=2.0,
                 use_low_rank_adap=False,
                 rank_factor_adap=0.5,
                 n_head=4,
                 d_kc=0.25,
                 d_vc=0.25):
        super().__init__()

        self.kernel_size = kernel_size
        self.adap_graph = adap_graph
        self.weighted_sum = weighted_sum

        self.use_depthwise_separable_conv = use_depthwise_separable_conv
        self.use_low_rank = use_low_rank

        self.bn = nn.BatchNorm2d(in_channels)

        if int(d_kc * in_channels) == 0:
            d_kc = 1
            d_vc = 1

        self.use_expansion_conv = use_expansion_conv
        self.expansion_factor = expansion_factor

        expanded_channels = int(in_channels * self.expansion_factor) if self.use_expansion_conv else in_channels

        conv_layers = []
        if self.use_expansion_conv:
            conv_layers.extend([
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=bias),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU(inplace=True),
            ])
            
        if self.use_depthwise_separable_conv:
            depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=(t_kernel_size, 1), groups=expanded_channels, 
                                            padding=(t_padding, 0), stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)
            if self.use_low_rank:
                rank = max(int(out_channels * rank_factor), 2)
                pointwise_conv = nn.Sequential(
                    nn.Conv2d(expanded_channels, rank, kernel_size=1, bias=False),
                    nn.Conv2d(rank, out_channels * kernel_size, kernel_size=1, bias=bias),
                )
            else:
                pointwise_conv = nn.Conv2d(expanded_channels, out_channels * kernel_size, kernel_size=1, bias=bias)
            conv_layers.extend([depthwise_conv, pointwise_conv])
        elif self.use_low_rank:
            rank = max(int(out_channels * rank_factor), 2)
            conv_layers.extend([nn.Conv2d(expanded_channels, rank, kernel_size=(t_kernel_size, 1), 
                                             padding=(t_padding, 0), stride=(t_stride, 1), 
                                             dilation=(t_dilation, 1), bias=False),
                                nn.Conv2d(rank, out_channels * kernel_size, kernel_size=1, bias=bias)])
        else:
            conv_layers.append(nn.Conv2d(expanded_channels, out_channels * kernel_size, kernel_size=(t_kernel_size, 1), 
                                  padding=(t_padding, 0), stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias))

        self.conv = nn.Sequential(*conv_layers)

        if adap_graph is True:
            self.adapconv = AdapGraphConv(n_head,
                                          d_in=in_channels,
                                          d_out=out_channels,
                                          d_k=int(d_kc * out_channels),
                                          d_v=int(out_channels * d_vc),
                                          use_low_rank=use_low_rank_adap,
                                          rank_factor=rank_factor_adap,
                                          residual=True,
                                          res_fc=False)
            if weighted_sum is True:
                print('[Info] gate activated.')
                w = nn.Parameter(torch.tensor(1.0, dtype=torch.float32),
                                requires_grad=True)
                self.register_parameter('w', w)

        self.out = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True))

    def forward(self, x, A):

        inp = self.bn(x)

        f_c = self.conv(inp)

        # spatial graph convolution
        N, KC, T, V = f_c.size()
        f_c = f_c.view(N, self.kernel_size, KC // self.kernel_size, T, V)
        f_c = torch.einsum('nkctv,kvw->nctw', (f_c, A))

        if self.adap_graph:
            N, C, T, V = inp.size()
            f_a = inp.permute(0, 2, 3, 1).contiguous().view(N * T, V, C)
            f_a, _ = self.adapconv(f_a, f_a, f_a)
            f_a = f_a.view(N, T, V, -1).permute(0, 3, 1, 2)  # N, C, T, V

            if self.weighted_sum:
                f = (f_a * self.w + f_c) / 2
            else:
                f = (f_a + f_c) / 2
        else:
            f = f_c

        f = self.out(f)

        return f
    
class AdapGraphConv(nn.Module):
    def __init__(self,
                 n_head,
                 d_in,
                 d_out,
                 d_k,
                 d_v,
                 residual=True,
                 res_fc=False,
                 use_low_rank=False,
                 rank_factor=0.5,
                 dropout=0.1,
                 a_dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_in = d_in
        self.d_out = d_out
        self.d_k = d_k
        self.d_v = d_v

        self.residual = residual
        self.res_fc = res_fc

        self.use_low_rank = use_low_rank

        if self.use_low_rank:
            rank = max(int(self.d_in * rank_factor), 2)
            self.w_q_low_rank = nn.Sequential(
                nn.Linear(d_in, rank, bias=False),
                nn.Linear(rank, n_head * d_k, bias=False),
            )
            self.w_k_low_rank = nn.Sequential(
                nn.Linear(d_in, rank, bias=False),
                nn.Linear(rank, n_head * d_k, bias=False),
            )
            self.w_v_low_rank = nn.Sequential(
                nn.Linear(d_in, rank, bias=False),
                nn.Linear(rank, n_head * d_v, bias=False),
            )
        else:
            self.w_q = nn.Linear(d_in, n_head * d_k, bias=False)
            self.w_k = nn.Linear(d_in, n_head * d_k, bias=False)
            self.w_v = nn.Linear(d_in, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_out, bias=False)

        if residual:
            self.res = nn.Linear(
                d_in, d_out) if res_fc or (d_in != d_out) else lambda x: x

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.a_drop = nn.Dropout(a_dropout)

    def forward(self, q, k, v):

        assert self.d_in == v.size(2)

        NT, V, C = v.size()

        if self.residual:
            res = self.res(v)

        q = self.layer_norm(q)

        if self.use_low_rank:
            q = self.w_q_low_rank(q)
            k = self.w_k_low_rank(k)
            v = self.w_v_low_rank(v)
        else:
            q = self.w_q(q)
            k = self.w_k(k)
            v = self.w_v(v)

        q = q.view(NT, V, self.n_head, self.d_k)
        k = k.view(NT, V, self.n_head, self.d_k)
        v = v.view(NT, V, self.n_head, self.d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        A_adap = torch.matmul(q, k.transpose(2, 3)) / (self.d_k**0.5)
        A_adap = self.a_drop(F.softmax(A_adap, dim=3))

        x = torch.matmul(A_adap, v)

        x = x.transpose(1, 2).contiguous().view(NT, V, -1)
        x = self.dropout(self.fc(x))

        if self.residual:
            x += res

        return x, A_adap
        
class Swish(nn.Module):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        inner = nn.functional.relu6(x + 3.).div_(6.)
        return x.mul_(inner) if self.inplace else x.mul(inner)