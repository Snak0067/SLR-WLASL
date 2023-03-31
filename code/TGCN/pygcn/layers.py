import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    该文件是一个实现图卷积网络层的 Python 模块。
    主要包含了一个名为 GraphConvolution 的类，用于构建图卷积层。其中使用了 PyTorch 框架提供的张量操作，如 mm 和 spmm。
    在 GraphConvolution 类的初始化中，使用了 torch.nn.parameter 模块中的 Parameter 类来定义可学习参数（即权重和偏置），
    并采用了 Xavier 初始化的方法。在 forward 函数中，通过矩阵乘法和稀疏矩阵乘法实现了图卷积操作。如果 bias 参数不为 None，
    则输出结果会加上偏置值。最后，该文件还定义了 repr 函数，用于输出图卷积层的信息。
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        使用了 torch.nn.parameter 模块中的 Parameter 类来定义可学习参数（即权重和偏置），并采用了 Xavier 初始化的方法
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        通过矩阵乘法和稀疏矩阵乘法实现了图卷积操作。如果 bias 参数不为 None，则输出结果会加上偏置值
        """
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        """
        用于输出图卷积层的信息
        """
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
