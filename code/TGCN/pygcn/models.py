import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    """
    该模型通过两层图卷积网络（GraphConvolution）将输入x转换为输出，
    1、第一层包含nfeat节点输入特征和nhid个隐藏单元，第二层包含nhid个隐藏单元和nclass个输出节点。
    2、模型的前向传递函数定义了relu激活函数和log_softmax作为最后一步的输出。
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        模型的前向传递函数定义了relu激活函数和log_softmax作为最后一步的输出，
        dropout正则化功能以避免过度拟合
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
