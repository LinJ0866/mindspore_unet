import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.operations as F
from mindspore import Tensor
from mindspore.ops import functional as F2

from config import cfg

class MyLoss(nn.Cell):
    def __init__(self, reduction='mean'):
        super(MyLoss, self).__init__()

        if reduction is None:
            reduction = 'none'

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction method for {reduction.lower()} is not supported")

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False
        self.reduce_mean = F.ReduceMean() # 各个维度的平均值
        self.reduce_sum = F.ReduceSum()
        self.mul = F.Mul()
        self.cast = F.Cast()
    
    def get_axis(self, x):
        shape = F2.shape(x)
        length = F2.tuple_len(shape)
        perm = F2.make_range(0, length)
        return perm
    
    def get_loss(self, x, weights=1.0):
        input_dtype = x.dtype
        x=self.cast(x, mstype.float32)
        weights = self.cast(weights, mstype.float32)
        x = self.mul(weights, x)
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))

        x = self.cast(x, input_dtype)
        return x
    
    def construct(self, base, target):
        raise NotImplementedError


class CrossEntropyWithLogits(MyLoss):
    def __init__(self):
        super(CrossEntropyWithLogits, self).__init__()
        self.one_hot = F.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = F.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = F.NotEqual()
        self.num_cls = cfg.num_cls
        self.ignore_label = cfg.ignore_label
        self.mul = F.Mul()
        self.sum = F.ReduceSum(False)
        self.div = F.RealDiv()
        self.transpose = F.Transpose()
        self.reshape = F.Reshape()
    
    def construct(self, logits, label):
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))

        labels_int = self.cast(label, mstype.int32)
        weights = self.not_equal(labels_int, self.ignore_label)
        labels_int = self.mul(weights, labels_int)

        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        one_hot_labels = self.reshape(one_hot_labels, (-1, self.num_cls))

        loss = self.ce(logits_, one_hot_labels)
        return loss

class MultiCrossEntropyWithLogits(nn.Cell):
    def __init__(self):
        super(MultiCrossEntropyWithLogits, self).__init__()
        self.loss = CrossEntropyWithLogits()
        self.squeeze = F.Squeeze(axis=0)

    def construct(self, logits, label):
        total_loss = 0
        for i in range(len(logits)):
            total_loss += self.loss(self.squeeze(logits[i:i+1]), label)
        return total_loss