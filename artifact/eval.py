from torch import nn

class MYevaluate(nn.Module):
    def __init__(self):#定义loss计算方式
        super(MYevaluate, self).__init__()
        self.MSELOSS = nn.MSELoss(reduction='mean')

    def forward(self, input, target):
        return self.MSELOSS(input, target)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res