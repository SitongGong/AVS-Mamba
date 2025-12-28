from utils.pyutils import AverageMeter
import torch.distributed as dist
import torch

class LossUtil:
    def __init__(self, weight_dict) -> None:
        self.loss_weight_dict = weight_dict
        self.avg_loss = dict()
        self.avg_loss['total_loss'] = AverageMeter('total_loss')
        # for k in weight_dict.keys():
        #     self.avg_loss[k] = AverageMeter(k)

    def add_loss(self, loss, loss_dict):
        self.avg_loss['total_loss'].add({'total_loss': loss.item()})
        for k, v in loss_dict.items():
            meter = self.avg_loss.get(k, None)
            if meter is None:
                meter = AverageMeter(k)
                self.avg_loss[k] = meter

            self.avg_loss[k].add({k: v})

    def pretty_out(self):
        f = 'Total_Loss:%.4f, ' % (
            self.avg_loss['total_loss'].pop('total_loss'))
        for k in self.avg_loss.keys():
            if k == 'total_loss':
                continue
            t = '%s:%.4f, ' % (k, self.avg_loss[k].pop(k))
            f += t
        return f

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()           # 全局进程数量
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)                 # keys
            values.append(input_dict[k])      # values
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v.item() for k, v in zip(names, values)}
    return reduced_dict


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()