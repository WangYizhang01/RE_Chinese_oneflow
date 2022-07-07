import oneflow
import random
import logging
import numpy as np
from typing import List, Tuple, Dict, Union

logger = logging.getLogger(__name__)

__all__ = [
    'manual_seed',
    'seq_len_to_mask',
    'to_one_hot',
]


def manual_seed(seed: int = 1) -> None:
    """
        设置seed。
    """
    random.seed(seed)
    np.random.seed(seed)
    oneflow.manual_seed(seed)
    oneflow.cuda.manual_seed(seed)
    oneflow.cuda.manual_seed_all(seed)
    oneflow.backends.cudnn.deterministic = True
    oneflow.backends.cudnn.benchmark = False



def seq_len_to_mask(seq_len: Union[List, np.ndarray, oneflow.Tensor], max_len=None, mask_pos_to_true=True):
    """
    将一个表示sequence length的一维数组转换为二维的mask，默认pad的位置为1。
    转变 1-d seq_len到2-d mask。

    Args :
        seq_len (list, np.ndarray, oneflow.LongTensor) : shape将是(B,)
        max_len (int): 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    Return: 
        mask (np.ndarray, oneflow.Tensor) : shape将是(B, max_length)， oneflow.uint8
    """
    if isinstance(seq_len, list):
        seq_len = np.array(seq_len)

    if isinstance(seq_len, np.ndarray):
        seq_len = oneflow.from_numpy(seq_len)

    if isinstance(seq_len, oneflow.Tensor):
        assert seq_len.dim() == 1, logger.error(f"seq_len can only have one dimension, got {seq_len.dim()} != 1.")
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        if isinstance(max_len, oneflow.Tensor):
            max_len = max_len.item()
        try:
            broad_cast_seq_len = oneflow.arange(max_len).expand(batch_size, -1).to(seq_len.device)
        except TypeError:
            print(f'max_len: {max_len}, {type(max_len)}')
            print(f'oneflow.arange(max_len): {oneflow.arange(max_len)}, {type(oneflow.arange(max_len))}')
        if mask_pos_to_true:
            mask = broad_cast_seq_len.ge(seq_len.unsqueeze(1))
        else:
            mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise logger.error("Only support 1-d list or 1-d numpy.ndarray or 1-d oneflow.Tensor.")

    return mask


def to_one_hot(x: oneflow.Tensor, length: int) -> oneflow.Tensor:
    """
    Args:
        x (oneflow.Tensor):[B] , 一般是 target 的值
        length (int) : L ,一般是关系种类树
    Return:
        x_one_hot.to(device=x.device) (oneflow.Tensor) : [B, L]  每一行，只有对应位置为1，其余为0
    """
    B = x.size(0)
    x_one_hot = oneflow.zeros(B, length).to(device=x.device)
    for i in range(B):
        x_one_hot[i][x[i]] = 1.0

    return x_one_hot.to(device=x.device)