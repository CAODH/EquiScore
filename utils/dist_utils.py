import torch
import math
import numpy as np
import os.path
import time
from torch import distributed as dist
import random
def seed_torch(seed=42):
    """
    Random seed for reproducibility

    """
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    empty = []
    for i in range(8):
        command = 'nvidia-smi -i '+str(i)+' | grep "No running" | wc -l'
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        if int(output)==1:
            empty.append(i)
    if len(empty)<ngpus:
        print ('avaliable gpus are less than required')
        exit(-1)
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
    return cmd
def get_available_gpu(num_gpu=1, min_memory=1000, sample=3, nitro_restriction=True, verbose=True):
    """
    docstring:
        get available GPU for you, if you have 4 GPU, it will return the GPU with lowest memory usage.

    :param num_gpu: number of GPU you want to use
    :param min_memory: minimum memory
    :param sample: number of sample
    :param nitro_restriction: if True then will not distribute the last GPU for you.
    :param verbose: verbose mode
    :return: str of best choices, e.x. '1, 2'
    """
    sum = None
    for _ in range(sample):
        info = os.popen('nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv').read()
        info = np.array([[id] + t.replace('%', '').replace('MiB','').split(',') for id, t in enumerate(info.split('\n')[1:-1])]).\
            astype(np.int)
        sum = info + (sum if sum is not None else 0)
        time.sleep(0.2)
    avg = sum//sample
    if nitro_restriction:
        avg = avg[:-1]
    available = avg[np.where(avg[:,2] > min_memory)]  
    if len(available) < num_gpu:
        print ('avaliable gpus are less than required')
        exit(-1)
    if available.shape[0] == 0:
        print('No GPU available')
        return ''
    select = ', '.join(available[np.argsort(available[:,1])[:num_gpu],0].astype(np.str).tolist())
    if verbose:
        print('Available GPU List')
        first_line = [['id', 'utilization.gpu(%)', 'memory.free(MiB)']]
        matrix = first_line + available.astype(np.int).tolist()
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))
        print('Select id #' + select + ' for you.')
    return select
def data_to_device(sample,device):

    """
    set graph data to device
    
    """
    data_flag = []
    data = []
    for i in sample.get_att():
        if type(i) is torch.Tensor:
            data.append(i.to(device))
            data_flag.append(1)
        else:
            data_flag.append(None)
    return data_flag,data
def average_gradients(model):
    """
    average gradients for distributed training
    """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data,op = torch.distributed.ReduceOp.SUM)
            param.grad.data /= size
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    ref from: https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/trainer_pt_utils.py
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)
    def __len__(self):
        return self.num_samples

def distributed_concat(tensor, num_total_examples):
    """
    Concat multi tensor from different process
    """
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]