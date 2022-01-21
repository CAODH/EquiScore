import os
import glob
# files = glob.glob('../GNN_graphformer/*')
import torch 
print('cuda: is on time? ',torch.cuda.is_available())
# for key in files:
    # print(key)
ngpus = torch.cuda.device_count()
print('一共有多少GPU: ',ngpus)
def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    empty = []
    for i in range(8):
        command = 'nvidia-smi -i '+str(i)+' | grep "No running" | wc -l'
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        #print('nvidia-smi -i '+str(i)+' | grep "No running" | wc -l > empty_gpu_check')
        if int(output)==1:
            empty.append(i)
    if len(empty)<ngpus:
        print ('avaliable gpus are less than required')
        exit(-1)
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
    return cmd
print('GPU can be used: ',set_cuda_visible_device(ngpus))