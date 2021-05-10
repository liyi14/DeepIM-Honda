import torch

def show_contents(data):
    target = data
    for k in target.keys():
        if torch.is_tensor(target[k]):
            print(k, target[k].shape, target[k].dtype, target[k].device)
        else:
            print(k, type(target[k]))
