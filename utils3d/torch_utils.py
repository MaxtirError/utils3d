import torch
import numpy as np

def to_torch_tensor_(arr, to_cuda=True, to_float=True, method="stack"):
    assert method in ["stack", "cat"], f"method must be 'stack' or 'cat', but got {method}"
    if isinstance(arr, list):
        if isinstance(arr[0], torch.Tensor):
            if method == "stack":
                arr = torch.stack(arr, dim=0)
            else:
                arr = torch.cat(arr, dim=0)
        elif isinstance(arr[0], np.ndarray):
            if method == "stack":
                arr = torch.from_numpy(np.stack(arr, axis=0))
            else:
                arr = torch.from_numpy(np.concatenate(arr, axis=0))
    elif isinstance(arr, dict):
        try:
            arr = {k: to_torch_tensor_(v, to_cuda=to_cuda, to_float=to_float, method=method) for k, v in arr.items()}
            return arr
        except:
            "dict values must be list, np.ndarray or torch.Tensor"
            return None
    elif isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    else:
        assert isinstance(arr, torch.Tensor), f"arr must be list, np.ndarray or torch.Tensor, but got {type(arr)}"
    
    if to_float:
        arr = arr.float()
    if to_cuda:
        arr = arr.cuda()
    return arr

def to_numpy_array_(arr):
    return arr.detach().cpu().numpy()

def to_tensors(*arrs, **kwargs):
    return tuple(to_torch_tensor_(arr, **kwargs) for arr in arrs)

def to_numpys(*arrs):
    return tuple(to_numpy_array_(arr) for arr in arrs)
        
def to_tensor(arr, **kwargs):
    return to_torch_tensor_(arr, **kwargs)

def to_numpy(arr, **kwargs):
    return to_numpy_array_(arr, **kwargs)