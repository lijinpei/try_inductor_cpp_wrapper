import torch
import torch._inductor.config as config
config.cpp_wrapper = False

def fn(x):
    return torch.tensor(list(range(2, 40, 2)), device=x.device) + x

x = torch.randn(1)
opt_fn = torch.compile()(fn)
y = opt_fn(x)
