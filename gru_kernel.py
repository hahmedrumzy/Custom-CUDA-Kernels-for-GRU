
import os
import torch
from torch.utils.cpp_extension import load

KERNEL_READY = False

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        here = os.path.dirname(os.path.abspath(__file__))
        _ext = load(name="gru_cell_ext", sources=[os.path.join(here, "gru_cell.cu")], verbose=True)
    return _ext





def custom_gru(x, w_ih, w_hh, b_ih, b_hh  ):
    if not KERNEL_READY or not x.is_cuda:
        raise NotImplementedError
    ext = _get_ext()
    batch, seq_len, _ = x.shape
    hidden = w_hh.shape[1]

    gi_all = x @ w_ih.T + b_ih                 
    h = x.new_zeros(batch, hidden)
    outputs = []
    for t in range(seq_len):
        gh = h @ w_hh.T + b_hh                 

        h = ext.gru_pointwise(gi_all[:, t].contiguous(), gh.contiguous(), h.contiguous())
        outputs.append(h)
    return torch.stack(outputs, dim=1)
