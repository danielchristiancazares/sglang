import sys

import torch

sys.path.insert(0, "/private/tmp")
import _sglang_metal_gguf as metal_gguf

BATCH = 1
KEY_HEADS = 1
VALUE_HEADS = 3
KEY_DIM = 128
VALUE_DIM = 4

query = torch.ones(1, BATCH, KEY_HEADS, KEY_DIM, device="mps")
key = torch.ones_like(query)
value = torch.ones(1, BATCH, VALUE_HEADS, VALUE_DIM, device="mps")
a = torch.zeros(BATCH, VALUE_HEADS, device="mps")
b = torch.zeros(BATCH, VALUE_HEADS, device="mps")
A_log = torch.zeros(VALUE_HEADS, device="mps")
dt_bias = torch.zeros(VALUE_HEADS, device="mps")
state = torch.zeros(BATCH, VALUE_HEADS, VALUE_DIM, KEY_DIM, device="mps")
indices = torch.arange(BATCH, dtype=torch.int32, device="mps")

output = metal_gguf.gdn_decode(
    query, key, value, a, b, A_log, dt_bias, state, indices
)
torch.mps.synchronize()
print(output.cpu())
