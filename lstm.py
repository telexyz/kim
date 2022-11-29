#!/usr/bin/python3

import numpy as np
import kim as kim
import kim.nn as nn

import datetime
from timeit import default_timer as timer

started_at = timer()

# Fix python3 -m pytest -k "test_lstm[cuda-True-True-1-1-1-1-13]" # bị hang => đã fix!
# See https://github.com/telexyz/kim/commit/62a339af99d07af04d7f97cb29508a3435ce2299#diff-82bae5d94a873b1e01bbe5992c1dc0855b66b0bf41f3489945fa5bcd83ee3f91R225
device, init_hidden, bias = kim.default_device(), True, True
hidden_size, input_size, batch_size, num_layers, seq_length = 80, 100, 100, 6, 200

x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
c0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)

model = nn.LSTM(input_size, hidden_size, num_layers, bias, device=device)

if init_hidden:
    output, (h, c) = model(kim.Tensor(x, device=device), (kim.Tensor(h0, device=device), kim.Tensor(c0, device=device)))
else:
    output, (h, c) = model(kim.Tensor(x, device=device), None)

print(">>>", output.shape)
output.sum().backward()
print("\nDONE")

node_count = kim.autograd.CompGraph.NODE_COUNT
timespent = datetime.timedelta(seconds=timer() - started_at)
print(f"seq_length %i, node_count %i, timespent: %s" % (seq_length, node_count, timespent))

print(model.lstm_cells[0].W_ih.grad.detach().numpy())
