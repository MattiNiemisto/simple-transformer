import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from model import Transformer



src_vocab_size = 50
tgt_vocab_size = 50
d_model = 128 #512
num_heads = 8
num_layers = 2 #6
d_ff = 512 #2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)


criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

with open("logfile.txt", "w") as log:
    print("starting training....", file = log)

for epoch in range(100):
    print(epoch)
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    with open("logfile.txt", "a") as log:
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}", file = log)