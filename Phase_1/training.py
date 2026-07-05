import math
import random
import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42) # Let there be order among chaos
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Let there be an input dataset `docs`: list[str] of documents (e.g. a dataset of names)
docs = [l.strip() for l in open('./Data/data.txt', 'r', encoding='utf-8').read().strip().split('\n') if l.strip()] # list[str] of documents
random.shuffle(docs)
val_size = int(0.05 * len(docs))
val_docs = docs[:val_size]
docs = docs[val_size:]
print(f"train docs: {len(docs)}, val docs: {len(val_docs)}")


# Let there be a Tokenizer to translate strings to discrete symbols and back
with open('./Data/token_id.json', 'r', encoding='utf-8') as file:
    token_ids = json.load(file)
token_ids_map = {tok: i for i, tok in enumerate(token_ids)}
vocab_size = len(token_ids)
print(f"vocab size: {vocab_size}")


# Model parameters
n_embd = 64     # embedding dimension
n_head = 4      # number of attention heads
n_layer = 4     # number of layers
block_size = 96 # maximum sequence length
head_dim = n_embd // n_head # dimension of each head


# Helper function for RMSNorm
def rmsnorm(x):
    return x * (x.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()


# Let there be a Transformer model
class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_wq = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_wk = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_wv = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_wo = nn.Linear(n_embd, n_embd, bias=False)
        self.mlp_fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.mlp_fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, _ = x.shape

        # 1) Multi-head attention block
        x_residual = x
        x = rmsnorm(x)
        q = self.attn_wq(x).view(B, T, n_head, head_dim).transpose(1, 2)
        k = self.attn_wk(x).view(B, T, n_head, head_dim).transpose(1, 2)
        v = self.attn_wv(x).view(B, T, n_head, head_dim).transpose(1, 2)

        x_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x_attn = x_attn.transpose(1, 2).contiguous().view(B, T, n_embd)
        x = self.attn_wo(x_attn) + x_residual

        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = F.relu(self.mlp_fc1(x))
        x = self.mlp_fc2(x)
        x = x + x_residual
        return x

    def forward_step(self, x, keys, values):

        # 1) Multi-head attention block
        x_residual = x
        x = rmsnorm(x)
        q = self.attn_wq(x)
        k = self.attn_wk(x)
        v = self.attn_wv(x)
        keys.append(k)
        values.append(v)
        x_attn = []
        for h in range(n_head):
            hs = slice(h * head_dim, (h + 1) * head_dim)
            q_h = q[hs]
            k_h = torch.stack([ki[hs] for ki in keys])
            v_h = torch.stack([vi[hs] for vi in values])
            attn_logits = (k_h @ q_h) / math.sqrt(head_dim)
            attn_weights = torch.softmax(attn_logits, dim=0)
            head_out = (v_h * attn_weights.unsqueeze(1)).sum(0)
            x_attn.append(head_out)
        x = self.attn_wo(torch.cat(x_attn)) + x_residual

        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = F.relu(self.mlp_fc1(x))
        x = self.mlp_fc2(x)
        x = x + x_residual
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([TransformerLayer() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        nn.init.normal_(self.wte.weight, std=0.08)
        nn.init.normal_(self.wpe.weight, std=0.08)

    def forward(self, token_ids_batch):
        B, T = token_ids_batch.shape
        pos = torch.arange(T, device=token_ids_batch.device)

        tok_emb = self.wte(token_ids_batch)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        x = rmsnorm(x)

        for layer in self.layers:
            x = layer(x)
        x = rmsnorm(x)
        return self.lm_head(x)

    def forward_step(self, token_id, pos_id, keys, values):
        tok_emb = self.wte.weight[token_id]
        pos_emb = self.wpe.weight[pos_id]
        x = tok_emb + pos_emb
        x = rmsnorm(x)

        for li, layer in enumerate(self.layers):
            x = layer.forward_step(x, keys[li], values[li])
        x = rmsnorm(x)
        return self.lm_head(x)


# instantiate the model
model = GPT().to(device)
print(f"num params: {sum(p.numel() for p in set(model.parameters()))}")


# Let there be Adam, the blessed optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.90, 0.99))


# Helper function to collate a batch of poems for training
def collate(batch_docs, block_size=block_size):
    pattern = r'<SEP>|<UNK>|.'
    PAD = token_ids_map['<EOS>']  # reuse EOS as pad id

    input_batch, target_batch = [], []
    for doc in batch_docs:
        doc_tokens = re.findall(pattern, doc)
        body_starts = doc_tokens.index('<SEP>')
        tokens = [token_ids_map['<BOS>']] + [token_ids_map[t] for t in doc_tokens] + [token_ids_map['<EOS>']]

        seq_in = tokens[:-1]
        seq_out = tokens[1:]

        # mask loss on title + <SEP>
        seq_out = [
            tok if i >= body_starts + 1 else -100
            for i, tok in enumerate(seq_out)
        ]

        pad_len = block_size - len(seq_in)
        seq_in = seq_in + [PAD] * pad_len
        seq_out = seq_out + [-100] * pad_len

        input_batch.append(seq_in)
        target_batch.append(seq_out)

    return torch.tensor(input_batch, device=device), torch.tensor(target_batch, device=device)


# Helper function to evaluate validation loss
@torch.no_grad()
def eval_val_loss(n_batches=5):
    model.eval()
    total_loss = 0.0
    for _ in range(n_batches):
        batch_docs = random.sample(val_docs, min(batch_size, len(val_docs)))
        input_ids, targets = collate(batch_docs)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)
        total_loss += loss.item()
    model.train()
    return total_loss / n_batches


# Repeat in sequence
num_steps = 50000 # number of training steps
batch_size = 64
losses = []

best_val_loss = float('inf')
val_losses = []

for step in range(num_steps):
    batch_docs = [docs[i] for i in random.sample(range(len(docs)), batch_size)]
    input_ids, targets = collate(batch_docs)
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    lr_t = 0.001 * (1 - step / num_steps)
    for pg in optimizer.param_groups:
        pg['lr'] = lr_t

    if (step % 1000 == 0):
        val_loss = eval_val_loss()
        val_losses.append(val_loss)
        print(f"step {step+1:4d} / {num_steps:4d} \t loss {loss.item():.4f} \t val {val_loss:.4f}")

        # save the model with the best validation loss
        if (val_loss < best_val_loss):
            best_val_loss = val_loss
            torch.save(model.state_dict(), './Data/best_model.pt') 


# Visualize the training and validation loss curves
import matplotlib.pyplot as plt
plt.plot(list(range(0, 50000)), losses)
plt.plot(list(range(0, 50000, 1000)), val_losses)
plt.show()