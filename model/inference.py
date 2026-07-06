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


# Let there be a Tokenizer to translate strings to discrete symbols and back
with open('./Data/token_id.json', 'r', encoding='utf-8') as file:
    token_ids = json.load(file)
token_ids_map = {tok: i for i, tok in enumerate(token_ids)}
vocab_size = len(token_ids)


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


# Load best model
model = GPT().to(device)
model.load_state_dict(torch.load('./Data/best_model.pt', map_location=device))
model.eval()


# Load rhyme table, tone table, and poem templates
with open('./Data/rhyme_table.json', 'r', encoding='utf-8') as file:
    rhyme_table = json.load(file)

with open('./Data/tone_table.json', 'r', encoding='utf-8') as file:
    tone_table = json.load(file)

with open('./Data/templates.json', 'r', encoding='utf-8') as file:
    templates = json.load(file)

rhyme_dict = {}
for character, rhyme_group in rhyme_table.items():
    rhyme_dict.setdefault(rhyme_group, []).append(character)

tone_dict = {}
for character, tone in tone_table.items():
    tone_dict.setdefault(tone, []).append(character)


# Helper function to enforce tone and rhyme requirements
def apply_constraints(logits, tone, rhyme_required, rhyme_group, prev_rhymes):
    mask = torch.zeros_like(logits)

    # check tone
    if tone == "平":
        ze_chars = tone_dict['仄']
        ze_token_ids = [token_ids_map[char] for char in ze_chars]
        mask[ze_token_ids] = float('-inf')
    elif tone == '仄':
        ping_chars = tone_dict['平']
        ping_token_ids = [token_ids_map[char] for char in ping_chars]
        mask[ping_token_ids] = float('-inf')

    # check rhyme
    if rhyme_required:
        non_rhyme_chars = [
            character for rhyme_groups, characters in rhyme_dict.items()
            if rhyme_groups != rhyme_group
            for character in characters
        ]
        non_rhyme_token_ids = [token_ids_map[char] for char in non_rhyme_chars]
        mask[non_rhyme_token_ids] = float('-inf')

        # cannot rhyme on the same character
        if len(prev_rhymes) > 0:
            prev_rhyme_token_ids = [token_ids_map[char] for char in prev_rhymes]
            mask[prev_rhyme_token_ids] = float('-inf')

    return mask


# Inference - generate new poems
title = str(input("Enter a title for the poem: "))
prefix = [token_ids_map['<BOS>']] + [token_ids_map[c] for c in title] + [token_ids_map['<SEP>']]

temperature = 0.6 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated poems) ---")

with torch.no_grad():
    for sample_idx in range(5):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]

        # Prime KV cache with title — run forward but don't sample
        for pos_id, token_id in enumerate(prefix):
            logits = model.forward_step(token_id, pos_id, keys, values)

        print("Sample %d:" % (sample_idx + 1))
        sample = []
        rhyme_group = None
        rhyme_required = False
        prev_rhymes = []
        generated_ids = []
        repetition_penalty = 1.3

        # random template
        template_num = random.randint(1, 4)
        template = templates["template" + str(template_num)]

        for pos_id in range(block_size - len(prefix)):

            # ban unknown characters and invalid tokens
            logits[token_ids_map['<UNK>']] = float('-inf')
            logits[token_ids_map['<BOS>']] = float('-inf')
            logits[token_ids_map['<SEP>']] = float('-inf')

            # repetition penalty
            if generated_ids:
                for tid in generated_ids:
                    if logits[tid] > 0:
                        logits[tid] /= repetition_penalty
                    else:
                        logits[tid] *= repetition_penalty

            # check if rhyme required
            if (pos_id % 14 == 13 and rhyme_group != None):
                rhyme_required = True
            else:
                rhyme_required = False

            mask = apply_constraints(logits, template[pos_id % 56], rhyme_required, rhyme_group, prev_rhymes)
            probs = torch.softmax((logits + mask) / temperature, dim=0)
            token_id = torch.multinomial(probs, 1).item()

            # check if poem ends
            if token_id == token_ids_map['<EOS>']:
                break

            character = token_ids[token_id]
            sample.append(character)
            print(character, end="")

            if token_id not in generated_ids:
                generated_ids.append(token_id)

            if (rhyme_required):
                prev_rhymes.append(character)

            # confirm rhyme group if character is the first in poem that rhymes
            if (template_num in [1, 3] and pos_id % 14 == 13) or (template_num in [2, 4] and pos_id % 14 == 6):
                if rhyme_group == None:
                    rhyme_group = rhyme_table[character]
                    prev_rhymes.append(character)

            # add punctuations
            if (pos_id % 14 == 6):
                print("，", end="")
            elif (pos_id % 14 == 13):
                print("。")

            logits = model.forward_step(token_id, pos_id + len(prefix), keys, values)
        print()