import json

with open("./Data/activation_scales.json", 'r') as file:
    activation_scale = json.load(file)

with open("./Data/weight_scales.json", 'r') as file:
    weight_scale = json.load(file)

num_of_layers = 4
scale_table = {}


##### For embed addition #####
S = 8
scale_table['S_emb'] = S

print("Embed:")
scale_wte = weight_scale["wte.weight"]
scale_wpe = weight_scale["wpe.weight"]
scale_emb_sum = activation_scale["emb_sum"]

# wte
r_wte = scale_wte / scale_emb_sum
M_wte = round(r_wte * (2**S))
scale_table['M_wte'] = M_wte

# wpe
r_wpe = scale_wpe / scale_emb_sum
M_wpe = round(r_wpe * (2**S))
scale_table['M_wpe'] = M_wpe

print(f"r_wte: {r_wte} \t S_wte: {S} \t M_wte: {M_wte}")
print(f"r_wpe: {r_wpe} \t S_wpe: {S} \t M_wpe: {M_wpe}")


##### For matvec multiply & accumulate #####
print("\nMatvec:")
S = 16
scale_table['S_matvec'] = S

# attn_wq
scale_x_norm = activation_scale["x_norm"]
scale_q = activation_scale["q"]
for i in range(num_of_layers):
    scale_x_attn = weight_scale[f"layers.{i}.attn_wq.weight"]
    r = scale_x_norm * scale_x_attn / scale_q
    M = round(r * (2**S))
    scale_table[f"M_attn_wq_layer{i}"] = M
    print(f"attn_wq_layer{i}: \t r: {r} \t S: {S} \t M: {M}")

# attn_wk
scale_k = activation_scale["k"]
for i in range(num_of_layers):
    scale_x_attn = weight_scale[f"layers.{i}.attn_wk.weight"]
    r = scale_x_norm * scale_x_attn / scale_k
    M = round(r * (2**S))
    scale_table[f"M_attn_wk_layer{i}"] = M
    print(f"attn_wk_layer{i}: \t r: {r} \t S: {S} \t M: {M}")

# attn_wv
scale_v = activation_scale["v"]
for i in range(num_of_layers):
    scale_x_attn = weight_scale[f"layers.{i}.attn_wv.weight"]
    r = scale_x_norm * scale_x_attn / scale_v
    M = round(r * (2**S))
    scale_table[f"M_attn_wv_layer{i}"] = M
    print(f"attn_wv_layer{i}: \t r: {r} \t S: {S} \t M: {M}")

# attn_wo
scale_x_attn = activation_scale["x_attn"]
scale_post_wo = activation_scale["post_wo"]
for i in range(num_of_layers):
    scale_x_attn_wo = weight_scale[f"layers.{i}.attn_wo.weight"]
    r = scale_x_attn * scale_x_attn_wo / scale_post_wo
    M = round(r * (2**S))
    scale_table[f"M_attn_wo_layer{i}"] = M
    print(f"attn_wo_layer{i}: \t r: {r} \t S: {S} \t M: {M}")

# mlp_fc1
scale_post_mlp_norm = activation_scale["post_mlp_norm"]
scale_post_mlp_fc1 = activation_scale["post_mlp_fc1"]
for i in range(num_of_layers):
    scale_x_mlp_fc1 = weight_scale[f"layers.{i}.mlp_fc1.weight"]
    r = scale_post_mlp_norm * scale_x_mlp_fc1 / scale_post_mlp_fc1
    M = round(r * (2**S))
    scale_table[f"M_mlp_fc1_layer{i}"] = M
    print(f"mlp_fc1_layer{i}: \t r: {r} \t S: {S} \t M: {M}")

# mlp_fc2
scale_post_relu = activation_scale["post_relu"]
scale_post_mlp_fc2 = activation_scale["post_mlp_fc2"]
for i in range(num_of_layers):
    scale_x_mlp_fc2 = weight_scale[f"layers.{i}.mlp_fc2.weight"]
    r = scale_post_relu * scale_x_mlp_fc2 / scale_post_mlp_fc2
    M = round(r * (2**S))
    scale_table[f"M_mlp_fc2_layer{i}"] = M
    print(f"mlp_fc2_layer{i}: \t r: {r} \t S: {S} \t M: {M}")

# lm_head
scale_final_norm = activation_scale["final_norm"]
scale_final_logits = activation_scale["final_logits"]
scale_lm_head = weight_scale["wte.weight"] # weight tied
r = scale_final_norm * scale_lm_head / scale_final_logits
M = round(r * (2**S))
scale_table[f"M_lm_head"] = M
print(f"lm_head: \t r: {r} \t S: {S} \t M: {M}")

with open("./Data/scales.json", 'w') as file:
    json.dump(scale_table, file, indent=4)