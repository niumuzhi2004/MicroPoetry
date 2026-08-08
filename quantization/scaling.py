import json

with open("./Data/activation_scales.json", 'r') as file:
    activation_scale = json.load(file)

with open("./Data/weight_scales.json", 'r') as file:
    weight_scale = json.load(file)

with open("./Data/rmsnorm.json", 'r') as file:
    rmsnorm_scale = json.load(file)
LUTs = rmsnorm_scale["luts"]
steps = rmsnorm_scale["steps"]

with open("./Data/softmax.json", 'r') as file:
    softmax_scale = json.load(file)
exp_step, recip_step = softmax_scale["exp_step"], softmax_scale["total_step"]

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


##### For norm calculating LUT index #####
print("\nNorm:")
S = 16
S_ms_real = 8
scale_table['S_norm'] = S
scale_table['S_ms_real'] = S_ms_real


# emb_norm
scale_x_input = activation_scale["emb_sum"]
r = (scale_x_input ** 2) * (2 ** 10)
M = round(r * 2**S_ms_real)
scale_table["M_emb_norm_ms_real"] = M
print(f"emb_norm ms_real: \t r: {r} \t S: {S_ms_real} \t M: {M}")

r = 1 / ((steps["emb_norm"] / (2 ** 15)) * (2 ** 10))
M = round(r * 2**S)
scale_table["M_emb_norm_idx"] = M
print(f"emb_norm idx: \t r: {r} \t S: {S} \t M: {M}")

scale_x_output = activation_scale["emb_norm"]
r = scale_x_input / (2**13 * scale_x_output)
M = round(r * 2**S)
scale_table["M_emb_norm_output"] = M
print(f"emb_norm output: \t r: {r} \t S: {S} \t M: {M}")


# x_norm
# layer 0
scale_x_input = activation_scale["emb_norm"]
r = (scale_x_input ** 2) * (2 ** 10)
M = round(r * 2**S_ms_real)
scale_table["M_x_norm_layer0_ms_real"] = M
print(f"x_norm (layer 0) ms_real: \t r: {r} \t S: {S_ms_real} \t M: {M}")

scale_x_output = activation_scale["x_norm"]
r = scale_x_input / (2**13 * scale_x_output)
M = round(r * 2**S)
scale_table["M_x_norm_layer0_output"] = M
print(f"x_norm (layer 0) output: \t r: {r} \t S: {S} \t M: {M}")

# layer 1-3
scale_x_input = activation_scale["last_x+residual"]
r = (scale_x_input ** 2) * (2 ** 10)
M = round(r * 2**S_ms_real)
scale_table["M_x_norm_layer123_ms_real"] = M
print(f"x_norm (layer 1-3) ms_real: \t r: {r} \t S: {S_ms_real} \t M: {M}")

scale_x_output = activation_scale["x_norm"]
r = scale_x_input / (2**13 * scale_x_output)
M = round(r * 2**S)
scale_table["M_x_norm_layer123_output"] = M
print(f"x_norm (layer 1-3) output: \t r: {r} \t S: {S} \t M: {M}")

r = 1 / ((steps["x_norm"] / (2 ** 15)) * (2 ** 10))
M = round(r * 2**S)
scale_table["M_x_norm_idx"] = M
print(f"x_norm idx: \t r: {r} \t S: {S} \t M: {M}")


# pre_mlp_norm
scale_x_input = activation_scale["x+residual"]
r = (scale_x_input ** 2) * (2 ** 10)
M = round(r * 2**S_ms_real)
scale_table["M_pre_mlp_norm_ms_real"] = M
print(f"pre_mlp_norm ms_real: \t r: {r} \t S: {S_ms_real} \t M: {M}")

r = 1 / ((steps["post_mlp_norm"] / (2 ** 15)) * (2 ** 10))
M = round(r * 2**S)
scale_table["M_pre_mlp_norm_idx"] = M
print(f"pre_mlp_norm idx: \t r: {r} \t S: {S} \t M: {M}")

scale_x_output = activation_scale["post_mlp_norm"]
r = scale_x_input / (2**13 * scale_x_output)
M = round(r * 2**S)
scale_table["M_pre_mlp_norm_output"] = M
print(f"pre_mlp_norm output: \t r: {r} \t S: {S} \t M: {M}")


# final_norm
scale_x_input = activation_scale["last_x+residual"]
r = (scale_x_input ** 2) * (2 ** 10)
M = round(r * 2**S_ms_real)
scale_table["M_final_norm_ms_real"] = M
print(f"final_norm ms_real: \t r: {r} \t S: {S_ms_real} \t M: {M}")

r = 1 / ((steps["final_norm"] / (2 ** 15)) * (2 ** 10))
M = round(r * 2**S)
scale_table["M_final_norm_idx"] = M
print(f"final_norm idx: \t r: {r} \t S: {S} \t M: {M}")

scale_x_output = activation_scale["final_norm"]
r = scale_x_input / (2**13 * scale_x_output)
M = round(r * 2**S)
scale_table["M_final_norm_output"] = M
print(f"final_norm output: \t r: {r} \t S: {S} \t M: {M}")


##### For softmax #####
print("\nSoftmax:")
S = 16
scale_table['S_softmax'] = S

# attn_softmax
r = activation_scale["attn_logits"]
M = round(r * 2**S)
scale_table["M_attn_softmax_diff"] = M
print(f"attn_softmax diff: \t r: {r} \t S: {S} \t M: {M}")

# final_softmax
r = activation_scale["final_logits"]
M = round(r * 2**S)
scale_table["M_final_softmax_diff"] = M
print(f"final_softmax diff: \t r: {r} \t S: {S} \t M: {M}")


r = 1 / (2**12 * exp_step)
M = round(r * 2**S)
scale_table["M_exp_idx"] = M
print(f"softmax exp idx: \t r: {r} \t S: {S} \t M: {M}")

r = 1 / (2**15 * recip_step)
M = round(r * 2**S)
scale_table["M_recip_idx"] = M
print(f"softmax recip idx: \t r: {r} \t S: {S} \t M: {M}")


S_output = 32
scale_table['S_softmax_output'] = S_output
r = 255 / (2**30)
M = round(r * 2**S_output)
scale_table["M_softmax_output"] = M
print(f"softmax_output: \t r: {r} \t S: {S_output} \t M: {M}")



##### For attn_score #####
print("\nattn_score:")
S = 16
scale_table['S_attn_score'] = S

scale_q = activation_scale["q"]
scale_k = activation_scale["k"]
scale_attn_logits = activation_scale["attn_logits"]

r = (scale_q * scale_k) / (4 * scale_attn_logits)
M = round(r * 2**S)
scale_table["M_attn_score"] = M
print(f"attn_score: \t r: {r} \t S: {S} \t M: {M}")



##### For attn_sum #####
print("\nattn_sum:")
S = 16
scale_table['S_attn_sum'] = S

scale_v = activation_scale["v"]
scale_attn_weights = 1.0 / 255
scale_x_attn = activation_scale["x_attn"]

r = (scale_attn_weights * scale_v) / (scale_x_attn)
M = round(r * 2**S)
scale_table["M_attn_sum"] = M
print(f"attn_sum: \t r: {r} \t S: {S} \t M: {M}")



with open("./Data/scales.json", 'w') as file:
    json.dump(scale_table, file, indent=4)