import json
import sys
import numpy as np


def to_signed8(v):
    return v - 256 if v >= 128 else v


def load_hex(path):
    """Read a .hex file of one unsigned value per line."""
    with open(path, 'r') as file:
        return [int(line.strip(), 16) for line in file]


def load_int8_hex(path, shape):
    """Read a .hex file of int8 weights and reshape it."""
    return np.array([to_signed8(v) for v in load_hex(path)]).reshape(shape)


def golden_embed(token_id, pos_id, q_wte, q_wpe, M_WTE, M_WPE, S_GLOBAL):
    """
    token_id, pos_id: ints
    q_wte: (VOCAB_SIZE, N_EMBD) int8 array
    q_wpe: (BLOCK_SIZE, N_EMBD) int8 array
    Returns: list of N_EMBD signed int8 values — matches wr_data exactly.
    """
    tok_emb_int = q_wte[token_id].astype(np.int32)
    pos_emb_int = q_wpe[pos_id].astype(np.int32)

    raw = (tok_emb_int * M_WTE + pos_emb_int * M_WPE) >> S_GLOBAL
    result = np.clip(raw, -127, 127).astype(np.int8)

    return result.tolist()

def golden_norm(x_int, param, layer, luts, M_table, S_NORM=16, S_MS_REAL=8, debug=False):
    n = len(x_int)
    ms = sum(int(v) * int(v) for v in x_int)          # raw INT32 accumulate, matches ADD state

    if param == "x_norm":
        if layer == 0:
            M_ms_real = M_table[f"M_{param}_layer0_ms_real"]
        else:
            M_ms_real = M_table[f"M_{param}_layer123_ms_real"]
    else:
        M_ms_real = M_table[f"M_{param}_ms_real"]                    # select same as always_comb case

    ms_real = (ms * M_ms_real) >> (S_MS_REAL + 6)                                    # >> log2(64), matches ms_real_d
    M_idx = M_table[f"M_{param}_idx"]

    idx = (ms_real * M_idx) >> S_NORM

    idx = max(0, min(idx, 31))                           # clamp to valid LUT range

    y = luts[param][idx]                                 # Q3.13, direct LUT value

    # one NR iteration, matching RSQRT state exactly
    P = ms_real * y * y                                   # Q12.36

    P_scaled = 24576 - (P >> 23)                          # Q2.14

    y_unscaled = y * P_scaled                             # Q5.27

    y = (y_unscaled >> 14) & 0xFFFF                       # Q3.13 — match RTL's [29:14] slice exactly
    assert 0 <= y < 65536, f"y out of Q3.13 range: {y}"

    if param == "x_norm":
        if layer == 0:
            M_output = M_table[f"M_{param}_layer0_output"]
        else:
            M_output = M_table[f"M_{param}_layer123_output"]
    else:
        M_output = M_table[f"M_{param}_output"]
    result = []
    for xi in x_int:
        val = ((xi * y * M_output) + (1 << (S_NORM - 1))) >> S_NORM
        val = max(-127, min(val, 127))                    # match clamp
        result.append(val)

    if debug == True:
        print(f"first ms: {int(x_int[0])**2 + int(x_int[1])**2}")
        print(f"ms: {ms}")
        print(f"ms_real: {ms_real}")
        print(f"P: {P}")
        print(f"P_scaled: {P_scaled}")
        print(f"y_unscaled: {y_unscaled}")
        print(f"y: {y}")
        print(f"M_output: {M_output}")
        print(f"first val: {(x_int[0] * y * M_output) >> S_NORM}")

    return result

def golden_attn_score(q_int, k_cache_int, M_table, debug=False):
    """
    q_int: list of HEAD_DIM signed int8 values (current position's query, one head)
    k_cache_int: list of lists — k_cache_int[t] is HEAD_DIM signed int8 values
                 for cached position t (0 to current position)
    Returns: list of signed int8 attn_logits, one per cached position — matches wr_data_a exactly.
    """
    M_ATTN_SCORE = M_table["M_attn_score"]
    S_ATTN_SCORE = M_table["S_attn_score"]
    head_dim = len(q_int)
    result = []

    for k_vec in k_cache_int:
        acc = sum(q_int[j] * k_vec[j] for j in range(head_dim))   # raw INT32 accumulate, matches SUM state

        raw_val = (acc * M_ATTN_SCORE) >> S_ATTN_SCORE            # arithmetic shift, matches >>>
        clamped = max(-127, min(raw_val, 127))                     # matches WRITE's clamp (once you add one — see note)
        result.append(clamped)

    return result

def golden_attn_sum(attn_weights_int, v_cache_int, M_table, debug=False):
    """
    attn_weights_int: list of T unsigned bytes [0,255] (softmax output for this head)
    v_cache_int: list of T lists, each HEAD_DIM signed int8 values (cached V, one head)
    Returns: list of HEAD_DIM signed int8 values — matches wr_data_a exactly.
    """
    M_ATTN_SUM = M_table["M_attn_sum"]
    S_ATTN_SUM = M_table["S_attn_sum"]

    T = len(attn_weights_int)
    head_dim = len(v_cache_int[0])
    result = []

    for d in range(head_dim):
        acc = sum(attn_weights_int[t] * v_cache_int[t][d] for t in range(T))
        # attn_weights_int[t] is unsigned [0,255], v_cache_int[t][d] is signed —
        # in Python this multiply is already correct since attn_weights_int is
        # stored as a plain non-negative int, matching the RTL's zero-pad fix

        raw_val = (acc * M_ATTN_SUM) >> S_ATTN_SUM
        clamped = max(-127, min(raw_val, 127))
        result.append(clamped)

        if debug:
            print(f"{d}: acc: {acc}, raw_val: {raw_val}, clamped: {clamped}")

    return result


def golden_matvec(x_int8, W_int8_matrix, param, layer, M_table, S_MATVEC=16, debug=False):
    """
    x_int8: list/array of n_in signed int8 values (scratchpad contents)
    W_int8_matrix: (n_out, n_in) int8 array — row j is the weight vector for output j
    param: "attn_wq" | "attn_wk" | "attn_wv" | "attn_wo" | "mlp_fc1" | "mlp_fc2" | "lm_head"
    Returns: list of n_out signed int8 values — matches wr_data exactly.
    """
    n_out, n_in = W_int8_matrix.shape
    assert len(x_int8) == n_in, f"input length {len(x_int8)} != matrix n_in {n_in}"

    # per-layer constant selection, matching the RTL's always_comb case;
    # lm_head is layer-invariant
    if param == "lm_head":
        M = M_table["M_lm_head"]
    else:
        M = M_table[f"M_{param}_layer{layer}"]

    result = []
    for j in range(n_out):
        acc = 0
        for i in range(n_in):
            acc += int(x_int8[i]) * int(W_int8_matrix[j, i])   # raw INT32 accumulate, matches MAC state
        # print(f"acc: {acc}")
        temp = ((acc * M) + (1 << (S_MATVEC - 1))) >> S_MATVEC                            # arithmetic shift, matches >>>
        result.append(max(-127, min(temp, 127)))                # matches clamp
    return result


def golden_vecadd(input_a, input_b, param, layer, M_table):
    """
    input_a, input_b: lists of N_EMBD signed int8 values
    param: "ATTN_VEC_SUM" or "MLP_VEC_SUM"
    layer: int, 0-3 (only matters for ATTN_VEC_SUM)
    Returns: list of N_EMBD signed int8 values — matches wr_data_b exactly.
    """
    S_VECOP = M_table["S_vecop"]

    if param == "ATTN_VEC_SUM":
        M_a = M_table["M_vecop_attn_scale_a"]
        M_b = (M_table["M_vecop_attn_scale_b_layer0"] if layer == 0
               else M_table["M_vecop_attn_scale_b_layer123"])
    elif param == "MLP_VEC_SUM":
        M_a = M_table["M_vecop_mlp_scale_a"]
        M_b = M_table["M_vecop_mlp_scale_b"]
    else:
        raise ValueError(f"unknown param {param}")

    result = []
    for a, b in zip(input_a, input_b):
        product_a = (a * M_a)
        product_b = (b * M_b)
        temp_val = ((product_a + product_b) + (1 << (S_VECOP - 1))) >> S_VECOP;

        if temp_val > 127:
            wr_data = 0x7F
        elif temp_val < -127:
            wr_data = 0x81
        else:
            wr_data = temp_val & 0xFF   # matches temp_val[7:0] truncation

        result.append(to_signed8(wr_data))

    return result

def golden_vecmove(input_vals, param, M_table):
    """
    input_vals: list of signed int8 values
    param: "RELU" or one of the COPY_* variants
    Returns: list of signed int8 values — matches wr_data_b exactly.
    """
    if param == "RELU":
        M_RELU = M_table["M_relu"]
        S_RELU = M_table["S_relu"]

        result = []
        for x in input_vals:
            after_relu = x if x > 0 else 0
            temp_val = (after_relu * M_RELU) >> S_RELU   # both operands >= 0, no sign concern

            if temp_val > 127:
                wr_data = 0x7F
            else:
                wr_data = temp_val & 0xFF   # safe: temp_val is always in [0,127] here, no negative wraparound risk

            result.append(wr_data)
        return result

    else:
        # COPY_RESIDUAL_ATTN, COPY_RESIDUAL_MLP, COPY_TO_K_CACHE, COPY_TO_V_CACHE
        # all identity — no scale conversion, no clamp needed (input is already valid int8)
        return list(input_vals)


def golden_softmax(raw_logits, param, M_table, luts, debug=False):
    """
    raw_logits: list of signed int8 values (the actual scratchpad contents)
    param: "ATTN_SOFTMAX" or "FINAL_SOFTMAX"
    Returns: list of unsigned bytes [0,255], matching wr_data exactly.
    """
    n = len(raw_logits)

    # --- MAX ---
    max_val = max(raw_logits)   # signed int8 comparison, matches $signed() in MAX state
    if debug:
        print(f"max_val: {max_val}")

    # --- per-position DIFF -> EXP1-4 -> TOTAL ---
    M_diff = M_table["M_final_softmax_diff"] if param == "FINAL_SOFTMAX" else M_table["M_attn_softmax_diff"]
    S_diff  = M_table["S_softmax"]
    DIFF_LOWER_BOUND = -0x8000
    M_exp_idx = M_table["M_exp_idx"]
    exp_lut = luts["exp_lut"]   # 33 entries, Q1.15 ints

    exp_vals = []
    total_q = 0   # Q17.15 accumulator

    for i, raw in enumerate(raw_logits):
        raw_diff = raw - max_val                      # signed INT8 units

        if debug:
            print(f"raw_diff: {raw_diff}")
        
        diff_q = (raw_diff * M_diff) >> (S_diff - 12)         # now Q4.12, matches RTL >>>

        if diff_q < DIFF_LOWER_BOUND:
            exp_val = 0
        else:
            shifted_diff = diff_q - DIFF_LOWER_BOUND    # >= 0, Q4.12
            product = shifted_diff * M_exp_idx
            idx = min((product >> 16), 32)              # bits [20:16], clamped
            if idx >= 32:
                exp_val = exp_lut[32]
            else:
                frac = product & 0xFFFF                      # bits [15:0], Q0.16

                lut_s = exp_lut[idx]
                lut_l = exp_lut[idx + 1]
                lut_diff = lut_l - lut_s                     # always >= 0
                interp_product = frac * lut_diff             # Q0.16 x Q1.15 -> Q1.31
                exp_val = lut_s + (interp_product >> 16)     # back to Q1.15

        if debug:
            print(f"idx: {i} - {idx}")
            print(f"exp_val: {i} - {exp_val}")

        exp_vals.append(exp_val)
        total_q += exp_val                                # Q17.15 accumulate

        # print(f"diff_q: {diff_q} \t frac: {frac} \t lut_diff: {lut_diff} \t total: {total_q}")

    if debug:
        print(f"total: {total_q}")

    # --- RECIP0: exponent extraction ---
    # priority-encode bits [21:15] of total_q, subtract 15
    bit_pos = 15
    for b in range(21, 14, -1):
        if (total_q >> b) & 1:
            bit_pos = b
            break
    exp_q = bit_pos - 15

    m_q = (total_q >> exp_q) & 0xFFFF   # Q1.15, keep low 16 bits
    # print(f"m_q: {m_q}")

    # --- RECIP1-3 ---
    M_recip_idx = M_table["M_recip_idx"]
    RECIP_LOWER_BOUND = 0x8000   # 1.0 in Q1.15 = 32768
    recip_lut = luts["total_lut"]   # 33 entries, Q1.15 ints

    shifted_m = m_q - RECIP_LOWER_BOUND

    product = shifted_m * M_recip_idx
    idx = min((product >> 16), 31)
    frac = product & 0xFFFF

    lut_s = recip_lut[idx]
    lut_l = recip_lut[idx + 1]
    lut_diff = lut_l - lut_s                            # always <= 0, must stay signed
    interp_product = frac * lut_diff                    # signed
    recip_val_pre = lut_s + (interp_product >> 16)      # arithmetic shift, Q1.15 = 1/m

    # --- RECIP4: reconstruction ---
    recip_val = recip_val_pre >> exp_q                  # 1/total, Q1.15

    if debug:
        print(f"recip_val: {recip_val}")

    # --- WRITE ---
    M_output = M_table["M_softmax_output"]
    S_output = M_table["S_softmax_output"]

    result = []
    for i, exp_val in enumerate(exp_vals):
        prob = exp_val * recip_val                       # Q1.15 x Q1.15 -> Q2.30
        wr_data = (prob * M_output) >> S_output
        # print(f"{i}: {wr_data}")
        wr_data = max(0, min(wr_data, 255))               # clamp unsigned [0,255]
        result.append(wr_data)

        if debug:
            print(f"{i}: {prob * M_output}")

    return result


def golden_mask(logits_buffer, template_id, pos_id,
                 generated_ids, rhyme_group_cache, prev_rhymes,
                 rhyme_rom, template_rom, ping_ids, ze_ids,
                 M_table, POEM_LEN=56, VOCAB_SIZE=3005):
    """
    logits_buffer: list of 3005 signed int8 values (mutated in place, like vmem)
    generated_ids: list of previously emitted token ids, in order
    rhyme_group_cache: the locked rhyme group id (or None if not yet locked)
    prev_rhymes: list of rhyme-group ids already used at earlier rhyme positions
    rhyme_rom: list of 3005 rhyme-group ids, one per vocab entry
    template_rom: flat list, template_rom[template_id*POEM_LEN + pos] -> ANY/PING/ZE code
    ping_ids, ze_ids: lists of vocab ids belonging to each tone class
    """
    NEG = to_signed8(0x81)   # -127

    # --- BOS / SEP / UNK ---
    logits_buffer[0] = NEG
    logits_buffer[2] = NEG
    logits_buffer[3] = NEG

    # --- REP0-2: repetition penalty over generated_ids[0:pos_id] ---
    pos = len(generated_ids)   # matches rd_data_c from GENERATED_COUNT
    if pos != 0:
        M_RECIP = M_table["M_recip_repetition"]
        M_MULT  = M_table["M_multiply_repetition"]
        S_REP   = M_table["S_repetition"]

        for i in range(pos):
            tid = generated_ids[i]
            raw = logits_buffer[tid]
            if raw >= 0:              # rd_data_a[DATA_WIDTH-1] == 0
                temp_val = (raw * M_RECIP) >> S_REP
            else:
                temp_val = (raw * M_MULT) >> S_REP

            if temp_val > 127:
                wr = 127
            elif temp_val < -127:
                wr = -127
            else:
                wr = to_signed8(temp_val & 0xFF)

            logits_buffer[tid] = wr

    # --- CHECK: rhyme position or tone position? ---
    is_rhyme_pos = (
        (template_id in (0, 2) and pos_id in (27, 41, 55)) or
        (template_id in (1, 3) and pos_id in (13, 27, 41, 55))
    )

    if is_rhyme_pos:
        # --- RHYME1-2: mask every vocab entry not in the locked rhyme group ---
        rhyme_group = rhyme_group_cache
        for tid in range(VOCAB_SIZE):
            if rhyme_rom[tid] != rhyme_group:
                logits_buffer[tid] = NEG

        # --- RHYME3-4: ban rhyme groups already used at earlier rhyme positions ---
        for used_group_tid in prev_rhymes:
            logits_buffer[used_group_tid] = NEG

    else:
        # --- TONE1-2: direct-write the disallowed tone class ---
        tone_code = template_rom[template_id * POEM_LEN + pos_id]
        if tone_code == 1:
            for tid in ze_ids:      # PING required -> ban all ZE ids
                logits_buffer[tid] = NEG
        elif tone_code == 2:
            for tid in ping_ids:    # ZE required -> ban all PING ids
                logits_buffer[tid] = NEG
        # ANY -> no masking

    return logits_buffer

def golden_sampler(logits_buffer, lcg_state, template_id, generated_count,
                    rhyme_rom, TOKEN_ID_EOS, VOCAB_SIZE=3005,
                    LCG_A=1664525, LCG_C=1013904223, debug=False):
    """
    logits_buffer: list of VOCAB_SIZE unsigned bytes [0,255] (post mask, post softmax)
    lcg_state: current 32-bit LCG state (persisted across calls)
    generated_count: number of tokens already generated (before this call)
    rhyme_rom: list of VOCAB_SIZE rhyme-group tags
    Returns: (gen_token, new_lcg_state, rhyme_group_update or None, is_eos)
    """
    # --- SUM: full pass, no out-of-bounds read, odd tail included ---
    total = sum(logits_buffer[:VOCAB_SIZE])   # plain Python sum handles the odd count correctly;
                                                # RTL must explicitly cover the tail element (index 3004)

    # --- LCG: advance state, draw against the REAL total ---
    new_state = (LCG_A * lcg_state + LCG_C) & 0xFFFFFFFF   # mod 2^32, matches free wraparound
    draw16 = (new_state >> 16) & 0xFFFF                     # bits [31:16], matches the corrected slice
    product = draw16 * total
    r = product >> 16                                        # in [0, total)

    # --- CHECK: cumulative walk ---
    running = 0
    gen_token = None
    for tid in range(VOCAB_SIZE):
        running += logits_buffer[tid]

        if debug:
            if logits_buffer[tid] != 0:
                print(f"{tid}: {running}")

        if running > r:
            gen_token = tid
            break
    if gen_token is None:
        gen_token = VOCAB_SIZE - 1   # fallback, shouldn't occur if total/r are computed correctly

    # --- APPEND: state updates ---
    is_eos = (gen_token == TOKEN_ID_EOS)

    rhyme_update = None
    # first-rhyme-position check, matching whichever count->position mapping the RTL settled on
    is_first_rhyme_pos = (
        (template_id in (0, 2) and generated_count == 13) or
        (template_id in (1, 3) and generated_count == 6)
    )
    if is_first_rhyme_pos:
        rhyme_update = rhyme_rom[gen_token]

    new_generated_count = generated_count + 1

    if debug:
        print(f"total: {total}")
        print(f"new_state: {new_state}")
        print(f"product: {product}")
        print(f"rand_val: {r}")

    return gen_token, new_state, rhyme_update, new_generated_count, is_eos


# Model parameters
n_embd = 64     # embedding dimension
n_head = 4      # number of attention heads
n_layer = 4     # number of layers
block_size = 96 # maximum sequence length
vocab_size = 3005 # number of tokens
head_dim = n_embd // n_head # dimension of each head


m_table = {
    "S_emb": 8,
    "M_wte": 222,
    "M_wpe": 55,
    "S_matvec": 16,
    "M_attn_wq_layer0": 142,
    "M_attn_wq_layer1": 147,
    "M_attn_wq_layer2": 205,
    "M_attn_wq_layer3": 205,
    "M_attn_wk_layer0": 127,
    "M_attn_wk_layer1": 184,
    "M_attn_wk_layer2": 191,
    "M_attn_wk_layer3": 184,
    "M_attn_wv_layer0": 147,
    "M_attn_wv_layer1": 267,
    "M_attn_wv_layer2": 223,
    "M_attn_wv_layer3": 295,
    "M_attn_wo_layer0": 419,
    "M_attn_wo_layer1": 522,
    "M_attn_wo_layer2": 604,
    "M_attn_wo_layer3": 647,
    "M_mlp_fc1_layer0": 243,
    "M_mlp_fc1_layer1": 275,
    "M_mlp_fc1_layer2": 298,
    "M_mlp_fc1_layer3": 475,
    "M_mlp_fc2_layer0": 211,
    "M_mlp_fc2_layer1": 153,
    "M_mlp_fc2_layer2": 207,
    "M_mlp_fc2_layer3": 291,
    "M_lm_head": 426,
    "S_norm": 16,
    "S_ms_real": 8,
    "M_emb_norm_ms_real": 219,
    "M_emb_norm_idx": 1591,
    "M_emb_norm_output": 7,
    "M_x_norm_layer0_ms_real": 311,
    "M_x_norm_layer0_output": 7,
    "M_x_norm_layer123_ms_real": 8692,
    "M_x_norm_layer123_output": 36,
    "M_x_norm_idx": 85,
    "M_post_mlp_norm_ms_real": 5779,
    "M_post_mlp_norm_idx": 90,
    "M_post_mlp_norm_output": 29,
    "M_final_norm_ms_real": 8692,
    "M_final_norm_idx": 39,
    "M_final_norm_output": 38,
    "S_softmax": 16,
    "M_attn_softmax_diff": 17963,
    "M_final_softmax_diff": 16312,
    "M_exp_idx": 64,
    "M_recip_idx": 64,
    "S_softmax_output": 32,
    "M_softmax_output": 1020,
    "S_attn_score": 16,
    "M_attn_score": 274,
    "S_attn_sum": 16,
    "M_attn_sum": 265,
    "S_vecop": 15,
    "M_vecop_attn_scale_a": 4629,
    "M_vecop_attn_scale_b_layer0": 7596,
    "M_vecop_attn_scale_b_layer123": 40187,
    "M_vecop_mlp_scale_a": 22927,
    "M_vecop_mlp_scale_b": 26718,
    "S_relu": 16,
    "M_relu": 41479,
    "S_repetition": 15,
    "M_recip_repetition": 25206,
    "M_multiply_repetition": 42598
}


# ROM contents used by golden_mask / golden_sampler
template_rom = load_hex("./Data/templates.hex")
rhyme_rom    = load_hex("./Data/rhyme.hex")
tone_ping    = load_hex("./Data/tone_ping.hex")
tone_ze      = load_hex("./Data/tone_ze.hex")

with open("./Data/softmax.json", 'r') as file:
    softmax_LUTs = json.load(file)

with open("./Data/rmsnorm.json", 'r') as file:
    norm_LUTs = json.load(file)["luts"]

q_wpe = load_int8_hex("./Data/wpe_weight.hex", (block_size, n_embd))
q_wte = load_int8_hex("./Data/wte_weight.hex", (vocab_size, n_embd))

W_attn_wq = load_int8_hex("./Data/attn_wq_weight.hex", (n_layer, n_embd, n_embd))
W_attn_wk = load_int8_hex("./Data/attn_wk_weight.hex", (n_layer, n_embd, n_embd))
W_attn_wv = load_int8_hex("./Data/attn_wv_weight.hex", (n_layer, n_embd, n_embd))
W_attn_wo = load_int8_hex("./Data/attn_wo_weight.hex", (n_layer, n_embd, n_embd))
W_mlp_fc1 = load_int8_hex("./Data/mlp_fc1_weight.hex", (n_layer, 4 * n_embd, n_embd))
W_mlp_fc2 = load_int8_hex("./Data/mlp_fc2_weight.hex", (n_layer, n_embd, 4 * n_embd))

with open("../model/Data/token_id.json", 'r', encoding='utf-8') as file:
    token_id_map = json.load(file)

M_WTE, M_WPE, S_GLOBAL = 222, 55, 8

# token ids reserved in proj_pkg.sv
TOKEN_ID_BOS, TOKEN_ID_EOS, TOKEN_ID_SEP, TOKEN_ID_UNK = 0, 1, 2, 3
POEM_LEN = 56

# register file contents, hardcoded in regfile.sv
title = [TOKEN_ID_BOS]
title_chars = input("Please type the title in Chinese: ")

for character in title_chars:
    if character not in token_id_map:
        print("Error! Character %s is not in the vocabulary.", character)
        sys.exit()
    else:
        title.append(token_id_map.index(character))

title.append(TOKEN_ID_SEP)
title_len   = len(title)
template_id = 0
lcg_seed    = 20260815

# per-poem state, mirroring the rhyme cache and the LCG cache
keys            = [[] for _ in range(n_layer)]
values          = [[] for _ in range(n_layer)]
lcg_state       = lcg_seed
generated_ids   = []
generated_count = 0
rhyme_group     = None
prev_rhymes     = []


def run_layers(token_id, pos_id, debug=False):
    """Program counter 0..109: EMBED, EMB_NORM, then the four transformer
    layers, ending at the last MLP_VEC_SUM. Appends this position's K/V to the
    caches. This is the whole of a TITLE step; a BODY step continues past it."""
    x = golden_embed(token_id, pos_id, q_wte, q_wpe, M_WTE, M_WPE, S_GLOBAL)
    x = golden_norm(x, "emb_norm", 0, norm_LUTs, m_table)

    for li in range(n_layer):
        x_residual = x
        x = golden_norm(x, "x_norm", li, norm_LUTs, m_table)
        q = golden_matvec(x, W_attn_wq[li], "attn_wq", li, m_table)
        k = golden_matvec(x, W_attn_wk[li], "attn_wk", li, m_table)    
        v = golden_matvec(x, W_attn_wv[li], "attn_wv", li, m_table)
        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits  = golden_attn_score(q_h, k_h, m_table)
            attn_weights = golden_softmax(attn_logits, "ATTN_SOFTMAX", m_table, softmax_LUTs)
            head_out = golden_attn_sum(attn_weights, v_h, m_table)
            x_attn.extend(head_out)

        x = golden_matvec(x_attn, W_attn_wo[li], "attn_wo", li, m_table)
        x = golden_vecadd(x, x_residual, "ATTN_VEC_SUM", li, m_table)

        x_residual = x
        x = golden_norm(x, "post_mlp_norm", li, norm_LUTs, m_table)
        x = golden_matvec(x, W_mlp_fc1[li], "mlp_fc1", li, m_table)
        x = golden_vecmove(x, "RELU", m_table)
        x = golden_matvec(x, W_mlp_fc2[li], "mlp_fc2", li, m_table)
        x = golden_vecadd(x, x_residual, "MLP_VEC_SUM", li, m_table)

    return x


# --- TITLE phase (sequencer TITLE1/TITLE2) -----------------------------------
# Each title token runs PC 0..109 only: the layer stack primes the K/V caches
# and the result is discarded. The sequencer leaves TITLE after the token at
# pos_id == title_len - 2, so the last title token is handled as a BODY step.
for pos_id in range(title_len - 1):
    x = run_layers(title[pos_id], pos_id)

# --- BODY phase (sequencer BODY1/BODY2) --------------------------------------
# PC 0..114: the layer stack plus FINAL_NORM, LM_HEAD, MASK, FINAL_SOFTMAX and
# SAMPLER. The RTL repeats this, feeding gen_id back as the next token_id,
# until poem_done; here we stop after the first generated token.

poem = []

for pos_id in range(title_len - 1, title_len - 1 + POEM_LEN*2):
    if pos_id == title_len - 1:
        x = run_layers(title[pos_id], pos_id)
    else:
        x = run_layers(gen_token, pos_id)

    x = golden_norm(x, "final_norm", 0, norm_LUTs, m_table)
    logits = golden_matvec(x, q_wte, "lm_head", 0, m_table)
    logits = golden_mask(logits, template_id, generated_count,
                        generated_ids, rhyme_group, prev_rhymes,
                        rhyme_rom, template_rom, tone_ping, tone_ze,
                        m_table, POEM_LEN=POEM_LEN, VOCAB_SIZE=vocab_size)

    probs = golden_softmax(logits, "FINAL_SOFTMAX", m_table, softmax_LUTs)

    count_before = generated_count
    gen_token, lcg_state, rhyme_update, generated_count, is_eos = golden_sampler(
        probs, lcg_state, template_id, generated_count,
        rhyme_rom, TOKEN_ID_EOS, VOCAB_SIZE=vocab_size)

    generated_ids.append(gen_token)
    if rhyme_update is not None:
        rhyme_group = rhyme_update

    rec_counts = (13, 27, 41) if template_id in (0, 2) else (6, 13, 27, 41)
    if count_before in rec_counts:
        prev_rhymes.append(gen_token)

    # matches the $display in regfile.sv on reg_wr_en
    poem.append(token_id_map[gen_token])
    print(f"gen_pos={pos_id - (title_len - 1)} gen_token={gen_token} ({token_id_map[gen_token]})")

    if gen_token == TOKEN_ID_EOS:
        print("Poem generation complete!")
        break

# print result
title_str_list = [token_id_map[token_id] for token_id in title]
title_str = "".join(title_str_list[1:-1])

print(f"《{title_str}》")
print("".join(poem[:7]), "，", "".join(poem[7:14]), "。")
print("".join(poem[14:21]), "，", "".join(poem[21:28]), "。")
print("".join(poem[28:35]), "，", "".join(poem[35:42]), "。")
print("".join(poem[42:49]), "，", "".join(poem[49:56]), "。")