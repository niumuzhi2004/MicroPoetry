"""
Python Instruction Generator

Instruction format:
[ACTUATOR] [PARAM] [LAYER] [HEAD_ID]
[10:7]     [6:4]   [3:2]   [1:0]

ACTUATOR (4 bits): 
attn_score, attn_sum, embed, mask, matvec, norm, sampler, softmax, vecadd, vecmove

PARAM (3 bits):
parameter input to actuators; up to 3 bits

LAYER (2 bits):
neural network layer number; select from 0-3

HEAD_ID (2 bits):
attention head number; select from 0-3
"""

from enum import IntEnum

class Actuator(IntEnum):
    ATTN_SCORE = 0
    ATTN_SUM = 1
    EMBED = 2
    MASK = 3
    MATVEC = 4
    NORM = 5
    SAMPLER = 6
    SOFTMAX = 7
    VECADD = 8
    VECMOVE = 9

class Param(IntEnum):
    # matvec
    ATTN_WQ = 0
    ATTN_WK = 1
    ATTN_WV = 2
    ATTN_WO = 3
    MLP_FC1 = 4
    MLP_FC2 = 5
    LM_HEAD = 6

    # norm
    X_NORM = 0
    EMB_NORM = 1
    PRE_MLP_NORM = 2
    FINAL_NORM = 3

    # softmax
    ATTN_SOFTMAX = 0
    FINAL_SOFTMAX = 1

    # vecadd
    ATTN_VEC_SUM = 0
    MLP_VEC_SUM = 1

    # vecmove
    RELU = 0
    COPY_RESIDUAL_ATTN = 1
    COPY_RESIDUAL_MLP = 2
    COPY_TO_K_CACHE = 3
    COPY_TO_V_CACHE = 4

embed_instructions = [
#   [ACTUATOR]              [PARAM]                     [LAYER] [HEAD_ID]
    [Actuator.EMBED,        0,                          0,      0],
    [Actuator.NORM,         Param.EMB_NORM,             0,      0]
]

def create_layer_instructions(layer):

    layer_instructions = [
    #   [ACTUATOR]              [PARAM]                     [LAYER] [HEAD_ID]
        [Actuator.VECMOVE,      Param.COPY_RESIDUAL_ATTN,   layer,   0],
        [Actuator.NORM,         Param.X_NORM,               layer,   0],
        [Actuator.MATVEC,       Param.ATTN_WQ,              layer,   0],
        [Actuator.MATVEC,       Param.ATTN_WK,              layer,   0],
        [Actuator.MATVEC,       Param.ATTN_WV,              layer,   0],
        [Actuator.VECMOVE,      Param.COPY_TO_K_CACHE,      layer,   0],
        [Actuator.VECMOVE,      Param.COPY_TO_V_CACHE,      layer,   0],
        [Actuator.ATTN_SCORE,   0,                          layer,   0],
        [Actuator.SOFTMAX,      Param.ATTN_SOFTMAX,         layer,   0],
        [Actuator.ATTN_SUM,     0,                          layer,   0],
        [Actuator.ATTN_SCORE,   0,                          layer,   1],
        [Actuator.SOFTMAX,      Param.ATTN_SOFTMAX,         layer,   1],
        [Actuator.ATTN_SUM,     0,                          layer,   1],
        [Actuator.ATTN_SCORE,   0,                          layer,   2],
        [Actuator.SOFTMAX,      Param.ATTN_SOFTMAX,         layer,   2],
        [Actuator.ATTN_SUM,     0,                          layer,   2],
        [Actuator.ATTN_SCORE,   0,                          layer,   3],
        [Actuator.SOFTMAX,      Param.ATTN_SOFTMAX,         layer,   3],
        [Actuator.ATTN_SUM,     0,                          layer,   3],
        [Actuator.MATVEC,       Param.ATTN_WO,              layer,   0],
        [Actuator.VECADD,       Param.ATTN_VEC_SUM,         layer,   0],
        [Actuator.VECMOVE,      Param.COPY_RESIDUAL_MLP,    layer,   0],
        [Actuator.NORM,         Param.PRE_MLP_NORM,         layer,   0],
        [Actuator.MATVEC,       Param.MLP_FC1,              layer,   0],
        [Actuator.VECMOVE,      Param.RELU,                 layer,   0],
        [Actuator.MATVEC,       Param.MLP_FC2,              layer,   0],
        [Actuator.VECADD,       Param.MLP_VEC_SUM,          layer,   0]
    ]

    return layer_instructions

final_instructions = [
    [Actuator.NORM,         Param.FINAL_NORM,           0,      0],
    [Actuator.MATVEC,       Param.LM_HEAD,              0,      0],
    [Actuator.MASK,         0,                          0,      0],
    [Actuator.SOFTMAX,      Param.FINAL_SOFTMAX,        0,      0],
    [Actuator.SAMPLER,      0,                          0,      3]
]

# create program
program = embed_instructions
for i in range(4):
    program.extend(create_layer_instructions(i))
program.extend(final_instructions)

with open("./program.hex", 'w') as file:
    for instr in program:
        actuator, param, layer, head_id = instr[0], instr[1], instr[2], instr[3]
        raw_instr = (actuator << 7) | (param << 4) | (layer << 2) | head_id
        file.write(f"{raw_instr:03x}\n")