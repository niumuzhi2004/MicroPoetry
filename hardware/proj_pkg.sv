package proj_pkg;

    // scratchpad base address definitions
    parameter int K_CACHE_BASE_ADDR       = 16'h0000;
    parameter int K_LAYER0_BASE_ADDR      = 16'h0000;
    parameter int K_LAYER1_BASE_ADDR      = 16'h1800;
    parameter int K_LAYER2_BASE_ADDR      = 16'h3000;
    parameter int K_LAYER3_BASE_ADDR      = 16'h4800;
    parameter int V_CACHE_BASE_ADDR       = 16'h6000;
    parameter int V_LAYER0_BASE_ADDR      = 16'h6000;
    parameter int V_LAYER1_BASE_ADDR      = 16'h7800;
    parameter int V_LAYER2_BASE_ADDR      = 16'h9000;
    parameter int V_LAYER3_BASE_ADDR      = 16'hA800;
    parameter int KV_CACHE_LAYER_SIZE     = 16'h1800;

    parameter int SCRATCHPAD_BASE_ADDR    = 16'hC000;
    parameter int X_EMBD_BASE_ADDR        = 16'hC000;
    parameter int X_NORM_BASE_ADDR        = 16'hC040;
    parameter int Q_BASE_ADDR             = 16'hC080;
    parameter int K_BASE_ADDR             = 16'hC0C0;
    parameter int V_BASE_ADDR             = 16'hC100;
    parameter int ATTN_LOGITS_BASE_ADDR   = 16'hC140;
    parameter int ATTN_WEIGHTS_BASE_ADDR  = 16'hC2C0;
    parameter int HEAD_OUT_BASE_ADDR      = 16'hC440;
    parameter int POST_WO_BASE_ADDR       = 16'hC480;
    parameter int SUM_RESIDUAL_BASE_ADDR  = 16'hC4C0;
    parameter int POST_MLP_FC1_BASE_ADDR  = 16'hC500;
    parameter int POST_RELU_BASE_ADDR     = 16'hC600;
    parameter int POST_MLP_FC2_BASE_ADDR  = 16'hC700;

    parameter int LOGITS_BUFFER_BASE_ADDR = 16'hC800;


    // rhyme cache base address definitions
    parameter int RHYME_GROUP_BASE_ADDR     = 6'd0;
    parameter int PREV_RHYMES_BASE_ADDR     = 6'd1;
    parameter int GENERATED_COUNT_BASE_ADDR = 6'd6;
    parameter int GENERATED_IDS_BASE_ADDR   = 6'd7;


    // scaling constants, calculated from ./quantization/scaling.py
    parameter int S_EMBED = 8;
    parameter int M_WTE   = 222;
    parameter int M_WPE   = 55;
    
    parameter int S_MATVEC          = 16;
    parameter int M_ATTN_WQ_LAYER_0 = 142;
    parameter int M_ATTN_WQ_LAYER_1 = 147;
    parameter int M_ATTN_WQ_LAYER_2 = 205;
    parameter int M_ATTN_WQ_LAYER_3 = 205;
    parameter int M_ATTN_WK_LAYER_0 = 127;
    parameter int M_ATTN_WK_LAYER_1 = 184;
    parameter int M_ATTN_WK_LAYER_2 = 191;
    parameter int M_ATTN_WK_LAYER_3 = 184;
    parameter int M_ATTN_WV_LAYER_0 = 147;
    parameter int M_ATTN_WV_LAYER_1 = 267;
    parameter int M_ATTN_WV_LAYER_2 = 223;
    parameter int M_ATTN_WV_LAYER_3 = 295;
    parameter int M_ATTN_WO_LAYER_0 = 419;
    parameter int M_ATTN_WO_LAYER_1 = 522;
    parameter int M_ATTN_WO_LAYER_2 = 604;
    parameter int M_ATTN_WO_LAYER_3 = 647;
    parameter int M_MLP_FC1_LAYER_0 = 243;
    parameter int M_MLP_FC1_LAYER_1 = 275;
    parameter int M_MLP_FC1_LAYER_2 = 298;
    parameter int M_MLP_FC1_LAYER_3 = 475;
    parameter int M_MLP_FC2_LAYER_0 = 211;
    parameter int M_MLP_FC2_LAYER_1 = 153;
    parameter int M_MLP_FC2_LAYER_2 = 207;
    parameter int M_MLP_FC2_LAYER_3 = 291;
    parameter int M_LM_HEAD = 426;

    parameter int S_NORM                     = 16;
    parameter int S_MS_REAL                  = 8;
    parameter int M_EMB_NORM_MS_REAL         = 219;
    parameter int M_EMB_NORM_IDX             = 1591;
    parameter int M_EMB_NORM_OUTPUT          = 7;
    parameter int M_X_NORM_LAYER0_MS_REAL    = 311;
    parameter int M_X_NORM_LAYER123_MS_REAL  = 8692;
    parameter int M_X_NORM_LAYER0_OUTPUT     = 7;
    parameter int M_X_NORM_LAYER123_OUTPUT   = 36;
    parameter int M_X_NORM_IDX               = 85; 
    parameter int M_PRE_MLP_NORM_MS_REAL     = 5779;   
    parameter int M_PRE_MLP_NORM_IDX         = 90;
    parameter int M_PRE_MLP_NORM_OUTPUT      = 29;
    parameter int M_FINAL_NORM_MS_REAL       = 8692;
    parameter int M_FINAL_NORM_IDX           = 39;
    parameter int M_FINAL_NORM_OUTPUT        = 38;

    parameter int S_SOFTMAX            = 16;
    parameter int S_SOFTMAX_OUTPUT     = 32;
    parameter int M_ATTN_SOFTMAX_DIFF  = 17963;
    parameter int M_FINAL_SOFTMAX_DIFF = 16312;
    parameter int M_EXP_IDX            = 64;
    parameter int M_RECIP_IDX          = 64;
    parameter int M_SOFTMAX_OUTPUT     = 1020;

    parameter int S_ATTN_SCORE = 16;
    parameter int M_ATTN_SCORE = 274;
    parameter int S_ATTN_SUM   = 16;
    parameter int M_ATTN_SUM   = 265;

    parameter int S_VECADD                       = 15;
    parameter int M_VECADD_ATTN_SCALE_A          = 4629;
    parameter int M_VECADD_ATTN_SCALE_B_LAYER0   = 7596;
    parameter int M_VECADD_ATTN_SCALE_B_LAYER123 = 40187;
    parameter int M_VECADD_MLP_SCALE_A           = 22927;
    parameter int M_VECADD_MLP_SCALE_B           = 26718;

    parameter int S_RELU = 16;
    parameter int M_RELU = 41479;

    parameter int S_REPETITION_PENALTY  = 15;
    parameter int M_RECIP_REPETITION    = 25206;
    parameter int M_MULTIPLY_REPETITION = 42598;


    // constants for softmax
    parameter logic signed [15:0] DIFF_LOWER_BOUND = 16'sh8000; // -8 in Q4.12
    parameter logic [15:0] RECIP_LOWER_BOUND       = 16'h8000;  // 1 in Q1.15

    // token ids for special characters
    parameter int TOKEN_ID_BOS = 0;
    parameter int TOKEN_ID_EOS = 1;
    parameter int TOKEN_ID_SEP = 2;
    parameter int TOKEN_ID_UNK = 3;

    // parameters for LCG random number generation in sampler
    parameter int LCG_PARAM_A = 1664525;
    parameter int LCG_PARAM_C = 1013904223;

    // typedef enum for matvec
    typedef enum logic [2:0] {  
        ATTN_WQ         = 3'b000,
        ATTN_WK         = 3'b001,
        ATTN_WV         = 3'b010,
        ATTN_WO         = 3'b011,
        MLP_FC1         = 3'b100,
        MLP_FC2         = 3'b101,
        LM_HEAD         = 3'b110,
        MATVEC_SEL_NONE = 3'b111
    } matvec_param_t;

    // typedef enum for norm
    typedef enum logic [2:0] { 
        X_NORM        = 3'b000,
        EMB_NORM      = 3'b001,
        PRE_MLP_NORM  = 3'b010,
        FINAL_NORM    = 3'b011,
        NORM_SEL_NONE = 3'b100
    } norm_param_t;

    // typedef enum for softmax
    typedef enum logic [1:0] {
        ATTN_SOFTMAX     = 2'b00,
        FINAL_SOFTMAX    = 2'b01,
        SOFTMAX_SEL_NONE = 2'b10
    } softmax_param_t;

    // typedef enum for vecadd
    typedef enum logic [1:0] {
        ATTN_VEC_SUM    = 2'b00,
        MLP_VEC_SUM     = 2'b01,
        VECADD_SEL_NONE = 2'b10
    } vecadd_param_t;

    // typedef enum for vecmove
    typedef enum logic [2:0] {
        RELU               = 3'b000,
        COPY_RESIDUAL_ATTN = 3'b001,
        COPY_RESIDUAL_MLP  = 3'b010,
        COPY_TO_K_CACHE    = 3'b011,
        COPY_TO_V_CACHE    = 3'b100,
        VECMOVE_SEL_NONE   = 3'b101
    } vecmove_param_t;

    // typedef enum for tone encoding
    typedef enum logic [3:0] {
        ANY  = 4'h0,
        PING = 4'h1,
        ZE   = 4'h2
    } tone_t;

    // typedef enum for actuator select in sequencer
    typedef enum logic [3:0] {
        ATTN_SCORE = 4'd0,
        ATTN_SUM   = 4'd1,
        EMBED      = 4'd2,
        MASK       = 4'd3,
        MATVEC     = 4'd4,
        NORM       = 4'd5,
        SAMPLER    = 4'd6,
        SOFTMAX    = 4'd7,
        VECADD     = 4'd8,
        VECMOVE    = 4'd9,
        SEL_NONE   = 4'd10
    } actuator_sel_t;

endpackage