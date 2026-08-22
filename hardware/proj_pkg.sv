package proj_pkg;

    // scratchpad base address definitions
    localparam int K_CACHE_BASE_ADDR       = 16'h0000;
    localparam int K_LAYER0_BASE_ADDR      = 16'h0000;
    localparam int K_LAYER1_BASE_ADDR      = 16'h1800;
    localparam int K_LAYER2_BASE_ADDR      = 16'h3000;
    localparam int K_LAYER3_BASE_ADDR      = 16'h4800;
    localparam int V_CACHE_BASE_ADDR       = 16'h6000;
    localparam int V_LAYER0_BASE_ADDR      = 16'h6000;
    localparam int V_LAYER1_BASE_ADDR      = 16'h7800;
    localparam int V_LAYER2_BASE_ADDR      = 16'h9000;
    localparam int V_LAYER3_BASE_ADDR      = 16'hA800;
    localparam int KV_CACHE_LAYER_SIZE     = 16'h1800;

    localparam int SCRATCHPAD_BASE_ADDR    = 16'hC000;
    localparam int X_EMBD_BASE_ADDR        = 16'hC000;
    localparam int X_NORM_BASE_ADDR        = 16'hC040;
    localparam int Q_BASE_ADDR             = 16'hC080;
    localparam int K_BASE_ADDR             = 16'hC0C0;
    localparam int V_BASE_ADDR             = 16'hC100;
    localparam int ATTN_LOGITS_BASE_ADDR   = 16'hC140;
    localparam int ATTN_WEIGHTS_BASE_ADDR  = 16'hC2C0;
    localparam int HEAD_OUT_BASE_ADDR      = 16'hC440;
    localparam int POST_WO_BASE_ADDR       = 16'hC480;
    localparam int SUM_RESIDUAL_BASE_ADDR  = 16'hC4C0;
    localparam int POST_MLP_FC1_BASE_ADDR  = 16'hC500;
    localparam int POST_RELU_BASE_ADDR     = 16'hC600;
    localparam int POST_MLP_FC2_BASE_ADDR  = 16'hC700;

    localparam int LOGITS_BUFFER_BASE_ADDR = 16'hC800;


    // rhyme cache base address definitions
    localparam int RHYME_GROUP_BASE_ADDR     = 6'd0;
    localparam int PREV_RHYMES_BASE_ADDR     = 6'd1;
    localparam int GENERATED_COUNT_BASE_ADDR = 6'd6;
    localparam int GENERATED_IDS_BASE_ADDR   = 6'd7;


    // scaling constants, calculated from ./quantization/scaling.py
    localparam int S_EMBED = 8;
    localparam int M_WTE   = 222;
    localparam int M_WPE   = 55;
    
    localparam int S_MATVEC          = 16;
    localparam int M_ATTN_WQ_LAYER_0 = 142;
    localparam int M_ATTN_WQ_LAYER_1 = 147;
    localparam int M_ATTN_WQ_LAYER_2 = 205;
    localparam int M_ATTN_WQ_LAYER_3 = 205;
    localparam int M_ATTN_WK_LAYER_0 = 127;
    localparam int M_ATTN_WK_LAYER_1 = 184;
    localparam int M_ATTN_WK_LAYER_2 = 191;
    localparam int M_ATTN_WK_LAYER_3 = 184;
    localparam int M_ATTN_WV_LAYER_0 = 147;
    localparam int M_ATTN_WV_LAYER_1 = 267;
    localparam int M_ATTN_WV_LAYER_2 = 223;
    localparam int M_ATTN_WV_LAYER_3 = 295;
    localparam int M_ATTN_WO_LAYER_0 = 419;
    localparam int M_ATTN_WO_LAYER_1 = 522;
    localparam int M_ATTN_WO_LAYER_2 = 604;
    localparam int M_ATTN_WO_LAYER_3 = 647;
    localparam int M_MLP_FC1_LAYER_0 = 243;
    localparam int M_MLP_FC1_LAYER_1 = 275;
    localparam int M_MLP_FC1_LAYER_2 = 298;
    localparam int M_MLP_FC1_LAYER_3 = 475;
    localparam int M_MLP_FC2_LAYER_0 = 211;
    localparam int M_MLP_FC2_LAYER_1 = 153;
    localparam int M_MLP_FC2_LAYER_2 = 207;
    localparam int M_MLP_FC2_LAYER_3 = 291;
    localparam int M_LM_HEAD = 426;

    localparam int S_NORM                     = 16;
    localparam int S_MS_REAL                  = 8;
    localparam int M_EMB_NORM_MS_REAL         = 219;
    localparam int M_EMB_NORM_IDX             = 1591;
    localparam int M_EMB_NORM_OUTPUT          = 7;
    localparam int M_X_NORM_LAYER0_MS_REAL    = 311;
    localparam int M_X_NORM_LAYER123_MS_REAL  = 8692;
    localparam int M_X_NORM_LAYER0_OUTPUT     = 7;
    localparam int M_X_NORM_LAYER123_OUTPUT   = 36;
    localparam int M_X_NORM_IDX               = 85; 
    localparam int M_PRE_MLP_NORM_MS_REAL     = 5779;   
    localparam int M_PRE_MLP_NORM_IDX         = 90;
    localparam int M_PRE_MLP_NORM_OUTPUT      = 29;
    localparam int M_FINAL_NORM_MS_REAL       = 8692;
    localparam int M_FINAL_NORM_IDX           = 39;
    localparam int M_FINAL_NORM_OUTPUT        = 38;

    localparam int S_SOFTMAX            = 16;
    localparam int S_SOFTMAX_OUTPUT     = 32;
    localparam int M_ATTN_SOFTMAX_DIFF  = 17963;
    localparam int M_FINAL_SOFTMAX_DIFF = 16312;
    localparam int M_EXP_IDX            = 64;
    localparam int M_RECIP_IDX          = 64;
    localparam int M_SOFTMAX_OUTPUT     = 1020;

    localparam int S_ATTN_SCORE = 16;
    localparam int M_ATTN_SCORE = 274;
    localparam int S_ATTN_SUM   = 16;
    localparam int M_ATTN_SUM   = 265;

    localparam int S_VECADD                       = 15;
    localparam int M_VECADD_ATTN_SCALE_A          = 4629;
    localparam int M_VECADD_ATTN_SCALE_B_LAYER0   = 7596;
    localparam int M_VECADD_ATTN_SCALE_B_LAYER123 = 40187;
    localparam int M_VECADD_MLP_SCALE_A           = 22927;
    localparam int M_VECADD_MLP_SCALE_B           = 26718;

    localparam int S_RELU = 16;
    localparam int M_RELU = 41479;

    localparam int S_REPETITION_PENALTY  = 15;
    localparam int M_RECIP_REPETITION    = 25206;
    localparam int M_MULTIPLY_REPETITION = 42598;


    // constants for softmax
    localparam logic signed [15:0] DIFF_LOWER_BOUND = 16'sh8000; // -8 in Q4.12
    localparam logic [15:0] RECIP_LOWER_BOUND       = 16'h8000;  // 1 in Q1.15

    // token ids for special characters
    localparam int TOKEN_ID_BOS = 0;
    localparam int TOKEN_ID_EOS = 1;
    localparam int TOKEN_ID_SEP = 2;
    localparam int TOKEN_ID_UNK = 3;

    // localparams for LCG random number generation in sampler
    localparam int LCG_PARAM_A = 1664525;
    localparam int LCG_PARAM_C = 1013904223;

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