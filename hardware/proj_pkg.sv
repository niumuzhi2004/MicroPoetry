package proj_pkg;

    // base address definitions
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
    parameter int M_FINAL_SOFTMAX_DIFF = 9787;
    parameter int M_EXP_IDX            = 64;
    parameter int M_RECIP_IDX          = 64;
    parameter int M_SOFTMAX_OUTPUT     = 1020;

    parameter int S_ATTN_SCORE = 16;
    parameter int M_ATTN_SCORE = 274;
    parameter int S_ATTN_SUM   = 16;
    parameter int M_ATTN_SUM   = 265;


    parameter logic signed [15:0] DIFF_LOWER_BOUND = 16'sh8000; // -8 in Q4.12
    parameter logic [15:0] RECIP_LOWER_BOUND       = 16'h8000;  // 1 in Q1.15         


    // typedef enum for matvec
    typedef enum logic [2:0] {  
        ATTN_WQ = 3'b000,
        ATTN_WK = 3'b001,
        ATTN_WV = 3'b010,
        ATTN_WO = 3'b011,
        MLP_FC1 = 3'b100,
        MLP_FC2 = 3'b101,
        LM_HEAD = 3'b110
    } matvec_param_t;

    // typedef enum for norm
    typedef enum logic [1:0] { 
        X_NORM       = 2'b00,
        EMB_NORM     = 2'b01,
        PRE_MLP_NORM = 2'b10,
        FINAL_NORM   = 2'b11
    } norm_param_t;

    // typedef enum for softmax
    typedef enum logic {
        ATTN_SOFTMAX  = 1'b0,
        FINAL_SOFTMAX = 1'b1
    } softmax_param_t;

endpackage