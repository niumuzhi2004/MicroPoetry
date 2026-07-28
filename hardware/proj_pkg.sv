package proj_pkg;

    // base address definitions
    parameter int KV_CACHE_BASE_ADDR      = 16'h0000;

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
    parameter int M_ATTN_WQ_LAYER_0 = 128;
    parameter int M_ATTN_WQ_LAYER_1 = 133;
    parameter int M_ATTN_WQ_LAYER_2 = 186;
    parameter int M_ATTN_WQ_LAYER_3 = 185;
    parameter int M_ATTN_WK_LAYER_0 = 139;
    parameter int M_ATTN_WK_LAYER_1 = 202;
    parameter int M_ATTN_WK_LAYER_2 = 210;
    parameter int M_ATTN_WK_LAYER_3 = 203;
    parameter int M_ATTN_WV_LAYER_0 = 158;
    parameter int M_ATTN_WV_LAYER_1 = 286;
    parameter int M_ATTN_WV_LAYER_2 = 239;
    parameter int M_ATTN_WV_LAYER_3 = 316;
    parameter int M_ATTN_WO_LAYER_0 = 171;
    parameter int M_ATTN_WO_LAYER_1 = 213;
    parameter int M_ATTN_WO_LAYER_2 = 246;
    parameter int M_ATTN_WO_LAYER_3 = 263;
    parameter int M_MLP_FC1_LAYER_0 = 120;
    parameter int M_MLP_FC1_LAYER_1 = 136;
    parameter int M_MLP_FC1_LAYER_2 = 148;
    parameter int M_MLP_FC1_LAYER_3 = 235;
    parameter int M_MLP_FC2_LAYER_0 = 161;
    parameter int M_MLP_FC2_LAYER_1 = 117;
    parameter int M_MLP_FC2_LAYER_2 = 157;
    parameter int M_MLP_FC2_LAYER_3 = 222;
    parameter int M_LM_HEAD = 225;

    parameter int S_NORM                     = 16;
    parameter int S_MS_REAL                  = 8;
    parameter int M_EMB_NORM_MS_REAL         = 219;
    parameter int M_EMB_NORM_IDX             = 1591;
    parameter int M_EMB_NORM_OUTPUT          = 7;
    parameter int M_X_NORM_LAYER0_MS_REAL    = 311;
    parameter int M_X_NORM_LAYER0123_MS_REAL = 10970;
    parameter int M_X_NORM_LAYER0_OUTPUT     = 6;
    parameter int M_X_NORM_LAYER123_OUTPUT   = 37;
    parameter int M_X_NORM_IDX               = 85; 
    parameter int M_PRE_MLP_NORM_MS_REAL     = 10318;   
    parameter int M_PRE_MLP_NORM_IDX         = 90;
    parameter int M_PRE_MLP_NORM_OUTPUT      = 37;
    parameter int M_FINAL_NORM_MS_REAL       = 10970;
    parameter int M_FINAL_NORM_IDX           = 39;
    parameter int M_FINAL_NORM_OUTPUT        = 42;


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

endpackage