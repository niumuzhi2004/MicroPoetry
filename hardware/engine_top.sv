import proj_pkg::*;

module engine_top #(
    parameter int C_S_AXI_ADDR_WIDTH = 9,
    parameter int C_S_AXI_DATA_WIDTH = 32,

    parameter int LAYER_NUM = 4,
    parameter int N_HEAD = 4,
    parameter int BLOCK_SIZE = 96,
    parameter int DATA_WIDTH = 8,
    parameter int ADDR_WIDTH = 16,
    parameter int N_EMBD = 64,
    parameter int ZE_CHARS = 1848,
    parameter int N_TEMPLATE = 4,
    parameter int POEM_LEN = 56,
    parameter int PROGRAM_LEN = 115,
    parameter int INSTR_WIDTH = 12,
    parameter int TITLE_SIZE  = 12,
    parameter int VOCAB_SIZE  = 3005
) (
    input  logic  S_AXI_ACLK,
    input  logic  S_AXI_ARESETN,
    input  logic  [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR,
    input  logic  S_AXI_AWVALID,
    output logic  S_AXI_AWREADY,
    input  logic  [C_S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA,
    input  logic  [3:0] S_AXI_WSTRB,
    input  logic  S_AXI_WVALID,
    output logic  S_AXI_WREADY,
    input  logic  [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR,
    input  logic  S_AXI_ARVALID,
    output logic  S_AXI_ARREADY,
    output logic  [C_S_AXI_DATA_WIDTH-1:0] S_AXI_RDATA,
    output logic  [1:0] S_AXI_RRESP,
    output logic  S_AXI_RVALID,
    input  logic  S_AXI_RREADY,
    output logic  [1:0] S_AXI_BRESP,
    output logic  S_AXI_BVALID,
    input  logic  S_AXI_BREADY
);

    logic clk, rst_n;
    assign clk = S_AXI_ACLK;
    assign rst_n = S_AXI_ARESETN;

    // attn_score

    // actuator control logic 
    logic start_attn_score;
    logic done_attn_score;
    logic [$clog2(LAYER_NUM)-1:0] layer_attn_score;
    logic [$clog2(N_HEAD)-1:0] head_id_attn_score;
    logic [$clog2(BLOCK_SIZE)-1:0] pos_id_attn_score;

    // scratchpad port A
    logic [DATA_WIDTH-1:0] rd_data_a_attn_score;
    logic wr_en_a_attn_score;
    logic [ADDR_WIDTH-1:0] addr_a_attn_score;
    logic [DATA_WIDTH-1:0] wr_data_a_attn_score;

    // scratchpad port B
    logic [DATA_WIDTH-1:0] rd_data_b_attn_score;
    logic wr_en_b_attn_score;
    logic [ADDR_WIDTH-1:0] addr_b_attn_score;
    logic [DATA_WIDTH-1:0] wr_data_b_attn_score;

    attn_score attn_score_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_attn_score),
        .done(done_attn_score),
        .layer(layer_attn_score),
        .head_id(head_id_attn_score),
        .pos_id(pos_id_attn_score),
        .rd_data_a(rd_data_a_attn_score),
        .wr_en_a(wr_en_a_attn_score),
        .addr_a(addr_a_attn_score),
        .wr_data_a(wr_data_a_attn_score),
        .rd_data_b(rd_data_b_attn_score),
        .wr_en_b(wr_en_b_attn_score),
        .addr_b(addr_b_attn_score),
        .wr_data_b(wr_data_b_attn_score)
    );


    // attn_sum

    // actuator control logic
    logic start_attn_sum;
    logic done_attn_sum;
    logic [$clog2(LAYER_NUM)-1:0] layer_attn_sum;
    logic [$clog2(N_HEAD)-1:0] head_id_attn_sum;
    logic [$clog2(BLOCK_SIZE)-1:0] pos_id_attn_sum;

    // scratchpad port A
    logic [DATA_WIDTH-1:0] rd_data_a_attn_sum;
    logic wr_en_a_attn_sum;
    logic [ADDR_WIDTH-1:0] addr_a_attn_sum;
    logic [DATA_WIDTH-1:0] wr_data_a_attn_sum;

    // scratchpad port B
    logic [DATA_WIDTH-1:0] rd_data_b_attn_sum;
    logic wr_en_b_attn_sum;
    logic [ADDR_WIDTH-1:0] addr_b_attn_sum;
    logic [DATA_WIDTH-1:0] wr_data_b_attn_sum;

    attn_sum attn_sum_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_attn_sum),
        .done(done_attn_sum),
        .layer(layer_attn_sum),
        .head_id(head_id_attn_sum),
        .pos_id(pos_id_attn_sum),
        .rd_data_a(rd_data_a_attn_sum),
        .wr_en_a(wr_en_a_attn_sum),
        .addr_a(addr_a_attn_sum),
        .wr_data_a(wr_data_a_attn_sum),
        .rd_data_b(rd_data_b_attn_sum),
        .wr_en_b(wr_en_b_attn_sum),
        .addr_b(addr_b_attn_sum),
        .wr_data_b(wr_data_b_attn_sum)
    );


    // embed

    // actuator control logic
    logic start_embed;
    logic done_embed;
    logic [$clog2(VOCAB_SIZE)-1:0] token_id_embed;
    logic [$clog2(BLOCK_SIZE)-1:0] pos_id_embed;

    // weight ROM port
    logic [DATA_WIDTH-1:0] wte_data_embed;
    logic [DATA_WIDTH-1:0] wpe_data_embed;
    logic [$clog2(VOCAB_SIZE*N_EMBD)-1:0] wte_addr_embed;
    logic [$clog2(BLOCK_SIZE*N_EMBD)-1:0] wpe_addr_embed;

    // scratchpad port A
    logic wr_en_a_embed;
    logic [ADDR_WIDTH-1:0] addr_a_embed;
    logic [DATA_WIDTH-1:0] wr_data_a_embed;

    embed embed_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_embed),
        .done(done_embed),
        .token_id(token_id_embed),
        .pos_id(pos_id_embed),
        .wte_data(wte_data_embed),
        .wpe_data(wpe_data_embed),
        .wte_addr(wte_addr_embed),
        .wpe_addr(wpe_addr_embed),
        .wr_en(wr_en_a_embed),
        .wr_addr(addr_a_embed),
        .wr_data(wr_data_a_embed)
    );


    // mask

    // actuator control logic
    logic start_mask;
    logic done_mask;
    logic [1:0] template_id_mask;

    // scratchpad port A
    logic [DATA_WIDTH-1:0] rd_data_a_mask;
    logic wr_en_a_mask;
    logic [ADDR_WIDTH-1:0] addr_a_mask;
    logic [DATA_WIDTH-1:0] wr_data_a_mask;

    // scratchpad port B
    logic wr_en_b_mask;
    logic [ADDR_WIDTH-1:0] addr_b_mask;
    logic [DATA_WIDTH-1:0] wr_data_b_mask;

    // rhyme cache port
    logic [11:0] rd_data_c_mask;
    logic wr_en_c_mask;
    logic [5:0] addr_c_mask;
    logic [11:0] wr_data_c_mask;

    // rhyme rom port
    logic [$clog2(VOCAB_SIZE)-1:0] rhyme_rom_addr_a_mask, rhyme_rom_addr_b_mask;
    logic [7:0] rhyme_rom_data_a_mask, rhyme_rom_data_b_mask;

    // tone rom port
    logic tone_sel_mask;
    logic [$clog2(ZE_CHARS)-1:0] tone_rom_addr_a_mask, tone_rom_addr_b_mask;
    logic [11:0] tone_rom_data_a_mask, tone_rom_data_b_mask;

    // template rom port
    logic [$clog2(POEM_LEN*N_TEMPLATE)-1:0] template_rom_addr_mask;
    logic [3:0] template_rom_data_mask;

    mask mask_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_mask),
        .done(done_mask),
        .template_id(template_id_mask),
        .rd_data_a(rd_data_a_mask),
        .wr_en_a(wr_en_a_mask),
        .addr_a(addr_a_mask),
        .wr_data_a(wr_data_a_mask),
        .wr_en_b(wr_en_b_mask),
        .addr_b(addr_b_mask),
        .wr_data_b(wr_data_b_mask),
        .rd_data_c(rd_data_c_mask),
        .wr_en_c(wr_en_c_mask),
        .addr_c(addr_c_mask),
        .wr_data_c(wr_data_c_mask),
        .rhyme_rom_addr_a(rhyme_rom_addr_a_mask),
        .rhyme_rom_addr_b(rhyme_rom_addr_b_mask),
        .rhyme_rom_data_a(rhyme_rom_data_a_mask),
        .rhyme_rom_data_b(rhyme_rom_data_b_mask),
        .tone_sel(tone_sel_mask),
        .tone_rom_addr_a(tone_rom_addr_a_mask),
        .tone_rom_addr_b(tone_rom_addr_b_mask),
        .tone_rom_data_a(tone_rom_data_a_mask),
        .tone_rom_data_b(tone_rom_data_b_mask),
        .template_rom_addr(template_rom_addr_mask),
        .template_rom_data(template_rom_data_mask)
    );


    // matvec

    // actuator control logic
    logic start_matvec;
    logic done_matvec;
    logic [$clog2(LAYER_NUM)-1:0] layer_matvec;
    matvec_param_t param_matvec;
    
    // weight rom ports
    logic [$clog2(VOCAB_SIZE*N_EMBD)-1:0] wrom_addr_a_matvec, wrom_addr_b_matvec;
    logic [DATA_WIDTH-1:0] wrom_data_a_matvec, wrom_data_b_matvec;

    // scratchpad port A
    logic [DATA_WIDTH-1:0] rd_data_a_matvec;
    logic wr_en_a_matvec;
    logic [ADDR_WIDTH-1:0] addr_a_matvec;
    logic [DATA_WIDTH-1:0] wr_data_a_matvec;

    // scratchpad port B
    logic wr_en_b_matvec;
    logic [ADDR_WIDTH-1:0] addr_b_matvec;
    logic [DATA_WIDTH-1:0] wr_data_b_matvec;

    matvec matvec_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_matvec),
        .done(done_matvec),
        .layer(layer_matvec),
        .param(param_matvec),
        .row_odd_addr(wrom_addr_a_matvec),
        .row_even_addr(wrom_addr_b_matvec),
        .row_odd_data(wrom_data_a_matvec),
        .row_even_data(wrom_data_b_matvec),
        .vec_data(rd_data_a_matvec),
        .wr_en_a(wr_en_a_matvec),
        .wr_addr_a(addr_a_matvec),
        .wr_data_a(wr_data_a_matvec),
        .wr_en_b(wr_en_b_matvec),
        .wr_addr_b(addr_b_matvec),
        .wr_data_b(wr_data_b_matvec)
    );


    // norm 

    // actuator control logic
    logic start_norm;
    logic done_norm;
    norm_param_t param_norm;
    logic [$clog2(LAYER_NUM)-1:0] layer_norm;

    // scratchpad port A
    logic [DATA_WIDTH-1:0] rd_data_a_norm;
    logic wr_en_a_norm;
    logic [ADDR_WIDTH-1:0] addr_a_norm;
    logic [DATA_WIDTH-1:0] wr_data_a_norm;

    // scratchpad port B
    logic [DATA_WIDTH-1:0] rd_data_b_norm;
    logic wr_en_b_norm;
    logic [ADDR_WIDTH-1:0] addr_b_norm;
    logic [DATA_WIDTH-1:0] wr_data_b_norm;

    norm norm_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_norm),
        .done(done_norm),
        .param(param_norm),
        .layer(layer_norm),
        .rd_data_a(rd_data_a_norm),
        .wr_en_a(wr_en_a_norm),
        .addr_a(addr_a_norm),
        .wr_data_a(wr_data_a_norm),
        .rd_data_b(rd_data_b_norm),
        .wr_en_b(wr_en_b_norm),
        .addr_b(addr_b_norm),
        .wr_data_b(wr_data_b_norm)
    );

    
    // sampler

    // actuator control logic
    logic start_sampler;
    logic done_sampler;
    logic [1:0] template_id_sampler;
    logic end_poem_sampler;
    logic [$clog2(VOCAB_SIZE)-1:0] gen_id_sampler;

    // scratchpad port A
    logic [DATA_WIDTH-1:0] rd_data_a_sampler;
    logic [ADDR_WIDTH-1:0] addr_a_sampler;

    // scratchpad port B
    logic [DATA_WIDTH-1:0] rd_data_b_sampler;
    logic [ADDR_WIDTH-1:0] addr_b_sampler;

    // rhyme cache port
    logic [11:0] rd_data_c_sampler;
    logic wr_en_c_sampler;
    logic [5:0] addr_c_sampler;
    logic [11:0] wr_data_c_sampler;

    // lcg cache port
    logic [31:0] lcg_state_sampler;
    logic lcg_wr_en_sampler;
    logic [31:0] lcg_new_state_sampler;

    // rhyme rom port
    logic [$clog2(VOCAB_SIZE)-1:0] rhyme_rom_addr_sampler;
    logic [7:0] rhyme_rom_data_sampler;

    sampler sampler_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_sampler),
        .done(done_sampler),
        .template_id(template_id_sampler),
        .end_poem(end_poem_sampler),
        .gen_id(gen_id_sampler),
        .rd_data_a(rd_data_a_sampler),
        .addr_a(addr_a_sampler),
        .rd_data_b(rd_data_b_sampler),
        .addr_b(addr_b_sampler),
        .rd_data_c(rd_data_c_sampler),
        .wr_en_c(wr_en_c_sampler),
        .addr_c(addr_c_sampler),
        .wr_data_c(wr_data_c_sampler),
        .rhyme_rom_addr(rhyme_rom_addr_sampler),
        .rhyme_rom_data(rhyme_rom_data_sampler),
        .lcg_state(lcg_state_sampler),
        .lcg_new_state(lcg_new_state_sampler),
        .lcg_wr_en(lcg_wr_en_sampler)
    );


    // softmax

    // actuator control logic
    logic start_softmax;
    logic done_softmax;
    softmax_param_t param_softmax;
    logic [$clog2(N_HEAD)-1:0] head_id_softmax;
    logic [$clog2(BLOCK_SIZE)-1:0] pos_id_softmax;

    // scratchpad port A
    logic [DATA_WIDTH-1:0] rd_data_a_softmax;
    logic wr_en_a_softmax;
    logic [ADDR_WIDTH-1:0] addr_a_softmax;
    logic [DATA_WIDTH-1:0] wr_data_a_softmax;

    // scratchpad port B
    logic [DATA_WIDTH-1:0] rd_data_b_softmax;
    logic wr_en_b_softmax;
    logic [ADDR_WIDTH-1:0] addr_b_softmax;
    logic [DATA_WIDTH-1:0] wr_data_b_softmax;

    softmax softmax_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_softmax),
        .done(done_softmax),
        .param(param_softmax),
        .head_id(head_id_softmax),
        .pos_id(pos_id_softmax),
        .rd_data_a(rd_data_a_softmax),
        .wr_en_a(wr_en_a_softmax),
        .addr_a(addr_a_softmax),
        .wr_data_a(wr_data_a_softmax),
        .rd_data_b(rd_data_b_softmax),
        .wr_en_b(wr_en_b_softmax),
        .addr_b(addr_b_softmax),
        .wr_data_b(wr_data_b_softmax)
    );


    // vecadd

    // actuator control logic
    logic start_vecadd;
    logic done_vecadd;
    logic [$clog2(LAYER_NUM)-1:0] layer_vecadd;
    vecadd_param_t param_vecadd;

    // scratchpad port A
    logic [DATA_WIDTH-1:0] rd_data_a_vecadd;
    logic wr_en_a_vecadd;
    logic [ADDR_WIDTH-1:0] addr_a_vecadd;
    logic [DATA_WIDTH-1:0] wr_data_a_vecadd;

    // scratchpad port B
    logic [DATA_WIDTH-1:0] rd_data_b_vecadd;
    logic wr_en_b_vecadd;
    logic [ADDR_WIDTH-1:0] addr_b_vecadd;
    logic [DATA_WIDTH-1:0] wr_data_b_vecadd;

    vecadd vecadd_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_vecadd),
        .done(done_vecadd),
        .layer(layer_vecadd),
        .param(param_vecadd),
        .rd_data_a(rd_data_a_vecadd),
        .wr_en_a(wr_en_a_vecadd),
        .addr_a(addr_a_vecadd),
        .wr_data_a(wr_data_a_vecadd),
        .rd_data_b(rd_data_b_vecadd),
        .wr_en_b(wr_en_b_vecadd),
        .addr_b(addr_b_vecadd),
        .wr_data_b(wr_data_b_vecadd)
    );


    // vecmove

    // actuator control logic
    logic start_vecmove;
    logic done_vecmove;
    logic [$clog2(LAYER_NUM)-1:0] layer_vecmove;
    logic [$clog2(BLOCK_SIZE)-1:0] pos_id_vecmove;
    vecmove_param_t param_vecmove;

    // scratchpad port A
    logic [DATA_WIDTH-1:0] rd_data_a_vecmove;
    logic wr_en_a_vecmove;
    logic [ADDR_WIDTH-1:0] addr_a_vecmove;
    logic [DATA_WIDTH-1:0] wr_data_a_vecmove;

    // scratchpad port B
    logic [DATA_WIDTH-1:0] rd_data_b_vecmove;
    logic wr_en_b_vecmove;
    logic [ADDR_WIDTH-1:0] addr_b_vecmove;
    logic [DATA_WIDTH-1:0] wr_data_b_vecmove;

    vecmove vecmove_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_vecmove),
        .done(done_vecmove),
        .layer(layer_vecmove),
        .pos_id(pos_id_vecmove),
        .param(param_vecmove),
        .rd_data_a(rd_data_a_vecmove),
        .wr_en_a(wr_en_a_vecmove),
        .addr_a(addr_a_vecmove),
        .wr_data_a(wr_data_a_vecmove),
        .rd_data_b(rd_data_b_vecmove),
        .wr_en_b(wr_en_b_vecmove),
        .addr_b(addr_b_vecmove),
        .wr_data_b(wr_data_b_vecmove)
    );


    // lcg cache
    logic wr_en_lcg;
    logic [31:0] wr_data_lcg;
    logic [31:0] rd_data_lcg;

    lcg_cache lcg_cache_inst (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en_lcg),
        .wr_data(wr_data_lcg),
        .rd_data(rd_data_lcg)
    );


    // rhyme cache
    logic wr_en_rhyme;
    logic [5:0] addr_rhyme;
    logic [11:0] wr_data_rhyme;
    logic [11:0] rd_data_rhyme;

    rhyme_cache rhyme_cache_inst (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en_rhyme),
        .addr(addr_rhyme),
        .wr_data(wr_data_rhyme),
        .rd_data(rd_data_rhyme)
    );


    // scratchpad
    logic wr_en_a_scratchpad, wr_en_b_scratchpad;
    logic [ADDR_WIDTH-1:0] addr_a_scratchpad, addr_b_scratchpad;
    logic [DATA_WIDTH-1:0] wr_data_a_scratchpad, wr_data_b_scratchpad;
    logic [DATA_WIDTH-1:0] rd_data_a_scratchpad, rd_data_b_scratchpad;

    scratchpad scratchpad_inst (
        .clk(clk),
        .wr_en_a(wr_en_a_scratchpad),
        .addr_a(addr_a_scratchpad),
        .wr_data_a(wr_data_a_scratchpad),
        .rd_data_a(rd_data_a_scratchpad),
        .wr_en_b(wr_en_b_scratchpad),
        .addr_b(addr_b_scratchpad),
        .wr_data_b(wr_data_b_scratchpad),
        .rd_data_b(rd_data_b_scratchpad)
    );


    // attn_wk rom
    logic [DATA_WIDTH-1:0] data_a_attn_wk, data_b_attn_wk;

    attn_wk_rom attn_wk_rom_inst (
        .clk(clk),
        .addr_a(wrom_addr_a_matvec),
        .data_a(data_a_attn_wk),
        .addr_b(wrom_addr_b_matvec),
        .data_b(data_b_attn_wk)
    );

    // attn_wo rom
    logic [DATA_WIDTH-1:0] data_a_attn_wo, data_b_attn_wo;

    attn_wo_rom attn_wo_rom_inst (
        .clk(clk),
        .addr_a(wrom_addr_a_matvec),
        .data_a(data_a_attn_wo),
        .addr_b(wrom_addr_b_matvec),
        .data_b(data_b_attn_wo)
    );

    // attn_wq rom
    logic [DATA_WIDTH-1:0] data_a_attn_wq, data_b_attn_wq;

    attn_wq_rom attn_wq_rom_inst (
        .clk(clk),
        .addr_a(wrom_addr_a_matvec),
        .data_a(data_a_attn_wq),
        .addr_b(wrom_addr_b_matvec),
        .data_b(data_b_attn_wq)
    );

    // attn_wv rom
    logic [DATA_WIDTH-1:0] data_a_attn_wv, data_b_attn_wv;

    attn_wv_rom attn_wv_rom_inst (
        .clk(clk),
        .addr_a(wrom_addr_a_matvec),
        .data_a(data_a_attn_wv),
        .addr_b(wrom_addr_b_matvec),
        .data_b(data_b_attn_wv)
    );

    // mlp_fc1 rom
    logic [DATA_WIDTH-1:0] data_a_mlp_fc1, data_b_mlp_fc1;

    mlp_fc1_rom mlp_fc1_rom_inst (
        .clk(clk),
        .addr_a(wrom_addr_a_matvec),
        .data_a(data_a_mlp_fc1),
        .addr_b(wrom_addr_b_matvec),
        .data_b(data_b_mlp_fc1)
    );

    // mlp_fc2 rom
    logic [DATA_WIDTH-1:0] data_a_mlp_fc2, data_b_mlp_fc2;

    mlp_fc2_rom mlp_fc2_rom_inst (
        .clk(clk),
        .addr_a(wrom_addr_a_matvec),
        .data_a(data_a_mlp_fc2),
        .addr_b(wrom_addr_b_matvec),
        .data_b(data_b_mlp_fc2)
    );

    // rhyme rom
    logic [$clog2(VOCAB_SIZE)-1:0] rhyme_rom_addr_a, rhyme_rom_addr_b;
    logic [DATA_WIDTH-1:0] rhyme_rom_data_a, rhyme_rom_data_b;

    rhyme_rom rhyme_rom_inst (
        .clk(clk),
        .addr_a(rhyme_rom_addr_a),
        .data_a(rhyme_rom_data_a),
        .addr_b(rhyme_rom_addr_b),
        .data_b(rhyme_rom_data_b)
    );

    // template rom
    template_rom template_rom_inst (
        .clk(clk),
        .addr_a(template_rom_addr_mask),
        .data_a(template_rom_data_mask)
    );

    // tone rom
    tone_rom tone_rom_inst (
        .clk(clk),
        .tone_sel(tone_sel_mask),
        .addr_a(tone_rom_addr_a_mask),
        .data_a(tone_rom_data_a_mask),
        .addr_b(tone_rom_addr_b_mask),
        .data_b(tone_rom_data_b_mask)
    );

    // wpe rom
    wpe_rom wpe_rom_inst (
        .clk(clk),
        .addr(wpe_addr_embed),
        .data(wpe_data_embed)
    );

    // wte rom
    logic [$clog2(VOCAB_SIZE*N_EMBD)-1:0] addr_a_wte_rom, addr_b_wte_rom;
    logic [DATA_WIDTH-1:0] data_a_wte_rom, data_b_wte_rom;
    wte_rom wte_rom_inst (
        .clk(clk),
        .addr_a(addr_a_wte_rom),
        .data_a(data_a_wte_rom),
        .addr_b(addr_b_wte_rom),
        .data_b(data_b_wte_rom)
    );


    // program_rom
    logic [$clog2(PROGRAM_LEN)-1:0] program_counter;
    logic [INSTR_WIDTH-1:0] instr;

    program_rom program_rom_inst (
        .clk(clk),
        .addr(program_counter),
        .data(instr)
    );

    // register file
    logic [1:0] template_id_in;
    logic [$clog2(VOCAB_SIZE)-1:0] title [TITLE_SIZE];
    logic [$clog2(TITLE_SIZE)-1:0] title_len;
    logic [31:0] lcg_seed;
    logic poem_start;
    logic poem_end;
    logic reg_wr_en;
    logic [$clog2(BLOCK_SIZE)-1:0] gen_pos;
    logic [$clog2(VOCAB_SIZE)-1:0] gen_token;

    regfile regfile_inst (
        .S_AXI_ACLK(clk),
        .S_AXI_ARESETN(rst_n),
        .S_AXI_AWADDR(S_AXI_AWADDR),
        .S_AXI_AWVALID(S_AXI_AWVALID),
        .S_AXI_AWREADY(S_AXI_AWREADY),
        .S_AXI_WDATA(S_AXI_WDATA),
        .S_AXI_WSTRB(S_AXI_WSTRB),
        .S_AXI_WVALID(S_AXI_WVALID),
        .S_AXI_WREADY(S_AXI_WREADY),
        .S_AXI_ARADDR(S_AXI_ARADDR),
        .S_AXI_ARVALID(S_AXI_ARVALID),
        .S_AXI_ARREADY(S_AXI_ARREADY),
        .S_AXI_RDATA(S_AXI_RDATA),
        .S_AXI_RRESP(S_AXI_RRESP),
        .S_AXI_RVALID(S_AXI_RVALID),
        .S_AXI_RREADY(S_AXI_RREADY),
        .S_AXI_BRESP(S_AXI_BRESP),
        .S_AXI_BVALID(S_AXI_BVALID),
        .S_AXI_BREADY(S_AXI_BREADY),
        .template_id(template_id_in),
        .title(title),
        .title_len(title_len),
        .lcg_seed(lcg_seed),
        .poem_start(poem_start),
        .poem_end(poem_end),
        .reg_wr_en(reg_wr_en),
        .gen_pos(gen_pos),
        .gen_token(gen_token)
    );

    // sequencer
    logic lcg_wr_en_sequencer;
    logic [31:0] lcg_wr_data_sequencer;
    actuator_sel_t actuator_sel;
    logic start_sequencer;
    logic done_sequencer;
    logic poem_done_sequencer;
    logic [$clog2(VOCAB_SIZE)-1:0] gen_id_sequencer;
    logic [$clog2(VOCAB_SIZE)-1:0] token_id_sequencer;
    logic [$clog2(BLOCK_SIZE)-1:0] pos_id_sequencer;
    logic [$clog2(LAYER_NUM)-1:0] layer_sequencer;
    logic [$clog2(N_HEAD)-1:0] head_id_sequencer;
    logic [2:0] param_sequencer;
    logic [1:0] template_id_out_sequencer;

    sequencer sequencer_inst (
        .clk(clk),
        .rst_n(rst_n),
        .program_counter(program_counter),
        .instr(instr),
        .template_id_in(template_id_in),
        .title(title),
        .title_len(title_len),
        .lcg_seed(lcg_seed),
        .poem_start(poem_start),
        .poem_end(poem_end),
        .reg_wr_en(reg_wr_en),
        .gen_pos(gen_pos),
        .gen_token(gen_token),
        .lcg_wr_en(lcg_wr_en_sequencer),
        .lcg_wr_data(lcg_wr_data_sequencer),
        .actuator_sel(actuator_sel),
        .start(start_sequencer),
        .done(done_sequencer),
        .poem_done(poem_done_sequencer),
        .gen_id(gen_id_sequencer),
        .token_id(token_id_sequencer),
        .pos_id(pos_id_sequencer),
        .layer(layer_sequencer),
        .head_id(head_id_sequencer),
        .param(param_sequencer),
        .template_id_out(template_id_out_sequencer)
    );
    
    always_comb begin

        start_attn_score     = 1'b0;
        layer_attn_score     = 0;
        head_id_attn_score   = 0;
        pos_id_attn_score    = 0;
        rd_data_a_attn_score = 0;
        rd_data_b_attn_score = 0;

        start_attn_sum     = 1'b0;
        layer_attn_sum     = 0;
        head_id_attn_sum   = 0;
        pos_id_attn_sum    = 0;
        rd_data_a_attn_sum = 0;
        rd_data_b_attn_sum = 0;

        start_embed      = 1'b0;
        token_id_embed   = 0;
        pos_id_embed     = 0;
        wte_data_embed   = 0;

        start_mask            = 1'b0;
        template_id_mask      = 0;
        rd_data_a_mask        = 0;
        rd_data_c_mask        = 0;
        rhyme_rom_data_a_mask = 0;
        rhyme_rom_data_b_mask = 0;

        start_matvec     = 1'b0;
        layer_matvec     = 0;
        param_matvec     = MATVEC_SEL_NONE;
        rd_data_a_matvec = 0;

        start_norm     = 1'b0;
        param_norm     = NORM_SEL_NONE;
        layer_norm     = 0;
        rd_data_a_norm = 0;
        rd_data_b_norm = 0;

        start_sampler          = 1'b0;
        template_id_sampler    = 0;
        rd_data_a_sampler      = 0;
        rd_data_b_sampler      = 0;
        rd_data_c_sampler      = 0;
        lcg_state_sampler      = 0;
        rhyme_rom_data_sampler = 0;

        start_softmax     = 1'b0;
        param_softmax     = SOFTMAX_SEL_NONE;
        head_id_softmax   = 0;
        pos_id_softmax    = 0;
        rd_data_a_softmax = 0;
        rd_data_b_softmax = 0;

        start_vecadd     = 1'b0;
        layer_vecadd     = 0;
        param_vecadd     = VECADD_SEL_NONE;
        rd_data_a_vecadd = 0;
        rd_data_b_vecadd = 0;

        start_vecmove     = 1'b0;
        layer_vecmove     = 0;
        pos_id_vecmove    = 0;
        param_vecmove     = VECMOVE_SEL_NONE;
        rd_data_a_vecmove = 0;
        rd_data_b_vecmove = 0;

        wr_en_lcg   = lcg_wr_en_sequencer;
        wr_data_lcg = lcg_wr_data_sequencer;

        wr_en_rhyme   = 1'b0;
        addr_rhyme    = 0;
        wr_data_rhyme = 0;

        wr_en_a_scratchpad   = 1'b0;
        addr_a_scratchpad    = 0;
        wr_data_a_scratchpad = 0;
        wr_en_b_scratchpad   = 1'b0;
        addr_b_scratchpad    = 0;
        wr_data_b_scratchpad = 0;

        rhyme_rom_addr_a = 0;
        rhyme_rom_addr_b = 0;

        addr_a_wte_rom = 0;
        addr_b_wte_rom = 0;

        done_sequencer      = 1'b0;
        poem_done_sequencer = 1'b0;
        gen_id_sequencer    = 0;

        case (actuator_sel)
            ATTN_SCORE: begin
                start_attn_score     = start_sequencer;
                done_sequencer       = done_attn_score;
                layer_attn_score     = layer_sequencer;
                head_id_attn_score   = head_id_sequencer;
                pos_id_attn_score    = pos_id_sequencer;
                rd_data_a_attn_score = rd_data_a_scratchpad;
                wr_en_a_scratchpad   = wr_en_a_attn_score;
                addr_a_scratchpad    = addr_a_attn_score;
                wr_data_a_scratchpad = wr_data_a_attn_score;
                rd_data_b_attn_score = rd_data_b_scratchpad;
                wr_en_b_scratchpad   = wr_en_b_attn_score;
                addr_b_scratchpad    = addr_b_attn_score;
                wr_data_b_scratchpad = wr_data_b_attn_score;
            end

            ATTN_SUM: begin
                start_attn_sum       = start_sequencer;
                done_sequencer       = done_attn_sum;
                layer_attn_sum       = layer_sequencer;
                head_id_attn_sum     = head_id_sequencer;
                pos_id_attn_sum      = pos_id_sequencer;
                rd_data_a_attn_sum   = rd_data_a_scratchpad;
                wr_en_a_scratchpad   = wr_en_a_attn_sum;  
                addr_a_scratchpad    = addr_a_attn_sum;
                wr_data_a_scratchpad = wr_data_a_attn_sum;
                rd_data_b_attn_sum   = rd_data_b_scratchpad;
                wr_en_b_scratchpad   = wr_en_b_attn_sum;
                addr_b_scratchpad    = addr_b_attn_sum;
                wr_data_b_scratchpad = wr_data_b_attn_sum;
            end

            EMBED: begin
                start_embed          = start_sequencer;
                done_sequencer       = done_embed;
                token_id_embed       = token_id_sequencer;
                pos_id_embed         = pos_id_sequencer;
                wr_en_a_scratchpad   = wr_en_a_embed;
                addr_a_scratchpad    = addr_a_embed;
                wr_data_a_scratchpad = wr_data_a_embed;
                addr_a_wte_rom       = wte_addr_embed;
                wte_data_embed       = data_a_wte_rom;
            end

            MASK: begin
                start_mask            = start_sequencer;
                done_sequencer        = done_mask;
                template_id_mask      = template_id_out_sequencer;
                rd_data_a_mask        = rd_data_a_scratchpad;
                wr_en_a_scratchpad    = wr_en_a_mask;
                addr_a_scratchpad     = addr_a_mask;
                wr_data_a_scratchpad  = wr_data_a_mask;
                wr_en_b_scratchpad    = wr_en_b_mask;
                addr_b_scratchpad     = addr_b_mask;
                wr_data_b_scratchpad  = wr_data_b_mask;
                rd_data_c_mask        = rd_data_rhyme;
                wr_en_rhyme           = wr_en_c_mask;
                addr_rhyme            = addr_c_mask;
                wr_data_rhyme         = wr_data_c_mask;
                rhyme_rom_addr_a      = rhyme_rom_addr_a_mask;
                rhyme_rom_addr_b      = rhyme_rom_addr_b_mask;
                rhyme_rom_data_a_mask = rhyme_rom_data_a;
                rhyme_rom_data_b_mask = rhyme_rom_data_b;
            end

            MATVEC: begin
                start_matvec         = start_sequencer;
                done_sequencer       = done_matvec;
                layer_matvec         = layer_sequencer;
                param_matvec         = matvec_param_t'(param_sequencer);
                addr_a_wte_rom       = wrom_addr_a_matvec;
                addr_b_wte_rom       = wrom_addr_b_matvec;
                rd_data_a_matvec     = rd_data_a_scratchpad;
                wr_en_a_scratchpad   = wr_en_a_matvec;
                addr_a_scratchpad    = addr_a_matvec;
                wr_data_a_scratchpad = wr_data_a_matvec;
                wr_en_b_scratchpad   = wr_en_b_matvec;
                addr_b_scratchpad    = addr_b_matvec;
                wr_data_b_scratchpad = wr_data_b_matvec;
            end

            NORM: begin
                start_norm           = start_sequencer;
                done_sequencer       = done_norm;
                param_norm           = norm_param_t'(param_sequencer);
                layer_norm           = layer_sequencer;
                rd_data_a_norm       = rd_data_a_scratchpad;
                wr_en_a_scratchpad   = wr_en_a_norm;
                addr_a_scratchpad    = addr_a_norm;
                wr_data_a_scratchpad = wr_data_a_norm;
                rd_data_b_norm       = rd_data_b_scratchpad;
                wr_en_b_scratchpad   = wr_en_b_norm;
                addr_b_scratchpad    = addr_b_norm;
                wr_data_b_scratchpad = wr_data_b_norm;
            end

            SAMPLER: begin
                start_sampler          = start_sequencer;
                done_sequencer         = done_sampler;
                template_id_sampler    = template_id_out_sequencer;
                poem_done_sequencer    = end_poem_sampler;
                gen_id_sequencer       = gen_id_sampler;
                rd_data_a_sampler      = rd_data_a_scratchpad;
                addr_a_scratchpad      = addr_a_sampler;
                rd_data_b_sampler      = rd_data_b_scratchpad;
                addr_b_scratchpad      = addr_b_sampler;
                rd_data_c_sampler      = rd_data_rhyme;
                wr_en_rhyme            = wr_en_c_sampler;
                addr_rhyme             = addr_c_sampler;
                wr_data_rhyme          = wr_data_c_sampler;
                lcg_state_sampler      = rd_data_lcg;
                wr_en_lcg              = lcg_wr_en_sampler;
                wr_data_lcg            = lcg_new_state_sampler;
                rhyme_rom_addr_a       = rhyme_rom_addr_sampler;
                rhyme_rom_data_sampler = rhyme_rom_data_a;
            end

            SOFTMAX: begin
                start_softmax        = start_sequencer;
                done_sequencer       = done_softmax;
                param_softmax        = softmax_param_t'(param_sequencer);
                head_id_softmax      = head_id_sequencer;
                pos_id_softmax       = pos_id_sequencer;
                rd_data_a_softmax    = rd_data_a_scratchpad;
                wr_en_a_scratchpad   = wr_en_a_softmax;
                addr_a_scratchpad    = addr_a_softmax;
                wr_data_a_scratchpad = wr_data_a_softmax;
                rd_data_b_softmax    = rd_data_b_scratchpad;
                wr_en_b_scratchpad   = wr_en_b_softmax;
                addr_b_scratchpad    = addr_b_softmax;
                wr_data_b_scratchpad = wr_data_b_softmax;
            end

            VECADD: begin
                start_vecadd         = start_sequencer;
                done_sequencer       = done_vecadd;
                layer_vecadd         = layer_sequencer;
                param_vecadd         = vecadd_param_t'(param_sequencer);
                rd_data_a_vecadd     = rd_data_a_scratchpad;
                wr_en_a_scratchpad   = wr_en_a_vecadd;
                addr_a_scratchpad    = addr_a_vecadd;
                wr_data_a_scratchpad = wr_data_a_vecadd;
                rd_data_b_vecadd     = rd_data_b_scratchpad;
                wr_en_b_scratchpad   = wr_en_b_vecadd;
                addr_b_scratchpad    = addr_b_vecadd;
                wr_data_b_scratchpad = wr_data_b_vecadd;
            end

            VECMOVE: begin
                start_vecmove        = start_sequencer;
                done_sequencer       = done_vecmove;
                layer_vecmove        = layer_sequencer;
                pos_id_vecmove       = pos_id_sequencer;
                param_vecmove        = vecmove_param_t'(param_sequencer);
                rd_data_a_vecmove    = rd_data_a_scratchpad;
                wr_en_a_scratchpad   = wr_en_a_vecmove;
                addr_a_scratchpad    = addr_a_vecmove;
                wr_data_a_scratchpad = wr_data_a_vecmove;
                rd_data_b_vecmove    = rd_data_b_scratchpad;
                wr_en_b_scratchpad   = wr_en_b_vecmove;
                addr_b_scratchpad    = addr_b_vecmove;
                wr_data_b_scratchpad = wr_data_b_vecmove;
            end

            default: begin
                // do nothing
            end

        endcase

        if (actuator_sel == MATVEC) begin
            case (param_matvec)
                ATTN_WQ: begin
                    wrom_data_a_matvec = data_a_attn_wq;
                    wrom_data_b_matvec = data_b_attn_wq;
                end
                ATTN_WK: begin
                    wrom_data_a_matvec = data_a_attn_wk;
                    wrom_data_b_matvec = data_b_attn_wk;
                end
                ATTN_WV: begin
                    wrom_data_a_matvec = data_a_attn_wv;
                    wrom_data_b_matvec = data_b_attn_wv;
                end
                ATTN_WO: begin
                    wrom_data_a_matvec = data_a_attn_wo;
                    wrom_data_b_matvec = data_b_attn_wo;
                end
                MLP_FC1: begin
                    wrom_data_a_matvec = data_a_mlp_fc1;
                    wrom_data_b_matvec = data_b_mlp_fc1;
                end
                MLP_FC2: begin
                    wrom_data_a_matvec = data_a_mlp_fc2;
                    wrom_data_b_matvec = data_b_mlp_fc2;
                end
                LM_HEAD: begin
                    wrom_data_a_matvec = data_a_wte_rom;
                    wrom_data_b_matvec = data_b_wte_rom;
                end
                default: begin
                    wrom_data_a_matvec = 0;
                    wrom_data_b_matvec = 0;
                end
            endcase
        end else begin
            wrom_data_a_matvec = 0;
            wrom_data_b_matvec = 0;
        end
    end
    
endmodule