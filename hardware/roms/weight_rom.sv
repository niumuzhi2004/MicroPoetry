import proj_pkg::*;

module weight_rom #(
    parameter int N_BANKS,
    parameter int BANK_DEPTH = 1024,
    parameter int WORD_WIDTH = 32,
    parameter matvec_param_t WEIGHT_NAME
) (
    input  logic clk,

    // Port As
    input  logic                            wr_en_a     [N_BANKS],
    input  logic [WORD_WIDTH-1:0]           wr_data_a   [N_BANKS],
    input  logic [$clog2(BANK_DEPTH)-1:0]   addr_a      [N_BANKS],
    output logic [WORD_WIDTH-1:0]           rd_data_a   [N_BANKS],

    // Port Bs
    input  logic                            wr_en_b     [N_BANKS],
    input  logic [WORD_WIDTH-1:0]           wr_data_b   [N_BANKS],
    input  logic [$clog2(BANK_DEPTH)-1:0]   addr_b      [N_BANKS],
    output logic [WORD_WIDTH-1:0]           rd_data_b   [N_BANKS]
);

    genvar i;
    generate
        for (i=0; i<N_BANKS; ++i) begin
            (* ram_style = "block" *) logic [WORD_WIDTH-1:0] rom [BANK_DEPTH];

            initial begin
                case (WEIGHT_NAME)
                    ATTN_WK: $readmemh($sformatf("attn_wk_weight_%0d.hex", i), rom);
                    ATTN_WO: $readmemh($sformatf("attn_wo_weight_%0d.hex", i), rom);
                    ATTN_WQ: $readmemh($sformatf("attn_wq_weight_%0d.hex", i), rom);
                    ATTN_WV: $readmemh($sformatf("attn_wv_weight_%0d.hex", i), rom);
                    MLP_FC1: $readmemh($sformatf("mlp_fc1_weight_%0d.hex", i), rom);
                    MLP_FC2: $readmemh($sformatf("mlp_fc2_weight_%0d.hex", i), rom);
                    LM_HEAD: $readmemh($sformatf("wte_weight_%0d.hex", i), rom);
                endcase
            end
            
            // write ports are kept but unconnected to properly infer dual-port BRAM
            always_ff @(posedge clk) begin
                if (wr_en_a[i])
                    rom[addr_a[i]] <= wr_data_a[i];
                rd_data_a[i] <= rom[addr_a[i]];
            end

            always_ff @(posedge clk) begin
                if (wr_en_b[i])
                    rom[addr_b[i]] <= wr_data_b[i];
                rd_data_b[i] <= rom[addr_b[i]];
            end

        end
    endgenerate
    
endmodule