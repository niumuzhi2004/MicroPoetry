module wte_rom #(
    parameter int VOCAB_SIZE = 3005,
    parameter int N_EMBD     = 64,
    parameter int DATA_WIDTH = 8
) (
    input  logic clk,

    // Port A
    input  logic wr_en_a,
    input  logic [DATA_WIDTH-1:0] wr_data_a,
    input  logic [$clog2(VOCAB_SIZE*N_EMBD)-1:0] addr_a,
    output logic [DATA_WIDTH-1:0] rd_data_a,

    // Port B
    input  logic wr_en_b,
    input  logic [DATA_WIDTH-1:0] wr_data_b,
    input  logic [$clog2(VOCAB_SIZE*N_EMBD)-1:0] addr_b,
    output logic [DATA_WIDTH-1:0] rd_data_b
);

    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] rom [0:(VOCAB_SIZE*N_EMBD-1)];

    initial begin
        $readmemh("wte_weight.hex", rom);
    end

    // write ports are kept but unconnected to properly infer dual-port BRAM

    always_ff @(posedge clk) begin
        if (wr_en_a)
            rom[addr_a] <= wr_data_a;
        rd_data_a <= rom[addr_a];
    end

    always_ff @(posedge clk) begin
        if (wr_en_b)
            rom[addr_b] <= wr_data_b;
        rd_data_b <= rom[addr_b];
    end
    
endmodule