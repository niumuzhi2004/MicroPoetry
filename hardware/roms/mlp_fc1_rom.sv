module mlp_fc1_rom #(
    parameter int N_LAYERS   = 4,
    parameter int N_EMBD     = 64,
    parameter int DATA_WIDTH = 8
) (
    input  logic clk,

    // Port A
    input  logic [$clog2(4*N_EMBD*N_EMBD*N_LAYERS)-1:0] addr_a,
    output logic [DATA_WIDTH-1:0] data_a,

    // Port B
    input  logic [$clog2(4*N_EMBD*N_EMBD*N_LAYERS)-1:0] addr_b,
    output logic [DATA_WIDTH-1:0] data_b
);

    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] rom [0:(N_LAYERS*4*N_EMBD*N_EMBD-1)];

    initial begin
        $readmemh("mlp_fc1_weight.hex", rom);
    end

    // Port A logic
    always_ff @(posedge clk) begin
        data_a <= rom[addr_a];
    end

    // Port B logic
    always_ff @(posedge clk) begin
        data_b <= rom[addr_b];
    end
    
endmodule