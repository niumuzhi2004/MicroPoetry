module wpe_rom #(
    parameter int BLOCK_SIZE = 96,
    parameter int N_EMBD     = 64,
    parameter int DATA_WIDTH = 8
) (
    input  logic clk,
    input  logic [$clog2(BLOCK_SIZE*N_EMBD)-1:0] addr,
    output logic [DATA_WIDTH-1:0] data
);

    logic [DATA_WIDTH-1:0] rom [0:(BLOCK_SIZE*N_EMBD-1)];

    initial begin
        $readmemh("wpe_weight.hex", rom);
    end

    always_ff @(posedge clk) begin
        data <= rom[addr];
    end
    
endmodule