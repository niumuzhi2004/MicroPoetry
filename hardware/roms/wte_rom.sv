module wte_rom #(
    parameter int VOCAB_SIZE = 3005,
    parameter int N_EMBD     = 64,
    parameter int DATA_WIDTH = 8
) (
    input  logic clk,
    input  logic [$clog2(VOCAB_SIZE*N_EMBD)-1:0] addr,
    output logic [DATA_WIDTH-1:0] data
);

    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] rom [0:(VOCAB_SIZE*N_EMBD-1)];

    initial begin
        $readmemh("wte_weight.hex", rom);
    end

    always_ff @(posedge clk) begin
        data <= rom[addr];
    end
    
endmodule