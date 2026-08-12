module rhyme_rom #(
    parameter int VOCAB_SIZE = 3005,
    parameter int DATA_WIDTH = 8
) (
    input  logic clk,

    // Port A
    input  logic [$clog2(VOCAB_SIZE)-1:0] addr_a,
    output logic [DATA_WIDTH-1:0] data_a,

    // Port B
    input  logic [$clog2(VOCAB_SIZE)-1:0] addr_b,
    output logic [DATA_WIDTH-1:0] data_b
);

    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] rom [VOCAB_SIZE];

    initial begin
        $readmemh("rhyme.hex", rom);
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