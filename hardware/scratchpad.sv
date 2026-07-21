module scratchpad #(
    parameter int ADDR_WIDTH = 16,
    parameter int DATA_WIDTH = 8
) ( 
    input clk,

    // Port A
    input  logic wr_en_a,
    input  logic [ADDR_WIDTH-1:0] addr_a,
    input  logic [DATA_WIDTH-1:0] wr_data_a,
    output logic [DATA_WIDTH-1:0] rd_data_a,

    // Port B
    input  logic wr_en_b,
    input  logic [ADDR_WIDTH-1:0] addr_b,
    input  logic [DATA_WIDTH-1:0] wr_data_b,
    output logic [DATA_WIDTH-1:0] rd_data_b
);
    
    // Address Range:   0x0000-0xD3FF (53kb)
    // KV-cache:        0x0000-0xBFFF (48kb)
    // Scratchpad:      0xC000-0xC7FF  (2kb)
    // Logits buffer:   0xC800-0xD3FF  (3kb)

    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] ram [0:2**ADDR_WIDTH-1];

    // Port A logic
    always_ff @(posedge clk) begin
        rd_data_a <= ram[addr_a];
        if (wr_en_a)
            ram[addr_a] <= wr_data_a;
    end

    // Port B logic
    always_ff @(posedge clk) begin
        rd_data_b <= ram[addr_b];
        if (wr_en_b)
            ram[addr_b] <= wr_data_b;
    end
    
endmodule