module rhyme_cache #(
    parameter int ADDR_WIDTH = 6,
    parameter int DATA_WIDTH = 12
) ( 
    input  logic clk,
    input  logic rst_n,
    input  logic wr_en,
    input  logic [ADDR_WIDTH-1:0] addr,
    input  logic [DATA_WIDTH-1:0] wr_data,
    output logic [DATA_WIDTH-1:0] rd_data
);

    logic [DATA_WIDTH-1:0] mem [0:2**ADDR_WIDTH-1];

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            mem[6] <= 0;
        end else begin
            rd_data <= mem[addr];
            if (wr_en)
                mem[addr] <= wr_data;
        end
    end
    
endmodule