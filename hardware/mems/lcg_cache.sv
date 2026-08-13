module lcg_cache #(
    parameter int DATA_WIDTH = 32
) ( 
    input  logic clk,
    input  logic rst_n,
    input  logic wr_en,
    input  logic [DATA_WIDTH-1:0] wr_data,
    output logic [DATA_WIDTH-1:0] rd_data
);

    logic [DATA_WIDTH-1:0] state;

    always_ff @(posedge clk) begin
        if (~rst_n) begin
            state <= 32'hDEADBEEF;
        end else begin
            rd_data <= state;
            if (wr_en)
                state <= wr_data;
        end
    end
    
endmodule