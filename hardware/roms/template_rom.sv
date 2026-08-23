module template_rom #(
    parameter int POEM_LEN   = 56,
    parameter int N_TEMPLATE = 4,
    parameter int DATA_WIDTH = 4
) (
    input  logic clk,
    input  logic [$clog2(POEM_LEN*N_TEMPLATE)-1:0] addr,
    output logic [DATA_WIDTH-1:0] data
);

    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] rom [POEM_LEN*N_TEMPLATE];

    initial begin
        $readmemh("templates.hex", rom);
    end

    always_ff @(posedge clk) begin
        data <= rom[addr];
    end
    
endmodule