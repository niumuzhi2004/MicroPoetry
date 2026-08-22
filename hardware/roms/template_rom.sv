module template_rom #(
    parameter int POEM_LEN   = 56,
    parameter int N_TEMPLATE = 4,
    parameter int DATA_WIDTH = 4
) (
    input  logic clk,

    // Port A
    input  logic [$clog2(POEM_LEN*N_TEMPLATE)-1:0] addr_a,
    output logic [DATA_WIDTH-1:0] data_a,

    // Port B
    input  logic [$clog2(POEM_LEN*N_TEMPLATE)-1:0] addr_b,
    output logic [DATA_WIDTH-1:0] data_b
);

    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] rom [POEM_LEN*N_TEMPLATE];

    initial begin
        $readmemh("templates.hex", rom);
    end

    always_ff @(posedge clk) begin
        data_a <= rom[addr_a];
        data_b <= rom[addr_b];
    end
    
endmodule