module program_rom #(
    parameter int PROGRAM_LEN = 115,
    parameter int INSTR_WIDTH = 12
) (
    input  logic clk,
    input  logic [$clog2(PROGRAM_LEN)-1:0] addr,
    output logic [INSTR_WIDTH-1:0] data
);

    logic [INSTR_WIDTH-1:0] rom [PROGRAM_LEN];

    initial begin
        $readmemh("program.hex", rom);
    end

    always_ff @(posedge clk) begin
        data <= rom[addr];
    end
    
endmodule