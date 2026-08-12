module tone_rom #(
    parameter int PING_CHARS = 1152,
    parameter int ZE_CHARS   = 1848,
    parameter int DATA_WIDTH = 12
) (
    input  logic clk,
    input  logic tone_sel,

    // Port A
    input  logic [$clog2(ZE_CHARS)-1:0] addr_a,
    output logic [DATA_WIDTH-1:0] data_a,

    // Port B
    input  logic [$clog2(ZE_CHARS)-1:0] addr_b,
    output logic [DATA_WIDTH-1:0] data_b
);

    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] ping_rom [PING_CHARS];
    (* ram_style = "block" *) logic [DATA_WIDTH-1:0] ze_rom [ZE_CHARS];

    initial begin
        $readmemh("tone_ping.hex", ping_rom);
        $readmemh("tone_ze.hex",   ze_rom);
    end

    // Port A logic
    always_ff @(posedge clk) begin
        data_a <= tone_sel ? ping_rom[addr_a] : ze_rom[addr_a];
    end

    // Port B logic
    always_ff @(posedge clk) begin
        data_b <= tone_sel ? ping_rom[addr_b] : ze_rom[addr_b];
    end
    
endmodule