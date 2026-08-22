import proj_pkg::*;

module regfile #(
    parameter int C_S_AXI_ADDR_WIDTH = 9,
    parameter int C_S_AXI_DATA_WIDTH = 32,

    parameter int VOCAB_SIZE  = 3005,
    parameter int BLOCK_SIZE  = 96,
    parameter int POEM_LEN    = 56,
    parameter int TITLE_SIZE  = 12,
    parameter int LCG_WIDTH   = 32
) (
    // AXI4-Lite bus
    input  logic  S_AXI_ACLK,
    input  logic  S_AXI_ARESETN,
    input  logic  [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR,
    input  logic  S_AXI_AWVALID,
    output logic  S_AXI_AWREADY,
    input  logic  [C_S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA,
    input  logic  [3:0] S_AXI_WSTRB,
    input  logic  S_AXI_WVALID,
    output logic  S_AXI_WREADY,
    input  logic  [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR,
    input  logic  S_AXI_ARVALID,
    output logic  S_AXI_ARREADY,
    output logic  [C_S_AXI_DATA_WIDTH-1:0] S_AXI_RDATA,
    output logic  [1:0] S_AXI_RRESP,
    output logic  S_AXI_RVALID,
    input  logic  S_AXI_RREADY,
    output logic  [1:0] S_AXI_BRESP,
    output logic  S_AXI_BVALID,
    input  logic  S_AXI_BREADY,

    // sequencer interface
    output logic [1:0] template_id,
    output logic [$clog2(VOCAB_SIZE)-1:0] title [TITLE_SIZE],
    output logic [$clog2(TITLE_SIZE)-1:0] title_len,
    output logic [LCG_WIDTH-1:0] lcg_seed,
    output logic poem_start,
    input  logic poem_end,
    input  logic reg_wr_en,
    input  logic [$clog2(BLOCK_SIZE)-1:0] gen_pos,
    input  logic [$clog2(VOCAB_SIZE)-1:0] gen_token
);

    // AXI4-Lite Memory Map

    // Address      Register        Width       State
    // 0x000        template_id     2           Write-only
    // 0x004        title[0]        12          Write-only
    // 0x008        title[1]        12          Write-only
    // ...          ...             ...         ...
    // 0x030        title[11]       12          Write-only
    // 0x034        title_len       5           Write-only
    // 0x038        lcg_seed        32          Write-only
    // 0x03C        poem_start      1           Write-only

    // 0x040        poem_end        1           Read-only
    // 0x044        poem[0]         12          Read-only
    // 0x048        poem[1]         12          Read-only
    // ...          ...             ...         ...
    // 0x120        poem[55]        12          Read-only

    // flip-flop signals
    logic [C_S_AXI_ADDR_WIDTH-1:0] rd_addr_d, rd_addr_q;
    logic [1:0] bresp_d, bresp_q;
    logic [1:0] temp_id_d, temp_id_q;
    logic [$clog2(VOCAB_SIZE)-1:0] title_d [TITLE_SIZE];
    logic [$clog2(VOCAB_SIZE)-1:0] title_q [TITLE_SIZE];
    logic [$clog2(TITLE_SIZE)-1:0] title_len_d, title_len_q;
    logic [LCG_WIDTH-1:0] lcg_seed_d, lcg_seed_q;
    logic poem_start_d, poem_start_q, poem_end_d, poem_end_q;

    logic [$clog2(VOCAB_SIZE)-1:0] poem [BLOCK_SIZE];
    logic [$clog2(BLOCK_SIZE)-1:0] poem_idx;

    assign template_id = temp_id_q;
    assign title       = title_q;
    assign title_len   = title_len_q;
    assign lcg_seed    = lcg_seed_q;
    assign poem_start  = poem_start_q;

    // FSM states
    typedef enum logic [1:0] {
        IDLE, WR, RD
    } state_t;

    state_t curr_state, next_state;

    // FSM sequential logic
    always_ff @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            curr_state   <= IDLE;
            rd_addr_q    <= 0;
            bresp_q      <= 2'b0;
            temp_id_q    <= 2'b0;
            title_len_q  <= 0;
            lcg_seed_q   <= 0;
            poem_start_q <= 0;
            poem_end_q   <= 0;
            foreach (title_q[i])
                title_q[i] <= 0;
        end else begin
            curr_state   <= next_state;
            rd_addr_q    <= rd_addr_d;
            bresp_q      <= bresp_d;
            temp_id_q    <= temp_id_d;
            title_q      <= title_d;
            title_len_q  <= title_len_d;
            lcg_seed_q   <= lcg_seed_d;
            poem_start_q <= poem_start_d;
            poem_end_q   <= poem_end_d;
            if (reg_wr_en)
                poem[gen_pos] <= gen_token;
        end
    end

    // FSM combinational logic
    always_comb begin

        next_state   = curr_state;
        rd_addr_d    = rd_addr_q;
        bresp_d      = bresp_q;
        temp_id_d    = temp_id_q;
        title_d      = title_q;
        title_len_d  = title_len_q;
        lcg_seed_d   = lcg_seed_q;
        poem_start_d = poem_start_q;
        poem_idx     = 0;

        if (poem_start_q == 1'b1)
            poem_start_d = 1'b0;

        S_AXI_AWREADY = 0;
        S_AXI_WREADY  = 0;
        S_AXI_ARREADY = 0;
        S_AXI_RDATA   = 0;
        S_AXI_RRESP   = 0;
        S_AXI_RVALID  = 0;
        S_AXI_BRESP   = 0;
        S_AXI_BVALID  = 0;

        case (curr_state)

            IDLE: begin
                if (S_AXI_AWVALID && S_AXI_WVALID) begin
                    S_AXI_WREADY  = 1'b1;
                    S_AXI_AWREADY = 1'b1;
                    bresp_d       = 2'b0;
                    next_state    = WR;

                    case (S_AXI_AWADDR)
                        9'h000: temp_id_d   = S_AXI_WDATA[1:0];
                        9'h004: title_d[0]  = S_AXI_WDATA[$clog2(VOCAB_SIZE)-1:0];
                        9'h008: title_d[1]  = S_AXI_WDATA[$clog2(VOCAB_SIZE)-1:0];
                        9'h00C: title_d[2]  = S_AXI_WDATA[$clog2(VOCAB_SIZE)-1:0];
                        9'h010: title_d[3]  = S_AXI_WDATA[$clog2(VOCAB_SIZE)-1:0];
                        9'h014: title_d[4]  = S_AXI_WDATA[$clog2(VOCAB_SIZE)-1:0];
                        9'h018: title_d[5]  = S_AXI_WDATA[$clog2(VOCAB_SIZE)-1:0];
                        9'h01C: title_d[6]  = S_AXI_WDATA[$clog2(VOCAB_SIZE)-1:0];
                        9'h020: title_d[7]  = S_AXI_WDATA[$clog2(VOCAB_SIZE)-1:0];
                        9'h024: title_d[8]  = S_AXI_WDATA[$clog2(VOCAB_SIZE)-1:0];
                        9'h028: title_d[9]  = S_AXI_WDATA[$clog2(VOCAB_SIZE)-1:0];
                        9'h02C: title_d[10] = S_AXI_WDATA[$clog2(VOCAB_SIZE)-1:0];
                        9'h030: title_d[11] = S_AXI_WDATA[$clog2(VOCAB_SIZE)-1:0];
                        9'h034: title_len_d = S_AXI_WDATA[$clog2(TITLE_SIZE)-1:0];
                        9'h038: lcg_seed_d  = S_AXI_WDATA[LCG_WIDTH-1:0];
                        9'h03C: poem_start_d = S_AXI_WDATA[0];
                        default: bresp_d = 2'b11;
                    endcase
                end else if (S_AXI_ARVALID) begin
                    S_AXI_ARREADY = 1'b1;
                    rd_addr_d     = S_AXI_ARADDR;
                    next_state    = RD;
                end
            end

            WR: begin
                S_AXI_BVALID = 1'b1;
                S_AXI_BRESP  = bresp_q;
                if (S_AXI_BREADY) 
                    next_state = IDLE;
            end

            RD: begin
                S_AXI_RVALID = 1'b1;

                if (rd_addr_q == 9'h040) begin
                    S_AXI_RDATA = {31'b0, poem_end_q};
                end else if (rd_addr_q >= 9'h044 && rd_addr_q <= 9'h120) begin
                    poem_idx = (rd_addr_q - 9'h044) >> 2;
                    S_AXI_RDATA[$clog2(VOCAB_SIZE)-1:0] = poem[poem_idx];
                end else begin
                    S_AXI_RRESP = 2'b11;
                end

                if (S_AXI_RREADY)
                    next_state = IDLE;
            end

            default: next_state = IDLE;

        endcase

    end

endmodule