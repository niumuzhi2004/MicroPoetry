import proj_pkg::*;

module regfile #(
    parameter int VOCAB_SIZE  = 3005,
    parameter int BLOCK_SIZE  = 96,
    parameter int TITLE_SIZE  = 20,
    parameter int LCG_WIDTH   = 32
) (
    input  logic clk,
    input  logic rst_n,

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

    logic [$clog2(VOCAB_SIZE)-1:0] poem [BLOCK_SIZE];

    always_ff @(posedge clk) begin
        template_id <= 0;
        title[0]    <= TOKEN_ID_BOS;
        title[1]    <= 352;
        title[2]    <= 199;
        title[3]    <= TOKEN_ID_SEP;
        title_len   <= 4;
        lcg_seed    <= 20260815;

        if (reg_wr_en) begin
            poem[gen_pos] <= gen_token;
            $display("t=%0t: gen_pos=%0d gen_token=%0d", $time, gen_pos, gen_token);
        end
        if (poem_end) begin
            $display("t=%0t: POEM COMPLETE", $time);
        end
    end

endmodule