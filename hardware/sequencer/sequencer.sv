import proj_pkg::*;

module sequencer #(
    parameter int VOCAB_SIZE  = 3005,
    parameter int BLOCK_SIZE  = 96,
    parameter int TITLE_SIZE  = 20,
    parameter int LCG_WIDTH   = 32,
    parameter int LAYER_NUM   = 4,
    parameter int N_HEAD      = 4,
    parameter int PROGRAM_LEN = 115,
    parameter int INSTR_WIDTH = 12
) (
    input  logic clk,
    input  logic rst_n,

    // program ROM interface
    output logic [$clog2(PROGRAM_LEN)-1:0] program_counter,
    input  logic [INSTR_WIDTH-1:0] instr,

    // register file interface
    input  logic [1:0] template_id_in,
    input  logic [$clog2(VOCAB_SIZE)-1:0] title [TITLE_SIZE],
    input  logic [$clog2(TITLE_SIZE)-1:0] title_len,
    input  logic [LCG_WIDTH-1:0] lcg_seed,
    input  logic poem_start,
    output logic poem_end,
    output logic reg_wr_en,
    output logic [$clog2(BLOCK_SIZE)-1:0] gen_pos,
    output logic [$clog2(VOCAB_SIZE)-1:0] gen_token,

    // lcg cache interface
    output logic lcg_wr_en,
    output logic [LCG_WIDTH-1:0] lcg_wr_data,

    // actuator interface
    output actuator_sel_t actuator_sel,
    output logic start,
    input  logic done,
    input  logic poem_done,
    input  logic [$clog2(VOCAB_SIZE)-1:0] gen_id,
    output logic [$clog2(VOCAB_SIZE)-1:0] token_id,
    output logic [$clog2(BLOCK_SIZE)-1:0] pos_id,
    output logic [$clog2(LAYER_NUM)-1:0] layer,
    output logic [$clog2(N_HEAD)-1:0] head_id,
    output logic [2:0] param,
    output logic [1:0] template_id_out
);

    // internal registers
    logic [$clog2(VOCAB_SIZE)-1:0] token_id_d, token_id_q;
    logic [$clog2(BLOCK_SIZE)-1:0] pos_id_d, pos_id_q;
    logic [$clog2(PROGRAM_LEN)-1:0] program_counter_d, program_counter_q;
    actuator_sel_t actuator_sel_d, actuator_sel_q;

    assign token_id        = token_id_q;
    assign pos_id          = pos_id_q;
    assign template_id_out = template_id_in;
    assign program_counter = program_counter_q;
    assign actuator_sel    = actuator_sel_d;

    // FSM states
    typedef enum logic [2:0] {
        IDLE, TITLE1, TITLE2, BODY1, BODY2, DONE
    } state_t;

    state_t curr_state, next_state;

    // FSM sequential logic
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            curr_state        <= IDLE;
            token_id_q        <= 0;
            pos_id_q          <= 0;
            program_counter_q <= 0;
            actuator_sel_q    <= SEL_NONE;
        end else begin
            curr_state        <= next_state;
            token_id_q        <= token_id_d;
            pos_id_q          <= pos_id_d;
            program_counter_q <= program_counter_d;
            actuator_sel_q    <= actuator_sel_d;
        end
    end

    // FSM combinational logic
    always_comb begin

        next_state        = curr_state;
        token_id_d        = token_id_q;
        pos_id_d          = pos_id_q;
        program_counter_d = program_counter_q;
        actuator_sel_d    = actuator_sel_q;

        poem_end     = 1'b0;
        reg_wr_en    = 1'b0;
        gen_pos      = 0;
        gen_token    = 0;
        lcg_wr_en    = 0;
        lcg_wr_data  = 0;
        start        = 0;
        layer        = 0;
        head_id      = 0;
        param        = 0;

        case (curr_state)

            IDLE: begin
                if (poem_start) begin
                    lcg_wr_en   = 1'b1;
                    lcg_wr_data = lcg_seed;
                    token_id_d  = title[0];
                    next_state  = TITLE1;
                end
            end

            TITLE1: begin
                // decode instruction
                start          = 1'b1;
                actuator_sel_d = actuator_sel_t'(instr[10:7]);
                param          = instr[6:4];
                layer          = instr[3:2];
                head_id        = instr[1:0];
                next_state     = TITLE2;
            end

            TITLE2: begin
                if (done) begin
                    if (program_counter_q == 109) begin
                        program_counter_d = 0;
                        pos_id_d   = pos_id_q + 1;
                        token_id_d = title[pos_id_d];
                        if (pos_id_q == title_len - 2) begin // title priming done
                            next_state = BODY1;
                        end else begin
                            next_state = TITLE1;
                        end
                    end else begin
                        program_counter_d = program_counter_q + 1;
                        next_state = TITLE1;
                    end
                end
            end

            BODY1: begin
                // decode instruction
                start          = 1'b1;
                actuator_sel_d = actuator_sel_t'(instr[10:7]);
                param          = instr[6:4];
                layer          = instr[3:2];
                head_id        = instr[1:0];
                next_state     = BODY2;
            end

            BODY2: begin
                if (done) begin
                    if (program_counter_q == PROGRAM_LEN - 1) begin
                        reg_wr_en = 1'b1;
                        gen_pos   = pos_id_q - (title_len-1);
                        gen_token = gen_id;

                        if (~poem_done) begin
                            program_counter_d = 0;
                            pos_id_d   = pos_id_q + 1;
                            token_id_d = gen_id;
                            next_state = BODY1;
                        end else begin
                            poem_end   = 1'b1;
                            next_state = DONE;
                        end
                    end else begin
                        program_counter_d = program_counter_q + 1;
                        next_state = BODY1;
                    end
                end
            end
            
            default: next_state = IDLE;

        endcase
        
    end

    
endmodule