import proj_pkg::*;

module mask #(
    parameter int POEM_LEN   = 56,
    parameter int N_TEMPLATE = 4,
    parameter int PING_CHARS = 1152,
    parameter int ZE_CHARS   = 1848,
    parameter int VOCAB_SIZE = 3005,
    parameter int ADDR_WIDTH = 16,
    parameter int DATA_WIDTH = 8
) (
    input  logic clk,
    input  logic rst_n,

    // control logic
    input  logic start,
    output logic done,
    input  logic [1:0] template_id,

    // scratchpad port A for reading entries
    input  logic [DATA_WIDTH-1:0] rd_data_a,
    output logic wr_en_a,
    output logic [ADDR_WIDTH-1:0] addr_a,
    output logic [DATA_WIDTH-1:0] wr_data_a,

    // scratchpad port B for writing entries
    output logic wr_en_b,
    output logic [ADDR_WIDTH-1:0] addr_b,
    output logic [DATA_WIDTH-1:0] wr_data_b,

    // rhyme cache port to read previous characters & rhyme group
    input  logic [11:0] rd_data_c,
    output logic wr_en_c,
    output logic [5:0] addr_c,
    output logic [11:0] wr_data_c,

    // rhyme rom interface
    output logic [$clog2(VOCAB_SIZE)-1:0] rhyme_rom_addr_a,
    input  logic [7:0] rhyme_rom_data_a,
    output logic [$clog2(VOCAB_SIZE)-1:0] rhyme_rom_addr_b,
    input  logic [7:0] rhyme_rom_data_b,

    // tone rom interface
    output logic tone_sel,
    output logic [$clog2(ZE_CHARS)-1:0] tone_rom_addr_a,
    input  logic [11:0] tone_rom_data_a,
    output logic [$clog2(ZE_CHARS)-1:0] tone_rom_addr_b,
    input  logic [11:0] tone_rom_data_b,

    // template rom interface
    output logic [$clog2(POEM_LEN*N_TEMPLATE)-1:0] template_rom_addr,
    input  logic [3:0] template_rom_data
);

    // flip-flop signals
    logic [ADDR_WIDTH-1:0] rd_addr_d, rd_addr_q;
    logic [$clog2(VOCAB_SIZE)-1:0] count_d, count_q;
    logic [$clog2(VOCAB_SIZE)-1:0] pos_d, pos_q;
    logic [7:0] rhyme_group_d, rhyme_group_q;
    logic [2:0] prev_rhyme_count_d, prev_rhyme_count_q;
    logic tone_sel_d, tone_sel_q;
    logic signed [31:0] temp_val;
    logic [$clog2(ZE_CHARS)-1:0] total_tone_ids_d, total_tone_ids_q;

    assign addr_a   = rd_addr_d;
    assign tone_sel = tone_sel_d;

    // to avoid issues with over-running
    logic is_second_valid_rhyme, is_second_valid_tone;
    assign is_second_valid_rhyme = ((count_q + 1) < VOCAB_SIZE);
    assign is_second_valid_tone  = ((count_q + 1) < total_tone_ids_q);

    // FSM states
    typedef enum logic [3:0] {
        IDLE, BOS, SEP, UNK, REP0, REP1, REP2, CHECK, 
        RHYME1, RHYME2, RHYME3, RHYME4, TONE1, TONE2, DONE
    } state_t;

    state_t curr_state, next_state;

    // FSM sequential logic
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            curr_state         <= IDLE;
            rd_addr_q          <= 0;
            count_q            <= 0;
            pos_q              <= 0;
            rhyme_group_q      <= 0;
            prev_rhyme_count_q <= 0;
            total_tone_ids_q   <= 0;
            tone_sel_q         <= 0;
        end else begin
            curr_state         <= next_state;
            rd_addr_q          <= rd_addr_d;
            count_q            <= count_d;
            pos_q              <= pos_d;
            rhyme_group_q      <= rhyme_group_d;
            prev_rhyme_count_q <= prev_rhyme_count_d;
            total_tone_ids_q   <= total_tone_ids_d;
            tone_sel_q         <= tone_sel_d;
        end
    end

    // FSM combinational logic
    always_comb begin
        
        next_state         = curr_state;
        rd_addr_d          = rd_addr_q;
        count_d            = count_q;
        pos_d              = pos_q;
        rhyme_group_d      = rhyme_group_q;
        prev_rhyme_count_d = prev_rhyme_count_q;
        total_tone_ids_d   = total_tone_ids_q;
        tone_sel_d         = tone_sel_q;

        done      = 1'b0;
        wr_en_a   = 1'b0;
        wr_en_b   = 1'b0;
        wr_en_c   = 1'b0;
        addr_b    = 0;
        addr_c    = 0;
        wr_data_a = 0;
        wr_data_b = 0;
        wr_data_c = 0;

        rhyme_rom_addr_a  = 0;
        rhyme_rom_addr_b  = 0;
        tone_rom_addr_a   = 0;
        tone_rom_addr_b   = 0;
        template_rom_addr = 0;

        case (curr_state)

            IDLE: begin
                if (start)
                    next_state = BOS;
            end

            BOS: begin
                wr_en_b    = 1'b1;
                addr_b     = LOGITS_BUFFER_BASE_ADDR + TOKEN_ID_BOS;
                wr_data_b  = 8'h81;  // -127
                next_state = SEP;
            end

            SEP: begin
                wr_en_b    = 1'b1;
                addr_b     = LOGITS_BUFFER_BASE_ADDR + TOKEN_ID_SEP;
                wr_data_b  = 8'h81;
                next_state = UNK;
            end

            UNK: begin
                wr_en_b    = 1'b1;
                addr_b     = LOGITS_BUFFER_BASE_ADDR + TOKEN_ID_UNK;
                wr_data_b  = 8'h81;
                addr_c     = GENERATED_COUNT_BASE_ADDR;
                next_state = REP0;
            end

            REP0: begin
                count_d    = 0;
                pos_d      = rd_data_c;
                addr_c     = GENERATED_IDS_BASE_ADDR;
                if (pos_d == 0)
                    next_state = CHECK;
                else
                    next_state = REP1;
            end

            REP1: begin
                rd_addr_d  = LOGITS_BUFFER_BASE_ADDR + rd_data_c; // read original logit
                next_state = REP2;
            end

            REP2: begin
                wr_en_b   = 1'b1;
                addr_b    = rd_addr_q;

                if (rd_data_a[DATA_WIDTH-1] == 1'b0)
                    temp_val = ($signed(rd_data_a) * $signed(M_RECIP_REPETITION)) >>> S_REPETITION_PENALTY;
                else 
                    temp_val = ($signed(rd_data_a) * $signed(M_MULTIPLY_REPETITION)) >>> S_REPETITION_PENALTY;
                
                if (temp_val > 8'sd127)
                    wr_data_b = 8'sd127;
                else if (temp_val < -8'sd127)
                    wr_data_b = -8'sd127;
                else
                    wr_data_b = temp_val[7:0];

                count_d = count_q + 1;

                if (count_q < pos_q - 1) begin
                    addr_c     = GENERATED_IDS_BASE_ADDR + count_d;
                    next_state = REP1;
                end else begin
                    wr_en_c    = 1'b1;
                    addr_c     = GENERATED_COUNT_BASE_ADDR;
                    wr_data_c  = pos_q + 1;
                    next_state = CHECK;
                end
            end

            CHECK: begin
                count_d = 0;

                if (((template_id inside {2'd0, 2'd2}) && (pos_q inside {27, 41, 55})) 
                || ((template_id inside {2'd1, 2'd3}) && (pos_q inside {13, 27, 41, 55}))) begin
                    addr_c     = RHYME_GROUP_BASE_ADDR;
                    next_state = RHYME1;
                end else begin
                    template_rom_addr = template_id * POEM_LEN + pos_q;
                    next_state        = TONE1;
                end
            end

            RHYME1: begin
                rhyme_group_d    = rd_data_c;
                rhyme_rom_addr_a = 0;
                rhyme_rom_addr_b = 1;
                next_state       = RHYME2;
            end

            RHYME2: begin
                if (rhyme_rom_data_a != rhyme_group_q) begin
                    wr_en_a   = 1'b1;
                    rd_addr_d = LOGITS_BUFFER_BASE_ADDR + count_q;
                    wr_data_a = 8'h81;
                end
                if (rhyme_rom_data_b != rhyme_group_q) begin
                    wr_en_b   = is_second_valid_rhyme;
                    addr_b    = LOGITS_BUFFER_BASE_ADDR + count_q + 1;
                    wr_data_b = 8'h81;
                end

                count_d = count_q + 2;

                if (count_q < VOCAB_SIZE - 2) begin
                    rhyme_rom_addr_a = count_d;
                    rhyme_rom_addr_b = count_d + 1;
                    next_state       = RHYME2;
                end else begin
                    count_d    = 0;
                    next_state = RHYME3;
                end
            end

            RHYME3: begin
                if (template_id inside {2'd0, 2'd2}) begin
                    if (pos_q == 27) prev_rhyme_count_d = 1;
                    if (pos_q == 41) prev_rhyme_count_d = 2;
                    if (pos_q == 55) prev_rhyme_count_d = 3;
                end else begin
                    if (pos_q == 13) prev_rhyme_count_d = 1;
                    if (pos_q == 27) prev_rhyme_count_d = 2;
                    if (pos_q == 41) prev_rhyme_count_d = 3;
                    if (pos_q == 55) prev_rhyme_count_d = 4;
                end

                addr_c     = PREV_RHYMES_BASE_ADDR;
                next_state = RHYME4;
            end

            RHYME4: begin
                wr_en_b    = 1'b1;
                addr_b     = LOGITS_BUFFER_BASE_ADDR + rd_data_c; // ban repeated rhymes
                wr_data_b  = 8'h81;
                count_d    = count_q + 1;

                if (count_q < prev_rhyme_count_q - 1) begin
                    addr_c     = PREV_RHYMES_BASE_ADDR + count_d;
                    next_state = RHYME4;
                end else begin
                    next_state = DONE;
                end
            end

            TONE1: begin
                case (template_rom_data)
                    ANY:  next_state = DONE;
                    PING: begin
                        tone_sel_d       = 1'b0;
                        tone_rom_addr_a  = 0;
                        tone_rom_addr_b  = 1;
                        total_tone_ids_d = ZE_CHARS;
                        next_state       = TONE2;
                    end
                    ZE: begin
                        tone_sel_d       = 1'b1;
                        tone_rom_addr_a  = 0;
                        tone_rom_addr_b  = 1;
                        total_tone_ids_d = PING_CHARS;
                        next_state       = TONE2;
                    end
                    default: next_state = DONE;
                endcase
            end

            TONE2: begin
                wr_en_a   = 1'b1;
                rd_addr_d = LOGITS_BUFFER_BASE_ADDR + tone_rom_data_a;
                wr_data_a = 8'h81;

                wr_en_b   = is_second_valid_tone;
                addr_b    = LOGITS_BUFFER_BASE_ADDR + tone_rom_data_b;
                wr_data_b = 8'h81;

                count_d   = count_q + 2;

                if (count_q < total_tone_ids_q - 2) begin
                    tone_rom_addr_a = count_d;
                    tone_rom_addr_b = count_d + 1;
                    next_state      = TONE2;
                end else begin
                    next_state = DONE;
                end
            end

            DONE: begin
                done       = 1'b1;
                next_state = IDLE;
            end
            
            default: next_state = IDLE;

        endcase

    end
    
endmodule