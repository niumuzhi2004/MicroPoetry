import proj_pkg::*;

module sampler #(
    parameter int VOCAB_SIZE = 3005,
    parameter int POEM_LEN   = 56,
    parameter int ADDR_WIDTH = 16,
    parameter int DATA_WIDTH = 8
) (
    input  logic clk,
    input  logic rst_n,

    // control logic
    input  logic start,
    output logic done,
    input  logic [1:0] template_id,
    output logic end_poem,
    output logic [$clog2(VOCAB_SIZE)-1:0] gen_id,

    // scratchpad port A for reading odd entries
    input  logic [DATA_WIDTH-1:0] rd_data_a,
    output logic [ADDR_WIDTH-1:0] addr_a,

    // scratchpad port B for reading even entries
    input  logic [DATA_WIDTH-1:0] rd_data_b,
    output logic [ADDR_WIDTH-1:0] addr_b,

    // rhyme cache port to write generated id and rhyming token
    input  logic [11:0] rd_data_c,
    output logic wr_en_c,
    output logic [5:0] addr_c,
    output logic [11:0] wr_data_c,

    // lcg cache port to read and update LCG states
    input  logic [31:0] lcg_state,
    output logic lcg_wr_en,
    output logic [31:0] lcg_new_state,

    // rhyme rom interface
    output logic [$clog2(VOCAB_SIZE)-1:0] rhyme_rom_addr,
    input  logic [7:0] rhyme_rom_data
);

    // flip-flop signals
    logic [ADDR_WIDTH-1:0] odd_addr_d, odd_addr_q;
    logic [ADDR_WIDTH-1:0] even_addr_d, even_addr_q;
    logic [$clog2(VOCAB_SIZE)-1:0] count_d, count_q;
    logic [$clog2(VOCAB_SIZE)-1:0] pos_d, pos_q;
    logic [15:0] acc_d, acc_q;
    logic [15:0] rand_val_d, rand_val_q; // random value in [0, total)
    logic [$clog2(VOCAB_SIZE)-1:0] gen_token_d, gen_token_q;
    logic [2:0] idx_d, idx_q;
    logic [31:0] product, state;

    assign addr_a = odd_addr_d;
    assign addr_b = even_addr_d;
    assign gen_id = gen_token_q;

    // FSM states
    typedef enum logic [3:0] {
        IDLE, SUM, LAST_SUM, LCG, CHECK,
        APPEND1, APPEND2, APPEND3, APPEND4, DONE
    } state_t;

    state_t curr_state, next_state;

    // FSM sequential logic
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            curr_state  <= IDLE;
            odd_addr_q  <= 0;
            even_addr_q <= 0;
            count_q     <= 0;
            pos_q       <= 0;
            acc_q       <= 0;
            rand_val_q  <= 0;
            gen_token_q <= 0;
            idx_q       <= 0;
        end else begin
            curr_state  <= next_state;
            odd_addr_q  <= odd_addr_d;
            even_addr_q <= even_addr_d;
            count_q     <= count_d;
            pos_q       <= pos_d;
            acc_q       <= acc_d;
            rand_val_q  <= rand_val_d;
            gen_token_q <= gen_token_d;
            idx_q       <= idx_d;
        end
    end

    // FSM combinational logic
    always_comb begin

        next_state  = curr_state;
        odd_addr_d  = odd_addr_q;
        even_addr_d = even_addr_q;
        count_d     = count_q;
        pos_d       = pos_q;
        acc_d       = acc_q;
        rand_val_d  = rand_val_q;
        gen_token_d = gen_token_q;
        idx_d       = idx_q;

        done           = 1'b0;
        end_poem       = 1'b0;
        wr_en_c        = 1'b0;
        addr_c         = 0;
        wr_data_c      = 0;
        lcg_wr_en      = 1'b0;
        lcg_new_state  = 0;
        state          = 0;
        product        = 0;
        rhyme_rom_addr = 0;

        case (curr_state)

            IDLE: begin
                if (start) begin
                    odd_addr_d  = LOGITS_BUFFER_BASE_ADDR;
                    even_addr_d = LOGITS_BUFFER_BASE_ADDR + 1;
                    acc_d       = 0;
                    count_d     = 0;
                    idx_d       = 0;
                    next_state  = SUM;
                end
            end

            SUM: begin
                acc_d       = acc_q + rd_data_a + rd_data_b;
                odd_addr_d  = odd_addr_q + 2;
                even_addr_d = even_addr_q + 2;
                count_d     = count_q + 2;
                
                if (count_q == VOCAB_SIZE - 3) 
                    next_state = LAST_SUM;
            end

            LAST_SUM: begin
                acc_d      = acc_q + rd_data_a;
                next_state = LCG;
            end

            LCG: begin
                lcg_wr_en     = 1'b1;
                state         = LCG_PARAM_A * lcg_state + LCG_PARAM_C;
                lcg_new_state = state;
                product       = state[31:16] * acc_q;
                rand_val_d    = product >> 16;
                odd_addr_d    = LOGITS_BUFFER_BASE_ADDR;
                count_d       = 0;
                acc_d         = 0;
                next_state    = CHECK;
            end

            CHECK: begin
                acc_d      = acc_q + rd_data_a;
                count_d    = count_q + 1;
                odd_addr_d = odd_addr_q + 1;

                if (acc_d > rand_val_q) begin
                    gen_token_d = count_q;
                    addr_c      = GENERATED_COUNT_BASE_ADDR;
                    next_state  = APPEND1;
                end else if (count_q >= (VOCAB_SIZE - 1)) begin
                    gen_token_d = VOCAB_SIZE - 1;
                    addr_c      = GENERATED_COUNT_BASE_ADDR;
                    next_state  = APPEND1;
                end
            end

            APPEND1: begin
                wr_en_c    = 1'b1;
                addr_c     = GENERATED_IDS_BASE_ADDR + rd_data_c;
                wr_data_c  = gen_token_q;
                pos_d      = rd_data_c;

                if (((template_id inside {2'd0, 2'd2}) && (rd_data_c == 13)) 
                || ((template_id inside {2'd1, 2'd3}) && (rd_data_c == 6))) begin
                    rhyme_rom_addr = gen_token_q;
                    next_state     = APPEND2;
                end else begin
                    next_state = APPEND4;
                end

                if (template_id inside {2'd0, 2'd2}) begin
                    case (rd_data_c)
                        12'd13: begin
                            idx_d      = 3'd0;
                            next_state = APPEND2;
                        end
                        12'd27: begin
                            idx_d      = 3'd1;
                            next_state = APPEND3;
                        end
                        12'd41: begin
                            idx_d      = 3'd2;
                            next_state = APPEND3;
                        end
                        default: next_state = APPEND4;
                    endcase
                end else begin
                    case (rd_data_c)
                        12'd6: begin
                            idx_d      = 3'd0;
                            next_state = APPEND2;
                        end
                        12'd13: begin
                            idx_d      = 3'd1;
                            next_state = APPEND3;
                        end
                        12'd27: begin
                            idx_d      = 3'd2;
                            next_state = APPEND3;
                        end
                        12'd41: begin
                            idx_d      = 3'd3;
                            next_state = APPEND3;
                        end
                        default: next_state = APPEND4;
                    endcase
                end

            end

            APPEND2: begin
                wr_en_c    = 1'b1;
                addr_c     = RHYME_GROUP_BASE_ADDR;
                wr_data_c  = rhyme_rom_data; // update rhyme group
                next_state = APPEND3;
            end

            APPEND3: begin
                wr_en_c    = 1'b1;
                addr_c     = PREV_RHYMES_BASE_ADDR + idx_q;
                wr_data_c  = gen_token_q; // update rhyming token
                next_state = APPEND4;
            end

            APPEND4: begin
                wr_en_c    = 1'b1;
                addr_c     = GENERATED_COUNT_BASE_ADDR;
                wr_data_c  = pos_q + 1;
                next_state = DONE;
            end

            DONE: begin
                done       = 1'b1;
                end_poem   = (gen_token_q == TOKEN_ID_EOS) || (pos_q >= POEM_LEN - 1);
                next_state = IDLE;
            end
            
            default: next_state = IDLE;

        endcase

    end
    
endmodule