import proj_pkg::*;

module softmax #(
    parameter int N_EMBD     = 64,
    parameter int VOCAB_SIZE = 3005,
    parameter int BLOCK_SIZE = 96,
    parameter int ADDR_WIDTH = 16,
    parameter int DATA_WIDTH = 8
) (
    input  logic clk,
    input  logic rst_n,

    // control logic
    input  logic start,
    output logic done,
    input  softmax_param_t param,

    // scratchpad port A for reading and writing odd entries
    input  logic [DATA_WIDTH-1:0] rd_data_a,
    output logic wr_en_a,
    output logic [ADDR_WIDTH-1:0] addr_a,
    output logic [DATA_WIDTH-1:0] wr_data_a,

    // scratchpad port B for reading and writing even entries
    input  logic [DATA_WIDTH-1:0] rd_data_b,
    output logic wr_en_b,
    output logic [ADDR_WIDTH-1:0] addr_b,
    output logic [DATA_WIDTH-1:0] wr_data_b
);

    // determine input and output addresses in scratchpad and logit size
    logic [ADDR_WIDTH-1:0] input_base_addr, output_base_addr;
    int logit_size;
    always_comb begin
        case (param)
            ATTN_SOFTMAX: begin
                input_base_addr  = ATTN_LOGITS_BASE_ADDR;
                output_base_addr = ATTN_WEIGHTS_BASE_ADDR;
                logit_size       = BLOCK_SIZE;
            end
            FINAL_SOFTMAX: begin
                input_base_addr  = LOGITS_BUFFER_BASE_ADDR;
                output_base_addr = LOGITS_BUFFER_BASE_ADDR;
                logit_size       = VOCAB_SIZE;
            end
        endcase
    end

    // scaling for each softmax calculation
    logic [15:0] M_diff;
    logic signed [31:0] temp_val_odd, temp_val_even;

    always_comb begin
        case (param)
            ATTN_SOFTMAX:  M_diff = M_ATTN_SOFTMAX_DIFF;
            FINAL_SOFTMAX: M_diff = M_FINAL_SOFTMAX_DIFF;
            default:       M_diff = 16'h0001;
        endcase
    end

    // internal storage of exponential values
    logic [31:0] exps_mem [VOCAB_SIZE>>1];
    logic [31:0] exps_mem_wdata;
    logic [$clog2(VOCAB_SIZE>>1)-1:0] exps_mem_addr;
    logic exps_mem_we;

    always_ff @(posedge clk) begin
        if (exps_mem_we)
            exps_mem[exps_mem_addr] <= exps_mem_wdata;
    end

    // LUTs for exponential and reciprocal
    logic [5:0]  exp_idx, recip_idx;
    logic [15:0] exp_lut_val, recip_lut_val;

    exp_lut exp_lut_inst (
        .addr(exp_idx),
        .data(exp_lut_val)
    );

    recip_lut recip_lut_inst (
        .addr(recip_idx),
        .data(recip_lut_val)
    );

    // flip-flop signals
    logic [ADDR_WIDTH-1:0] addr_odd_d, addr_odd_q;
    logic [ADDR_WIDTH-1:0] addr_even_d, addr_even_q;
    logic signed [DATA_WIDTH-1:0] max_odd_d, max_odd_q;
    logic signed [DATA_WIDTH-1:0] max_even_d, max_even_q;
    logic signed [DATA_WIDTH-1:0] max_overall_d, max_overall_q;
    logic [$clog2(VOCAB_SIZE):0] count_d, count_q;
    logic [5:0] idx_d, idx_q;
    logic signed [15:0] diff_d, diff_q;                     // Q4.12
    logic [15:0] lut_val_s_d, lut_val_s_q;                  // Q1.15
    logic [15:0] lut_val_l_d, lut_val_l_q;                  // Q1.15
    logic signed [15:0] lut_val_diff_d, lut_val_diff_q;     // Q1.15
    logic [15:0] frac_d, frac_q;                            // Q0.16
    logic [15:0] exp_val_d, exp_val_q;                      // Q1.15
    logic [31:0] total_d, total_q;                          // Q17.15
    logic [15:0] m_d, m_q;                                  // Q1.15
    logic signed [15:0] recip_val_d, recip_val_q;           // Q1.15
    logic [2:0] exp_d, exp_q;           // exponent to extract in reciprocal approximation
    logic signed [15:0] raw_diff;       // unscaled diff
    logic signed [15:0] shifted_diff;   // diff-(-8)           Q4.12
    logic [15:0] shifted_m;             // m - 1               Q1.15
    logic signed [31:0] diff_prod;
    logic signed [21:0] idx_prod;                           // Q6.16
    logic signed [31:0] interp_prod;                        // Q1.31
    logic signed [63:0] result_prod_odd, result_prod_even;  // Q2.30

    assign addr_a    = addr_odd_d;
    assign addr_b    = addr_even_d;
    assign exp_idx   = idx_d;
    assign recip_idx = idx_d;

    // FSM states
    typedef enum logic [3:0] {
        IDLE, MAX, DIFF, EXP1, EXP2, EXP3, EXP4, TOTAL, 
        RECIP0, RECIP1, RECIP2, RECIP3, RECIP4, WRITE, DONE
    } state_t;

    state_t curr_state, next_state;

    // FSM sequential logic
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            curr_state     <= IDLE;
            count_q        <= 0;
            addr_odd_q     <= 0;
            addr_even_q    <= 0;
            max_odd_q      <= 0;
            max_even_q     <= 0;
            max_overall_q  <= 0;
            idx_q          <= 6'd0;
            diff_q         <= 16'd0;
            lut_val_s_q    <= 16'd0;
            lut_val_l_q    <= 16'd0;
            lut_val_diff_q <= 16'd0;
            frac_q         <= 16'd0;
            exp_val_q      <= 16'd0;
            total_q        <= 16'd0;
            m_q            <= 16'd0;
            recip_val_q    <= 16'd0;
            exp_q          <= 3'd0;
        end else begin
            curr_state     <= next_state;
            count_q        <= count_d;
            addr_odd_q     <= addr_odd_d;
            addr_even_q    <= addr_even_d;
            max_odd_q      <= max_odd_d;
            max_even_q     <= max_even_d;
            max_overall_q  <= max_overall_d;
            idx_q          <= idx_d;
            diff_q         <= diff_d;
            lut_val_s_q    <= lut_val_s_d;
            lut_val_l_q    <= lut_val_l_d;
            lut_val_diff_q <= lut_val_diff_d;
            frac_q         <= frac_d;
            exp_val_q      <= exp_val_d;
            total_q        <= total_d;
            m_q            <= m_d;
            recip_val_q    <= recip_val_d;
            exp_q          <= exp_d;
        end
    end

    // FSM combinational logic
    always_comb begin

        next_state     = curr_state;
        count_d        = count_q;
        addr_odd_d     = addr_odd_q;
        addr_even_d    = addr_even_q;
        max_odd_d      = max_odd_q;
        max_even_d     = max_even_q;
        max_overall_d  = max_overall_q;
        idx_d          = idx_q;
        diff_d         = diff_q;
        lut_val_s_d    = lut_val_s_q;
        lut_val_l_d    = lut_val_l_q;
        lut_val_diff_d = lut_val_diff_q;
        frac_d         = frac_q;
        exp_val_d      = exp_val_q;
        total_d        = total_q;
        m_d            = m_q;
        recip_val_d    = recip_val_q;
        exp_d          = exp_q;

        done           = 1'b0;
        wr_en_a        = 1'b0;
        wr_en_b        = 1'b0;
        wr_data_a      = 0;
        wr_data_b      = 0;
        exps_mem_we    = 1'b0;
        exps_mem_addr  = 0;
        exps_mem_wdata = 0;

        case (curr_state)

            IDLE: begin
                if (start) begin
                    max_odd_d     = -8'sd128; // smallest negative number possible
                    max_even_d    = -8'sd128;
                    max_overall_d = -8'sd128;
                    count_d       = 0;
                    addr_odd_d    = input_base_addr;
                    addr_even_d   = input_base_addr + 1;
                    next_state    = MAX;
                end
            end

            MAX: begin
                if ($signed(rd_data_a) > max_odd_q)
                    max_odd_d = rd_data_a;
                if ($signed(rd_data_b) > max_even_q)
                    max_even_d = rd_data_b;
                count_d = count_q + 2;

                if (count_q < logit_size - 2) begin
                    addr_odd_d  = addr_odd_q + 2;
                    addr_even_d = addr_even_q + 2;
                    next_state  = MAX;
                end else begin
                    if (max_odd_d > max_even_d)
                        max_overall_d = max_odd_d;
                    else
                        max_overall_d = max_even_d;
                    count_d    = 0;
                    addr_odd_d = input_base_addr;
                    next_state = DIFF;
                end
            end

            DIFF: begin
                raw_diff  = $signed(rd_data_a) - max_overall_q;
                diff_prod = raw_diff * $signed(M_diff);
                diff_d    = diff_prod >>> (S_SOFTMAX - 12);
                if (diff_d < DIFF_LOWER_BOUND) begin
                    exp_val_d  = 0;
                    next_state = TOTAL;
                end else begin
                    next_state = EXP1;
                end
            end

            EXP1: begin
                shifted_diff = diff_q - DIFF_LOWER_BOUND;
                idx_prod     = shifted_diff * $signed(M_EXP_IDX);
                idx_d        = idx_prod >> S_SOFTMAX;
                frac_d       = idx_prod[15:0];
                if (idx_d > 6'd31)
                    idx_d = 6'd31; // clamp to [0, 31]

                lut_val_s_d  = exp_lut_val;
                next_state   = EXP2;
            end

            EXP2: begin
                if (idx_q < 6'd32)
                    idx_d = idx_q + 1;
                lut_val_l_d = exp_lut_val;
                next_state  = EXP3;
            end

            EXP3: begin
                lut_val_diff_d = lut_val_l_q - lut_val_s_q;
                next_state     = EXP4;
            end

            EXP4: begin
                interp_prod   = $signed({1'b0, frac_q}) * lut_val_diff_q;            // Q1.31
                exp_val_d     = lut_val_s_q + (interp_prod >>> 16); // Q1.15
                exps_mem_we   = 1'b1;
                exps_mem_addr = count_q >> 1;
                if (count_q[0] == 1'b1)
                    exps_mem_wdata = {exps_mem[count_q >> 1][31:16], exp_val_d};
                else
                    exps_mem_wdata = {exp_val_d, exps_mem[count_q >> 1][15:0]};
                next_state    = TOTAL;
            end

            TOTAL: begin
                total_d    = total_q + exp_val_q;
                addr_odd_d = addr_odd_q + 1;
                count_d    = count_q + 1;
                if (count_q < logit_size - 1)
                    next_state = DIFF;
                else
                    next_state = RECIP0;
            end

            RECIP0: begin
                casez (total_q[21:15])
                    7'b1??????: exp_d = 6;
                    7'b01?????: exp_d = 5;
                    7'b001????: exp_d = 4;
                    7'b0001???: exp_d = 3;
                    7'b00001??: exp_d = 2;
                    7'b000001?: exp_d = 1;
                    7'b0000001: exp_d = 0;
                    default:    exp_d = 0;
                endcase
                m_d        = total_q >> exp_d;  // m in [1, 2)
                next_state = RECIP1;
            end

            RECIP1: begin
                shifted_m   = m_q - RECIP_LOWER_BOUND;
                idx_prod    = shifted_m * $signed(M_RECIP_IDX);
                idx_d       = idx_prod >> S_SOFTMAX;
                frac_d      = idx_prod[15:0];
                lut_val_s_d = recip_lut_val;
                next_state  = RECIP2;
            end

            RECIP2: begin
                if (idx_q < 32)
                    idx_d = idx_q + 1;
                lut_val_l_d = recip_lut_val;
                next_state  = RECIP3;
            end

            RECIP3: begin
                lut_val_diff_d = $signed(lut_val_l_q) - $signed(lut_val_s_q);
                next_state     = RECIP4;
            end

            RECIP4: begin
                interp_prod = $signed({1'b0, frac_q}) * lut_val_diff_q;  // Q1.31
                recip_val_d = ($signed(lut_val_s_q) + (interp_prod >>> 16)) >>> exp_q; // Q1.15
                count_d     = 0;
                next_state  = WRITE;
            end

            WRITE: begin
                wr_en_a     = 1'b1;
                wr_en_b     = 1'b1;
                addr_odd_d  = output_base_addr + count_q;
                addr_even_d = output_base_addr + count_q + 1;

                result_prod_odd  = $signed(exps_mem[count_q >> 1][31:16]) * recip_val_q * $signed(M_SOFTMAX_OUTPUT);
                result_prod_even = $signed(exps_mem[count_q >> 1][15:0]) * recip_val_q * $signed(M_SOFTMAX_OUTPUT);

                temp_val_odd  = (result_prod_odd) >>> S_SOFTMAX_OUTPUT;
                temp_val_even = (result_prod_even) >>> S_SOFTMAX_OUTPUT;

                if (temp_val_odd > 8'd255)
                    wr_data_a = 8'hFF;
                else if (temp_val_odd < 8'd0)
                    wr_data_a = 8'h00;
                else
                    wr_data_a = temp_val_odd[7:0];

                if (temp_val_even > 8'd255)
                    wr_data_b = 8'hFF;
                else if (temp_val_even < 8'd0)
                    wr_data_b = 8'h00;
                else
                    wr_data_b = temp_val_even[7:0];
                
                count_d = count_q + 2;
                if (count_q == logit_size - 2)
                    next_state = DONE;
            end

            DONE: begin
                done       = 1'b1;
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
endmodule