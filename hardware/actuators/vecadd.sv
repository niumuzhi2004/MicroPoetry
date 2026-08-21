import proj_pkg::*;

module vecadd #(
    parameter int N_EMBD     = 64,
    parameter int ADDR_WIDTH = 16,
    parameter int DATA_WIDTH = 8
) (
    input  logic clk,
    input  logic rst_n,

    // control logic
    input  logic start,
    output logic done,
    input  logic [1:0] layer,
    input  vecadd_param_t param,

    // scratchpad port A for reading entries
    input  logic [DATA_WIDTH-1:0] rd_data_a,
    output logic wr_en_a,
    output logic [ADDR_WIDTH-1:0] addr_a,
    output logic [DATA_WIDTH-1:0] wr_data_a,

    // scratchpad port B for writing entries
    input  logic [DATA_WIDTH-1:0] rd_data_b,
    output logic wr_en_b,
    output logic [ADDR_WIDTH-1:0] addr_b,
    output logic [DATA_WIDTH-1:0] wr_data_b
);

    // determine input and output addresses in scratchpad
    logic [ADDR_WIDTH-1:0] input_a_base_addr, input_b_base_addr, output_base_addr;

    always_comb begin
        case (param)
            ATTN_VEC_SUM: begin
                input_a_base_addr = POST_WO_BASE_ADDR;
                input_b_base_addr = X_EMBD_BASE_ADDR;
                output_base_addr  = SUM_RESIDUAL_BASE_ADDR;
            end
            MLP_VEC_SUM: begin
                input_a_base_addr = POST_MLP_FC2_BASE_ADDR;
                input_b_base_addr = X_EMBD_BASE_ADDR;
                output_base_addr  = SUM_RESIDUAL_BASE_ADDR;
            end
            default: begin
                input_a_base_addr = SCRATCHPAD_BASE_ADDR;
                input_b_base_addr = SCRATCHPAD_BASE_ADDR;
                output_base_addr  = SCRATCHPAD_BASE_ADDR;
            end
        endcase
    end

    // scaling for each vector add operation
    logic [15:0] M_scale_a, M_scale_b;

    always_comb begin
        case (param)
            ATTN_VEC_SUM: begin
                M_scale_a = M_VECADD_ATTN_SCALE_A;
                M_scale_b = (layer == 2'b00) ? M_VECADD_ATTN_SCALE_B_LAYER0 : M_VECADD_ATTN_SCALE_B_LAYER123;
            end
            MLP_VEC_SUM: begin
                M_scale_a = M_VECADD_MLP_SCALE_A;
                M_scale_b = M_VECADD_MLP_SCALE_B;
            end
            default: begin
                M_scale_a = 16'h0001;
                M_scale_b = 16'h0001;
            end
        endcase
    end

    // flip-flop signals
    logic [ADDR_WIDTH-1:0] rd_addr_d, rd_addr_q;
    logic [$clog2(N_EMBD)-1:0] count_d, count_q;
    logic signed [31:0] rescaled_a_d, rescaled_a_q;
    logic signed [31:0] rescaled_b_d, rescaled_b_q;
    logic signed [31:0] temp_val;

    assign addr_a = rd_addr_d;

    // FSM states
    typedef enum logic [2:0] {
        IDLE, RD1, RD2, WRITE, DONE
    } state_t;

    state_t curr_state, next_state;

    // FSM sequential logic
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            curr_state   <= IDLE;
            rd_addr_q    <= 0;
            count_q      <= 0;
            rescaled_a_q <= 32'd0;
            rescaled_b_q <= 32'd0;
        end else begin
            curr_state   <= next_state;
            rd_addr_q    <= rd_addr_d;
            count_q      <= count_d;
            rescaled_a_q <= rescaled_a_d;
            rescaled_b_q <= rescaled_b_d;
        end
    end

    // FSM combinational logic
    always_comb begin

        next_state   = curr_state;
        rd_addr_d    = rd_addr_q;
        count_d      = count_q;
        rescaled_a_d = rescaled_a_q;
        rescaled_b_d = rescaled_b_q;

        done      = 1'b0;
        wr_en_a   = 1'b0;
        wr_en_b   = 1'b0;
        addr_b    = 0;
        wr_data_a = 0;
        wr_data_b = 0;

        case (curr_state)

            IDLE: begin
                if (start) begin
                    count_d    = 0;
                    rd_addr_d  = input_a_base_addr;
                    next_state = RD1;
                end
            end

            RD1: begin
                rescaled_a_d = ($signed(rd_data_a) * $signed({1'b0, M_scale_a})) >>> S_VECADD;
                rd_addr_d    = input_b_base_addr + count_q;
                next_state   = RD2;
            end

            RD2: begin
                rescaled_b_d = ($signed(rd_data_a) * $signed({1'b0, M_scale_b})) >>> S_VECADD;
                next_state   = WRITE;
            end

            WRITE: begin
                addr_b   = output_base_addr + count_q;
                wr_en_b  = 1'b1;
                temp_val = rescaled_a_q + rescaled_b_q;

                if (temp_val > 8'sd127)
                    wr_data_b = 8'h7F;
                else if (temp_val < -8'sd127)
                    wr_data_b = 8'h81;
                else
                    wr_data_b = temp_val[7:0];

                count_d = count_q + 1;

                if (count_q < N_EMBD - 1) begin
                    rd_addr_d  = input_a_base_addr + count_d;
                    next_state = RD1;
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