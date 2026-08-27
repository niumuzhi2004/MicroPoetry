import proj_pkg::*;

module attn_sum #(
    parameter int LAYER_NUM  = 4,
    parameter int N_HEAD     = 4,
    parameter int HEAD_DIM   = 16,
    parameter int N_EMBD     = 64,
    parameter int BLOCK_SIZE = 96,
    parameter int ADDR_WIDTH = 16,
    parameter int DATA_WIDTH = 8
) (
    input  logic clk,
    input  logic rst_n,

    // control logic
    input  logic start,
    output logic done,
    input  logic [$clog2(LAYER_NUM)-1:0] layer,
    input  logic [$clog2(N_HEAD)-1:0] head_id,
    input  logic [$clog2(BLOCK_SIZE)-1:0] pos_id,

    // scratchpad port A for reading attn_weight values and writing odd entries
    input  logic [DATA_WIDTH-1:0] rd_data_a,
    output logic wr_en_a,
    output logic [ADDR_WIDTH-1:0] addr_a,
    output logic [DATA_WIDTH-1:0] wr_data_a,

    // scratchpad port B for reading v values and writing even entries
    input  logic [DATA_WIDTH-1:0] rd_data_b,
    output logic wr_en_b,
    output logic [ADDR_WIDTH-1:0] addr_b,
    output logic [DATA_WIDTH-1:0] wr_data_b
);

    // determine input and output addresses in scratchpad
    logic [ADDR_WIDTH-1:0] weights_base_addr, v_base_addr, output_base_addr;
    logic [$clog2(BLOCK_SIZE)-1:0] logit_size;
    always_comb begin
        weights_base_addr = ATTN_WEIGHTS_BASE_ADDR + head_id * BLOCK_SIZE;
        case (layer)
            2'b00:   v_base_addr = V_LAYER0_BASE_ADDR + head_id * HEAD_DIM;
            2'b01:   v_base_addr = V_LAYER1_BASE_ADDR + head_id * HEAD_DIM;
            2'b10:   v_base_addr = V_LAYER2_BASE_ADDR + head_id * HEAD_DIM;
            2'b11:   v_base_addr = V_LAYER3_BASE_ADDR + head_id * HEAD_DIM;
            default: v_base_addr = V_CACHE_BASE_ADDR;
        endcase
        output_base_addr = HEAD_OUT_BASE_ADDR + head_id * HEAD_DIM;
        logit_size       = pos_id + 1;
    end

    // flip-flop signals
    logic [ADDR_WIDTH-1:0] addr_weight_d, addr_weight_q;
    logic [ADDR_WIDTH-1:0] addr_v_d, addr_v_q;
    logic [$clog2(HEAD_DIM)-1:0] dim_count_d, dim_count_q;
    logic [$clog2(BLOCK_SIZE)-1:0] pos_count_d, pos_count_q;
    (* USE_DSP = "yes" *) logic signed [31:0] acc_d, acc_q;
    logic signed [47:0] temp_val;

    assign addr_a = addr_weight_d;
    assign addr_b = addr_v_d;

    // FSM states
    typedef enum logic [2:0] {
        IDLE, SUM, WRITE, WAIT, DONE
    } state_t;

    state_t curr_state, next_state;

    // FSM sequential logic
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            curr_state    <= IDLE;
            addr_weight_q <= 0;
            addr_v_q      <= 0;
            dim_count_q   <= 0;
            pos_count_q   <= 0;
            acc_q         <= 32'b0;
        end else begin
            curr_state    <= next_state;
            addr_weight_q <= addr_weight_d;
            addr_v_q      <= addr_v_d;
            dim_count_q   <= dim_count_d;
            pos_count_q   <= pos_count_d;
            acc_q         <= acc_d;
        end
    end

    // FSM combinational logic
    always_comb begin
        
        next_state    = curr_state;
        addr_weight_d = addr_weight_q;
        addr_v_d      = addr_v_q;
        dim_count_d   = dim_count_q;
        pos_count_d   = pos_count_q;
        acc_d         = acc_q;

        done      = 1'b0;
        wr_en_a   = 1'b0;
        wr_en_b   = 1'b0;
        wr_data_a = 0;
        wr_data_b = 0;
        temp_val  = 0;

        case (curr_state)

            IDLE: begin
                if (start) begin
                    addr_weight_d = weights_base_addr;
                    addr_v_d      = v_base_addr;
                    dim_count_d   = 0;
                    pos_count_d   = 0;
                    acc_d         = 0;
                    next_state    = SUM;
                end
            end

            SUM: begin
                acc_d       = acc_q + $signed({1'b0, rd_data_a}) * $signed(rd_data_b);
                pos_count_d = pos_count_q + 1;

                if (pos_count_q < logit_size - 1) begin
                    addr_weight_d = addr_weight_q + 1;
                    addr_v_d      = addr_v_q + N_EMBD;
                    next_state    = SUM;
                end else begin
                    addr_weight_d = output_base_addr + dim_count_q;
                    next_state = WRITE;
                end
            end

            WRITE: begin
                wr_en_a  = 1'b1;
                temp_val  = (acc_q * $signed(M_ATTN_SUM)) >>> S_ATTN_SUM;

                if (temp_val > 8'sd127)
                    wr_data_a = 8'sd127;
                else if (temp_val < -8'sd127)
                    wr_data_a = -8'sd127;
                else
                    wr_data_a = temp_val[7:0];

                acc_d       = 0;
                pos_count_d = 0;
                dim_count_d = dim_count_q + 1;

                if (dim_count_q < HEAD_DIM - 1)
                    next_state = WAIT;
                else 
                    next_state = DONE;
            end

            WAIT: begin
                addr_weight_d = weights_base_addr;
                addr_v_d      = v_base_addr + dim_count_q;
                next_state    = SUM;
            end

            DONE: begin
                done       = 1'b1;
                next_state = IDLE;
            end
            
            default: next_state = IDLE;

        endcase
    end
    
endmodule