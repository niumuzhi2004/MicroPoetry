import proj_pkg::*;

module attn_score #(
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

    // scratchpad port A for reading q values and writing odd entries
    input  logic [DATA_WIDTH-1:0] rd_data_a,
    output logic wr_en_a,
    output logic [ADDR_WIDTH-1:0] addr_a,
    output logic [DATA_WIDTH-1:0] wr_data_a,

    // scratchpad port B for reading k values and writing even entries
    input  logic [DATA_WIDTH-1:0] rd_data_b,
    output logic wr_en_b,
    output logic [ADDR_WIDTH-1:0] addr_b,
    output logic [DATA_WIDTH-1:0] wr_data_b
);

    // determine input and output addresses in scratchpad
    logic [ADDR_WIDTH-1:0] q_base_addr, k_base_addr, output_base_addr;
    logic [$clog2(BLOCK_SIZE)-1:0] logit_size;
    always_comb begin
        q_base_addr = Q_BASE_ADDR + head_id * HEAD_DIM;
        case (layer)
            2'b00:   k_base_addr = K_LAYER0_BASE_ADDR + head_id * HEAD_DIM;
            2'b01:   k_base_addr = K_LAYER1_BASE_ADDR + head_id * HEAD_DIM;
            2'b10:   k_base_addr = K_LAYER2_BASE_ADDR + head_id * HEAD_DIM;
            2'b11:   k_base_addr = K_LAYER3_BASE_ADDR + head_id * HEAD_DIM;
            default: k_base_addr = K_CACHE_BASE_ADDR;
        endcase
        output_base_addr = ATTN_LOGITS_BASE_ADDR + head_id * BLOCK_SIZE;
        logit_size       = pos_id + 1;
    end

    // flip-flop signals
    logic [ADDR_WIDTH-1:0] addr_q_d, addr_q_q;
    logic [ADDR_WIDTH-1:0] addr_k_d, addr_k_q;
    logic [$clog2(HEAD_DIM)-1:0] dim_count_d, dim_count_q;
    logic [$clog2(BLOCK_SIZE)-1:0] pos_count_d, pos_count_q;
    logic [DATA_WIDTH-1:0] rd_data_a_q, rd_data_b_q;
    (* USE_DSP = "yes" *) logic signed [31:0] acc_d, acc_q;
    logic signed [47:0] temp_val_d, temp_val_q;

    assign addr_a = addr_q_d;
    assign addr_b = addr_k_d;

    // FSM states
    typedef enum logic [2:0] {
        IDLE, WAIT1, SUM, WRITE1, WRITE2, WAIT2, DONE
    } state_t;

    state_t curr_state, next_state;

    // FSM sequential logic
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            curr_state  <= IDLE;
            addr_q_q    <= 0;
            addr_k_q    <= 0;
            dim_count_q <= 0;
            pos_count_q <= 0;
            rd_data_a_q <= 0;
            rd_data_b_q <= 0;
            acc_q       <= 32'b0;
            temp_val_q  <= 48'b0;
        end else begin
            curr_state  <= next_state;
            addr_q_q    <= addr_q_d;
            addr_k_q    <= addr_k_d;
            dim_count_q <= dim_count_d;
            pos_count_q <= pos_count_d;
            rd_data_a_q <= rd_data_a;
            rd_data_b_q <= rd_data_b;
            acc_q       <= acc_d;
            temp_val_q  <= temp_val_d;
        end
    end

    // FSM combinational logic
    always_comb begin
        
        next_state  = curr_state;
        addr_q_d    = addr_q_q;
        addr_k_d    = addr_k_q;
        dim_count_d = dim_count_q;
        pos_count_d = pos_count_q;
        acc_d       = acc_q;
        temp_val_d  = temp_val_q;

        done      = 1'b0;
        wr_en_a   = 1'b0;
        wr_en_b   = 1'b0;
        wr_data_a = 0;
        wr_data_b = 0;

        case (curr_state)

            IDLE: begin
                if (start) begin
                    addr_q_d    = q_base_addr;
                    addr_k_d    = k_base_addr;
                    dim_count_d = 0;
                    pos_count_d = 0;
                    acc_d       = 0;
                    next_state  = WAIT1;
                end
            end

            WAIT1: begin
                addr_q_d   = addr_q_q + 1;
                addr_k_d   = addr_k_q + 1;
                next_state = SUM;
            end

            SUM: begin
                acc_d       = acc_q + $signed(rd_data_a_q) * $signed(rd_data_b_q);
                dim_count_d = dim_count_q + 1;

                if (dim_count_q < HEAD_DIM - 1) begin
                    addr_q_d   = addr_q_q + 1;
                    addr_k_d   = addr_k_q + 1;
                    next_state = SUM;
                end else begin
                    addr_q_d   = output_base_addr + pos_count_q;
                    next_state = WRITE1;
                end
            end

            WRITE1: begin
                temp_val_d = (acc_q * $signed(M_ATTN_SCORE)) >>> S_ATTN_SCORE;
                next_state = WRITE2;
            end

            WRITE2: begin
                wr_en_a    = 1'b1;

                if (temp_val_q > 8'sd127)
                    wr_data_a = 8'sd127;
                else if (temp_val_q < -8'sd127)
                    wr_data_a = -8'sd127;
                else
                    wr_data_a = temp_val_q[7:0];

                acc_d       = 0;
                dim_count_d = 0;
                pos_count_d = pos_count_q + 1;

                if (pos_count_q < logit_size - 1)
                    next_state = WAIT2;
                else 
                    next_state = DONE;
            end

            WAIT2: begin
                addr_q_d   = q_base_addr;
                addr_k_d   = k_base_addr + N_EMBD * pos_count_d;
                next_state = WAIT1;
            end

            DONE: begin
                done       = 1'b1;
                next_state = IDLE;
            end
            
            default: next_state = IDLE;

        endcase
    end
    
endmodule