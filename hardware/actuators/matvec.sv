import proj_pkg::*;

module matvec #(
    parameter int LAYER_NUM  = 4,
    parameter int VOCAB_SIZE = 3005,
    parameter int N_EMBD     = 64,
    parameter int ADDR_WIDTH = 16,
    parameter int DATA_WIDTH = 8
) (
    input  logic clk,
    input  logic rst_n,

    // control logic
    input  logic start,
    output logic done,
    input  logic [$clog2(LAYER_NUM)-1:0] layer,
    input  matvec_param_t param,

    // weight ROM port A for reading odd rows
    output logic [$clog2(VOCAB_SIZE*N_EMBD)-1:0] row_odd_addr,
    input  logic [DATA_WIDTH-1:0] row_odd_data,

    // weight ROM port B for reading even rows
    output logic [$clog2(VOCAB_SIZE*N_EMBD)-1:0] row_even_addr,
    input  logic [DATA_WIDTH-1:0] row_even_data,

    // scratchpad port A for reading vector and writing even rows
    input  logic [DATA_WIDTH-1:0] vec_data,
    output logic wr_en_a,
    output logic [ADDR_WIDTH-1:0] wr_addr_a,
    output logic [DATA_WIDTH-1:0] wr_data_a,

    // scratchpad port B for writing odd rows
    output logic wr_en_b,
    output logic [ADDR_WIDTH-1:0] wr_addr_b,
    output logic [DATA_WIDTH-1:0] wr_data_b
);

    // determine matrix size
    logic [$clog2(VOCAB_SIZE)-1:0] num_of_rows;
    logic [$clog2(4*N_EMBD):0] num_of_cols;

    always_comb begin
        if (param == MLP_FC1) begin
            num_of_rows = 4 * N_EMBD;
            num_of_cols = N_EMBD;
        end else if (param == MLP_FC2) begin
            num_of_rows = N_EMBD;
            num_of_cols = 4 * N_EMBD;
        end else if (param == LM_HEAD) begin
            num_of_rows = VOCAB_SIZE;
            num_of_cols = N_EMBD;
        end else begin
            num_of_rows = N_EMBD;
            num_of_cols = N_EMBD;
        end
    end

    // determine matrix base address in weight ROM
    logic [31:0] mat_base_addr;
    assign mat_base_addr = layer * num_of_rows * num_of_cols;

    // determine vector and result base address in scratchpad
    logic [ADDR_WIDTH-1:0] vec_base_addr, result_base_addr;

    always_comb begin
        case (param)
            ATTN_WQ: begin
                vec_base_addr    = X_NORM_BASE_ADDR;
                result_base_addr = Q_BASE_ADDR;
            end
            ATTN_WK: begin
                vec_base_addr    = X_NORM_BASE_ADDR;
                result_base_addr = K_BASE_ADDR;
            end
            ATTN_WV: begin
                vec_base_addr    = X_NORM_BASE_ADDR;
                result_base_addr = V_BASE_ADDR;
            end
            ATTN_WO: begin
                vec_base_addr    = HEAD_OUT_BASE_ADDR;
                result_base_addr = POST_WO_BASE_ADDR;
            end
            MLP_FC1: begin
                vec_base_addr    = X_NORM_BASE_ADDR;
                result_base_addr = POST_MLP_FC1_BASE_ADDR;
            end
            MLP_FC2: begin
                vec_base_addr    = POST_RELU_BASE_ADDR;
                result_base_addr = POST_MLP_FC2_BASE_ADDR;
            end
            LM_HEAD: begin
                vec_base_addr    = X_NORM_BASE_ADDR;
                result_base_addr = LOGITS_BUFFER_BASE_ADDR;
            end
            default: begin
                vec_base_addr    = SCRATCHPAD_BASE_ADDR;
                result_base_addr = SCRATCHPAD_BASE_ADDR;
            end
        endcase
    end

    // scaling for each matrix-vector multiplication
    logic [15:0] M_scale;
    logic signed [47:0] temp_val_odd, temp_val_even; // used for clamping when scaling

    always_comb begin
        case (param)
            ATTN_WQ: begin
                case (layer)
                    0: M_scale = M_ATTN_WQ_LAYER_0;
                    1: M_scale = M_ATTN_WQ_LAYER_1;
                    2: M_scale = M_ATTN_WQ_LAYER_2;
                    3: M_scale = M_ATTN_WQ_LAYER_3;
                    default: M_scale = 16'h0001;
                endcase
            end
            ATTN_WK: begin
                case (layer)
                    0: M_scale = M_ATTN_WK_LAYER_0;
                    1: M_scale = M_ATTN_WK_LAYER_1;
                    2: M_scale = M_ATTN_WK_LAYER_2;
                    3: M_scale = M_ATTN_WK_LAYER_3;
                    default: M_scale = 16'h0001;
                endcase
            end
            ATTN_WV: begin
                case (layer)
                    0: M_scale = M_ATTN_WV_LAYER_0;
                    1: M_scale = M_ATTN_WV_LAYER_1;
                    2: M_scale = M_ATTN_WV_LAYER_2;
                    3: M_scale = M_ATTN_WV_LAYER_3;
                    default: M_scale = 16'h0001;
                endcase
            end
            ATTN_WO: begin
                case (layer)
                    0: M_scale = M_ATTN_WO_LAYER_0;
                    1: M_scale = M_ATTN_WO_LAYER_1;
                    2: M_scale = M_ATTN_WO_LAYER_2;
                    3: M_scale = M_ATTN_WO_LAYER_3;
                    default: M_scale = 16'h0001;
                endcase
            end
            MLP_FC1: begin
                case (layer)
                    0: M_scale = M_MLP_FC1_LAYER_0;
                    1: M_scale = M_MLP_FC1_LAYER_1;
                    2: M_scale = M_MLP_FC1_LAYER_2;
                    3: M_scale = M_MLP_FC1_LAYER_3;
                    default: M_scale = 16'h0001;
                endcase
            end
            MLP_FC2: begin
                case (layer)
                    0: M_scale = M_MLP_FC2_LAYER_0;
                    1: M_scale = M_MLP_FC2_LAYER_1;
                    2: M_scale = M_MLP_FC2_LAYER_2;
                    3: M_scale = M_MLP_FC2_LAYER_3;
                    default: M_scale = 16'h0001;
                endcase
            end
            LM_HEAD: M_scale = M_LM_HEAD;
            default: M_scale = 16'h0001;
        endcase
    end

    // flip-flop signals
    logic [$clog2(4*N_EMBD)-1:0] item_count_d, item_count_q;
    logic [$clog2(VOCAB_SIZE)-1:0] row_count_d, row_count_q;
    logic [$clog2(VOCAB_SIZE*N_EMBD*LAYER_NUM)-1:0] row_odd_addr_d, row_odd_addr_q;
    logic [$clog2(VOCAB_SIZE*N_EMBD*LAYER_NUM)-1:0] row_even_addr_d, row_even_addr_q;
    logic [ADDR_WIDTH-1:0] vec_addr_d, vec_addr_q;
    logic [ADDR_WIDTH-1:0] wr_addr_b_d, wr_addr_b_q;
    logic signed [31:0] acc_odd_d, acc_odd_q, acc_even_d, acc_even_q;
    
    assign row_odd_addr  = row_odd_addr_d;
    assign row_even_addr = row_even_addr_d;
    assign wr_addr_a     = vec_addr_d;
    assign wr_addr_b     = wr_addr_b_d;

    // to avoid issues with over-running
    logic is_second_valid;
    assign is_second_valid = ((row_count_q + 1) < num_of_rows);

    // FSM states
    typedef enum logic [2:0] {
        IDLE  = 3'b000,
        ADD   = 3'b001,
        WRITE = 3'b010,
        WAIT  = 3'b011,
        DONE  = 3'b100
    } state_t;

    state_t curr_state, next_state;

    // FSM sequential logic
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            curr_state      <= IDLE;
            item_count_q    <= 0;
            row_count_q     <= 0;
            row_odd_addr_q  <= 0;
            row_even_addr_q <= 0;
            vec_addr_q      <= 0;
            wr_addr_b_q     <= 0;
            acc_odd_q       <= 0;
            acc_even_q      <= 0;
        end else begin
            curr_state      <= next_state;
            item_count_q    <= item_count_d;
            row_count_q     <= row_count_d;
            row_odd_addr_q  <= row_odd_addr_d;
            row_even_addr_q <= row_even_addr_d;
            vec_addr_q      <= vec_addr_d;
            wr_addr_b_q     <= wr_addr_b_d;
            acc_odd_q       <= acc_odd_d;
            acc_even_q      <= acc_even_d;
        end
    end

    // FSM combinational logic
    always_comb begin

        next_state      = curr_state;
        item_count_d    = item_count_q;
        row_count_d     = row_count_q;
        row_odd_addr_d  = row_odd_addr_q;
        row_even_addr_d = row_even_addr_q;
        vec_addr_d      = vec_addr_q;
        wr_addr_b_d     = wr_addr_b_q;
        acc_odd_d       = acc_odd_q;
        acc_even_d      = acc_even_q;

        done          = 1'b0;
        wr_en_a       = 1'b0;
        wr_en_b       = 1'b0;
        wr_data_a     = 0;
        wr_data_b     = 0;
        temp_val_odd  = 0;
        temp_val_even = 0;

        case (curr_state)

            IDLE: begin
                if (start) begin
                    vec_addr_d      = vec_base_addr;
                    row_odd_addr_d  = mat_base_addr;
                    row_even_addr_d = mat_base_addr + num_of_cols;
                    acc_odd_d       = 32'b0;
                    acc_even_d      = 32'b0;
                    item_count_d    = 0;
                    row_count_d     = 0;
                    next_state      = ADD;
                end
            end

            ADD: begin
                acc_odd_d    = acc_odd_q + $signed(row_odd_data) * $signed(vec_data);
                acc_even_d   = acc_even_q + $signed(row_even_data) * $signed(vec_data);
                item_count_d = item_count_q + 1;

                if (item_count_q < num_of_cols - 1) begin 
                    row_odd_addr_d  = row_odd_addr_q + 1;
                    row_even_addr_d = row_even_addr_q + 1;
                    vec_addr_d      = vec_addr_q + 1;
                    next_state      = ADD;
                end else begin
                    vec_addr_d  = result_base_addr + row_count_q;
                    wr_addr_b_d = result_base_addr + row_count_q + 1;
                    next_state  = WRITE;
                end
            end

            WRITE: begin
                wr_en_a = 1'b1;
                wr_en_b = is_second_valid;
                
                // apply scaling
                temp_val_odd = ((acc_odd_q * $signed(M_scale)) + (1 <<< (S_MATVEC - 1))) >>> S_MATVEC;
                if (temp_val_odd > 8'sd127)
                    wr_data_a = 8'h7F;
                else if (temp_val_odd < -8'sd127)
                    wr_data_a = 8'h81;
                else
                    wr_data_a = temp_val_odd[7:0];

                temp_val_even = ((acc_even_q * $signed(M_scale)) + (1 <<< (S_MATVEC - 1))) >>> S_MATVEC;
                if (temp_val_even > 8'sd127)
                    wr_data_b = 8'h7F;
                else if (temp_val_even < -8'sd127)
                    wr_data_b = 8'h81;
                else
                    wr_data_b = temp_val_even[7:0];

                row_count_d     = row_count_q + 2;
                acc_odd_d       = 32'b0;
                acc_even_d      = 32'b0;
                item_count_d    = 0;
                row_odd_addr_d  = row_odd_addr_q + num_of_cols + 1;

                if (is_second_valid)
                    row_even_addr_d = row_even_addr_q + num_of_cols + 1;

                if (row_count_q < num_of_rows - 2) begin
                    next_state = WAIT;
                end else begin
                    next_state = DONE;
                end
            end

            WAIT: begin
                vec_addr_d = vec_base_addr;
                next_state = ADD;
            end

            DONE: begin
                done       = 1'b1;
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
            
        endcase
    end
    
endmodule