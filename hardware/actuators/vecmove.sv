import proj_pkg::*;

module vecmove #(
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
    input  logic [1:0] layer,
    input  logic [$clog2(BLOCK_SIZE)-1:0] pos_id,
    input  vecmove_param_t param,

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
    logic [ADDR_WIDTH-1:0] input_base_addr, output_base_addr;
    int loop_size;

    always_comb begin
        case (param)
            RELU: begin
                input_base_addr  = POST_MLP_FC1_BASE_ADDR;
                output_base_addr = POST_RELU_BASE_ADDR;
            end
            COPY_RESIDUAL_ATTN: begin
                input_base_addr  = (layer == 2'b00) ? X_NORM_BASE_ADDR : SUM_RESIDUAL_BASE_ADDR;
                output_base_addr = X_EMBD_BASE_ADDR;
            end
            COPY_RESIDUAL_MLP: begin
                input_base_addr  = SUM_RESIDUAL_BASE_ADDR;
                output_base_addr = X_EMBD_BASE_ADDR;
            end
            COPY_TO_K_CACHE: begin
                input_base_addr  = K_BASE_ADDR;
                output_base_addr = K_CACHE_BASE_ADDR + layer * KV_CACHE_LAYER_SIZE + pos_id * N_EMBD;
            end
            COPY_TO_V_CACHE: begin
                input_base_addr  = V_BASE_ADDR;
                output_base_addr = V_CACHE_BASE_ADDR + layer * KV_CACHE_LAYER_SIZE + pos_id * N_EMBD;
            end
            default: begin
                input_base_addr  = SCRATCHPAD_BASE_ADDR;
                output_base_addr = SCRATCHPAD_BASE_ADDR;
            end
        endcase

        loop_size = (param == RELU) ? (N_EMBD * 4) : N_EMBD;
    end

    // flip-flop signals
    logic [ADDR_WIDTH-1:0] rd_addr_d, rd_addr_q;
    logic [ADDR_WIDTH-1:0] wr_addr_d, wr_addr_q;
    logic [$clog2(N_EMBD*4)-1:0] count_d, count_q;
    logic signed [DATA_WIDTH-1:0] after_relu;
    logic signed [31:0] temp_val_d, temp_val_q;

    assign addr_a = rd_addr_d;
    assign addr_b = wr_addr_q;

    // FSM states
    typedef enum logic [2:0] {
        IDLE, LOOP, LOOP_RELU1, LOOP_RELU2, DONE
    } state_t;

    state_t curr_state, next_state;

    // FSM sequential logic
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            curr_state <= IDLE;
            rd_addr_q  <= 0;
            wr_addr_q  <= 0;
            count_q    <= 0;
            temp_val_q <= 0;
        end else begin
            curr_state <= next_state;
            rd_addr_q  <= rd_addr_d;
            wr_addr_q  <= wr_addr_d;
            count_q    <= count_d;
            temp_val_q <= temp_val_d;
        end
    end

    // FSM combinational logic
    always_comb begin

        next_state = curr_state;
        rd_addr_d  = rd_addr_q;
        wr_addr_d  = wr_addr_q;
        count_d    = count_q;
        temp_val_d = temp_val_q;

        done      = 1'b0;
        wr_en_a   = 1'b0;
        wr_en_b   = 1'b0;
        wr_data_a = 0;
        wr_data_b = 0;

        case (curr_state)

            IDLE: begin
                if (start) begin
                    count_d    = 0;
                    rd_addr_d  = input_base_addr;
                    wr_addr_d  = output_base_addr;
                    if (param == RELU)
                        next_state = LOOP_RELU1;
                    else  
                        next_state = LOOP;
                end
            end

            LOOP: begin
                wr_en_b  = 1'b1;
                wr_data_b = rd_data_a;
                count_d = count_q + 1;

                if (count_q < loop_size - 1) begin
                    rd_addr_d  = rd_addr_q + 1;
                    wr_addr_d  = wr_addr_q + 1;
                    next_state = LOOP;
                end else begin
                    next_state = DONE;
                end
            end

            LOOP_RELU1: begin
                after_relu = ($signed(rd_data_a) > 0) ? rd_data_a : 8'sd0;
                temp_val_d = (after_relu * $signed(M_RELU)) >>> S_RELU;
                next_state = LOOP_RELU2;
            end

            LOOP_RELU2: begin
                wr_en_b  = 1'b1;
                if (temp_val_q > 8'sd127)
                    wr_data_b = 8'h7F;
                else
                    wr_data_b = temp_val_q[7:0];
                
                count_d = count_q + 1;
                if (count_q < loop_size - 1) begin
                    rd_addr_d  = rd_addr_q + 1;
                    wr_addr_d  = wr_addr_q + 1;
                    next_state = LOOP_RELU1;
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