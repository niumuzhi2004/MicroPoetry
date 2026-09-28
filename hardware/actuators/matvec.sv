import proj_pkg::*;

module matvec #(
    parameter int LAYER_NUM  = 4,
    parameter int VOCAB_SIZE = 3005,
    parameter int N_EMBD     = 64,
    parameter int MAX_BANKS  = 47,
    parameter int BANK_DEPTH = 1024,
    parameter int WORD_WIDTH = 32,
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
    output logic [$clog2(BANK_DEPTH)-1:0] row_odd_addr  [MAX_BANKS],
    input  logic [WORD_WIDTH-1:0] row_odd_data          [MAX_BANKS],

    // weight ROM port B for reading even rows
    output logic [$clog2(BANK_DEPTH)-1:0] row_even_addr [MAX_BANKS],
    input  logic [WORD_WIDTH-1:0] row_even_data         [MAX_BANKS],

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
    logic [$clog2(ROWS_PER_BANK_WTE):0] rows_per_bank;
    logic [$clog2(MAX_BANKS)-1:0] num_of_banks;
    logic [$clog2(4*N_EMBD):0] bytes_per_row;

    always_comb begin
        if (param == MLP_FC1) begin
            num_of_banks  = N_BANKS_MLP;
            rows_per_bank = ROWS_PER_BANK_MLP_FC1;
            bytes_per_row = N_EMBD;
        end else if (param == MLP_FC2) begin
            num_of_banks  = N_BANKS_MLP;
            rows_per_bank = ROWS_PER_BANK_MLP_FC2;
            bytes_per_row = 4 * N_EMBD;
        end else if (param == LM_HEAD) begin
            num_of_banks  = N_BANKS_WTE;
            rows_per_bank = ROWS_PER_BANK_WTE;
            bytes_per_row = N_EMBD;
        end else begin
            num_of_banks  = N_BANKS_ATTN;
            rows_per_bank = ROWS_PER_BANK_ATTN_W;
            bytes_per_row = N_EMBD;
        end
    end

    // determine matrix base address in weight ROM
    logic [31:0] mat_base_addr;
    assign mat_base_addr = layer * rows_per_bank * bytes_per_row;

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
    logic [15:0] M_scale, M_scale_d, M_scale_q;

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
    logic [$clog2(4*N_EMBD)-1:0] col_count_d, col_count_q;
    logic [$clog2(MAX_BANKS)-1:0] bank_count_d, bank_count_q;
    logic [$clog2(VOCAB_SIZE)-1:0] row_count_d, row_count_q;
    logic [$clog2(BANK_DEPTH)-1:0] row_odd_addr_d [MAX_BANKS];
    logic [$clog2(BANK_DEPTH)-1:0] row_odd_addr_q [MAX_BANKS];
    logic [$clog2(BANK_DEPTH)-1:0] row_even_addr_d [MAX_BANKS];
    logic [$clog2(BANK_DEPTH)-1:0] row_even_addr_q [MAX_BANKS];
    logic [$clog2(BANK_DEPTH)-1:0] odd_base_addr_d, odd_base_addr_q;
    logic [$clog2(BANK_DEPTH)-1:0] even_base_addr_d, even_base_addr_q;
    logic [ADDR_WIDTH-1:0] vec_addr_d, vec_addr_q;
    (* USE_DSP = "yes" *) logic signed [31:0] acc_odd_d  [MAX_BANKS];
    (* USE_DSP = "yes" *) logic signed [31:0] acc_odd_q  [MAX_BANKS];
    (* USE_DSP = "yes" *) logic signed [31:0] acc_even_d [MAX_BANKS];
    (* USE_DSP = "yes" *) logic signed [31:0] acc_even_q [MAX_BANKS];
    logic signed [47:0] temp_val_odd_d, temp_val_odd_q, temp_val_even_d, temp_val_even_q; // used for clamping when scaling
    logic [WORD_WIDTH-1:0] row_odd_data_q  [MAX_BANKS];
    logic [WORD_WIDTH-1:0] row_even_data_q [MAX_BANKS];
    logic signed [DATA_WIDTH-1:0] vec_data_q;
    
    assign row_odd_addr  = row_odd_addr_d;
    assign row_even_addr = row_even_addr_d;

    // FSM states
    typedef enum logic [2:0] {
        IDLE, ADD, WRITE1, WRITE2, WRITE3, WAIT1, WAIT2, DONE
    } state_t;

    state_t curr_state, next_state;

    assign wr_addr_a = (curr_state == WRITE2) ? (result_base_addr + (bank_count_q * rows_per_bank) + row_count_q) : vec_addr_d;
    assign wr_addr_b = result_base_addr + (bank_count_q * rows_per_bank) + row_count_q + 1;

    // FSM sequential logic
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            curr_state       <= IDLE;
            item_count_q     <= 0;
            col_count_q      <= 0;
            bank_count_q     <= 0;
            row_count_q      <= 0;
            row_odd_addr_q   <= '{default: '0};
            row_even_addr_q  <= '{default: '0};
            odd_base_addr_q  <= 0;
            even_base_addr_q <= 0;
            vec_addr_q       <= 0;
            acc_odd_q        <= '{default: '0};
            acc_even_q       <= '{default: '0};
            M_scale_q        <= 0;
            temp_val_odd_q   <= 0;
            temp_val_even_q  <= 0;
            row_odd_data_q   <= '{default: '0};
            row_even_data_q  <= '{default: '0};
            vec_data_q       <= 0;
        end else begin
            curr_state       <= next_state;
            item_count_q     <= item_count_d;
            col_count_q      <= col_count_d;
            bank_count_q     <= bank_count_d;
            row_count_q      <= row_count_d;
            row_odd_addr_q   <= row_odd_addr_d;
            row_even_addr_q  <= row_even_addr_d;
            odd_base_addr_q  <= odd_base_addr_d;
            even_base_addr_q <= even_base_addr_d;
            vec_addr_q       <= vec_addr_d;
            acc_odd_q        <= acc_odd_d;
            acc_even_q       <= acc_even_d;
            M_scale_q        <= M_scale;
            temp_val_odd_q   <= temp_val_odd_d;
            temp_val_even_q  <= temp_val_even_d;
            row_odd_data_q   <= row_odd_data;
            row_even_data_q  <= row_even_data;
            vec_data_q       <= vec_data;
        end
    end

    // FSM combinational logic
    always_comb begin

        next_state       = curr_state;
        item_count_d     = item_count_q;
        col_count_d      = col_count_q;
        bank_count_d     = bank_count_q;
        row_count_d      = row_count_q;
        row_odd_addr_d   = row_odd_addr_q;
        row_even_addr_d  = row_even_addr_q;
        odd_base_addr_d  = odd_base_addr_q;
        even_base_addr_d = even_base_addr_q;
        vec_addr_d       = vec_addr_q;
        acc_odd_d        = acc_odd_q;
        acc_even_d       = acc_even_q;
        M_scale_d        = M_scale_q;
        temp_val_odd_d   = temp_val_odd_q;
        temp_val_even_d  = temp_val_even_q;

        done      = 1'b0;
        wr_en_a   = 1'b0;
        wr_en_b   = 1'b0;
        wr_data_a = 0;
        wr_data_b = 0;
        wr_data_b = 0;

        case (curr_state)

            IDLE: begin
                if (start) begin
                    vec_addr_d       = vec_base_addr;
                    odd_base_addr_d  = mat_base_addr[11:2];
                    even_base_addr_d = (mat_base_addr + bytes_per_row) >> 2;

                    for (int i=0; i<MAX_BANKS; ++i) begin
                        if (i < num_of_banks) begin
                            row_odd_addr_d[i]  = odd_base_addr_d;
                            row_even_addr_d[i] = even_base_addr_d;
                        end
                    end

                    acc_odd_d    = '{default: '0};
                    acc_even_d   = '{default: '0};
                    item_count_d = 0;
                    col_count_d  = 1;
                    bank_count_d = 0;
                    row_count_d  = 0;
                    next_state   = WAIT1;
                end
            end

            WAIT1: begin
                vec_addr_d       = vec_addr_q + 1;
                col_count_d      = col_count_q + 1;
                next_state       = ADD;
            end

            ADD: begin
                for (int i=0; i<MAX_BANKS; ++i) begin
                    if (i < num_of_banks) begin
                        acc_odd_d[i]  = acc_odd_q[i] + $signed(row_odd_data_q[i][8*item_count_q[1:0] +: 8]) * vec_data_q;
                        acc_even_d[i] = acc_even_q[i] + $signed(row_even_data_q[i][8*item_count_q[1:0] +: 8]) * vec_data_q;
                    end
                end

                item_count_d = item_count_q + 1;
                col_count_d  = col_count_q + 1;

                if (item_count_q < bytes_per_row - 1) begin
                    for (int i=0; i<MAX_BANKS; ++i) begin
                        if (i < num_of_banks) begin
                            row_odd_addr_d[i]  = odd_base_addr_q + (col_count_q >> 2);
                            row_even_addr_d[i] = even_base_addr_q + (col_count_q >> 2);
                        end
                    end
                    vec_addr_d = vec_addr_q + 1;
                    next_state = ADD;
                end else begin
                    next_state  = WRITE1;
                end
            end

            WRITE1: begin
                // apply scaling
                temp_val_odd_d  = ((acc_odd_q[0] * $signed(M_scale_q)) + (1 <<< (S_MATVEC - 1))) >>> S_MATVEC;
                temp_val_even_d = ((acc_even_q[0] * $signed(M_scale_q)) + (1 <<< (S_MATVEC - 1))) >>> S_MATVEC;
                next_state      = WRITE2;
            end

            WRITE2: begin
                wr_en_a   = ((bank_count_q * rows_per_bank) + row_count_q) < VOCAB_SIZE;
                wr_en_b   = ((bank_count_q * rows_per_bank) + row_count_q + 1) < VOCAB_SIZE;

                if (temp_val_odd_q > 8'sd127)
                    wr_data_a = 8'h7F;
                else if (temp_val_odd_q < -8'sd127)
                    wr_data_a = 8'h81;
                else
                    wr_data_a = temp_val_odd_q[7:0];
                
                if (temp_val_even_q > 8'sd127)
                    wr_data_b = 8'h7F;
                else if (temp_val_even_q < -8'sd127)
                    wr_data_b = 8'h81;
                else
                    wr_data_b = temp_val_even_q[7:0];

                bank_count_d = bank_count_q + 1;

                for (int i=0; i<MAX_BANKS-1; ++i) begin
                    if (i < num_of_banks - 1) begin
                        acc_odd_d[i]  = acc_odd_q[i+1];
                        acc_even_d[i] = acc_even_q[i+1];
                    end
                end
                acc_odd_d[num_of_banks-1]  = 0;
                acc_even_d[num_of_banks-1] = 0;

                if (bank_count_q < num_of_banks - 1) begin
                    next_state = WRITE1;
                end else begin
                    next_state = WRITE3;
                end
            end

            WRITE3: begin
                bank_count_d     = 0;
                row_count_d      = row_count_q + 2;
                acc_odd_d        = '{default: '0};
                acc_even_d       = '{default: '0};
                item_count_d     = 0;
                col_count_d      = 0;
                odd_base_addr_d  = odd_base_addr_q + (bytes_per_row >> 1);
                even_base_addr_d = even_base_addr_q + (bytes_per_row >> 1);

                for (int i=0; i<MAX_BANKS; ++i) begin
                    if (i < num_of_banks)
                        row_odd_addr_d[i] = odd_base_addr_q + (bytes_per_row >> 1);
                end

                for (int i=0; i<MAX_BANKS; ++i) begin
                    if (i < num_of_banks)
                        row_even_addr_d[i] = even_base_addr_q + (bytes_per_row >> 1);
                end

                if (row_count_q < rows_per_bank - 2) begin
                    next_state = WAIT2;
                end else begin
                    next_state = DONE;
                end
            end

            WAIT2: begin
                vec_addr_d  = vec_base_addr;
                col_count_d = col_count_q + 1;
                next_state  = WAIT1;
            end

            DONE: begin
                done       = 1'b1;
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
            
        endcase
    end
    
endmodule