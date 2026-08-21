import proj_pkg::*;

module norm #(
    parameter int LAYER_NUM  = 4,
    parameter int N_EMBD     = 64,
    parameter int ADDR_WIDTH = 16,
    parameter int DATA_WIDTH = 8
) (
    input  logic clk,
    input  logic rst_n,

    // control logic
    input  logic start,
    output logic done,
    input  norm_param_t param,
    input  logic [$clog2(LAYER_NUM)-1:0] layer,
    
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

    // determine input and output addresses in scratchpad
    logic [ADDR_WIDTH-1:0] input_base_addr, output_base_addr;
    assign input_base_addr  = (param == FINAL_NORM) ? SUM_RESIDUAL_BASE_ADDR : X_EMBD_BASE_ADDR;
    assign output_base_addr = X_NORM_BASE_ADDR;

    // scaling for each norm calculation
    logic [15:0] M_ms_real, M_idx, M_output;
    logic signed [31:0] temp_val_odd, temp_val_even;

    always_comb begin
        case (param)
            EMB_NORM: begin
                M_ms_real = M_EMB_NORM_MS_REAL;
                M_idx     = M_EMB_NORM_IDX;
                M_output  = M_EMB_NORM_OUTPUT;
            end
            X_NORM: begin
                M_ms_real = (layer == 0) ? M_X_NORM_LAYER0_MS_REAL : M_X_NORM_LAYER123_MS_REAL;
                M_idx     = M_X_NORM_IDX;
                M_output  = (layer == 0) ? M_X_NORM_LAYER0_OUTPUT : M_X_NORM_LAYER123_OUTPUT;
            end
            PRE_MLP_NORM: begin
                M_ms_real = M_PRE_MLP_NORM_MS_REAL;
                M_idx     = M_PRE_MLP_NORM_IDX;
                M_output  = M_PRE_MLP_NORM_OUTPUT;
            end
            FINAL_NORM: begin
                M_ms_real = M_FINAL_NORM_MS_REAL;
                M_idx     = M_FINAL_NORM_IDX;
                M_output  = M_FINAL_NORM_OUTPUT;
            end
            default: begin
                M_ms_real = 16'h0001;
                M_idx     = 16'h0001;
                M_output  = 16'h0001;
            end
        endcase
    end

    // internal storage of x input values
    logic [DATA_WIDTH*2-1:0] x_mem [N_EMBD>>1];
    logic [DATA_WIDTH*2-1:0] x_mem_wdata;
    logic [$clog2(N_EMBD>>1)-1:0] x_mem_addr;
    logic x_mem_we;

    always_ff @(posedge clk) begin
        if (x_mem_we)
            x_mem[x_mem_addr] <= x_mem_wdata;
    end

    // LUT for rsqrt
    logic [4:0]  idx;
    logic [15:0] lut_output;

    rsqrt_lut lut (
        .param(param),
        .addr(idx),
        .data(lut_output)
    );

    // flip-flop signals
    logic signed [31:0] ms_d, ms_q;
    logic [47:0] ms_scaled;
    logic [19:0] ms_real_d, ms_real_q;   // Q10.10
    logic [$clog2(N_EMBD):0] count_d, count_q;
    logic [ADDR_WIDTH-1:0] addr_odd_d, addr_odd_q;
    logic [ADDR_WIDTH-1:0] addr_even_d, addr_even_q;
    logic signed [47:0] product;                // Q6.26
    logic [15:0] y_d, y_q;                      // Q3.13
    logic [47:0] ms_y_prod;
    logic [47:0] P;                             // Q12.36 (P = ms_real * y^2)
    logic signed [26:0] P_scaled;               // Q13.14
    logic signed [47:0] y_unscaled;             // Q21.27
    logic signed [31:0] y_shifted;
    logic [31:0] raw_idx;

    assign addr_a = addr_odd_d;
    assign addr_b = addr_even_d;

    // FSM states
    typedef enum logic [2:0] {
        IDLE  = 3'b000,
        ADD   = 3'b001,
        LUT   = 3'b010,
        RSQRT = 3'b011,
        WRITE = 3'b100,
        DONE  = 3'b101
    } state_t;

    state_t curr_state, next_state;

    // FSM sequential logic
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            curr_state  <= IDLE;
            ms_q        <= 32'b0;
            ms_real_q   <= 16'b0;
            count_q     <= 0;
            addr_odd_q  <= 0;
            addr_even_q <= 0;
            y_q         <= 16'b0;
        end else begin
            curr_state  <= next_state;
            ms_q        <= ms_d;
            ms_real_q   <= ms_real_d;
            count_q     <= count_d;
            addr_odd_q  <= addr_odd_d;
            addr_even_q <= addr_even_d;
            y_q         <= y_d;
        end
    end

    // FSM combinational logic
    always_comb begin
        
        next_state  = curr_state;
        ms_d        = ms_q;
        ms_real_d   = ms_real_q;
        count_d     = count_q;
        addr_odd_d  = addr_odd_q;
        addr_even_d = addr_even_q;
        y_d         = y_q;

        done        = 1'b0;
        wr_en_a     = 1'b0;
        wr_en_b     = 1'b0;
        idx         = 5'b0;
        raw_idx     = 6'd0;
        wr_data_a   = 0;
        wr_data_b   = 0;
        x_mem_we    = 1'b0;
        x_mem_addr  = 0;
        x_mem_wdata = 0;

        case (curr_state)

            IDLE: begin
                if (start) begin
                    count_d     = 0;
                    ms_d        = 0;
                    addr_odd_d  = input_base_addr;
                    addr_even_d = input_base_addr + 1;
                    next_state  = ADD;
                end
            end

            ADD: begin
                x_mem_we    = 1'b1;
                x_mem_addr  = count_q >> 1;
                x_mem_wdata = {rd_data_a, rd_data_b};
                ms_d        = ms_q + $signed(rd_data_a) * $signed(rd_data_a) + $signed(rd_data_b) * $signed(rd_data_b);
                count_d     = count_q + 2;

                if (count_q < N_EMBD - 2) begin
                    addr_odd_d  = addr_odd_q + 2;
                    addr_even_d = addr_even_q + 2;
                    next_state  = ADD;
                end else begin
                    next_state  = LUT;
                end
            end

            LUT: begin
                ms_scaled  = ms_q * M_ms_real;
                ms_real_d  = ms_scaled >>> (S_MS_REAL + $clog2(N_EMBD));
                product    = ms_real_d * M_idx;
                raw_idx    = product >>> S_NORM;
                idx        = (raw_idx >= 32'd31) ? 5'd31 : raw_idx[4:0];
                y_d        = lut_output;            // Q3.13
                next_state = RSQRT;
            end

            RSQRT: begin
                ms_y_prod  = ms_real_q * y_q;
                P          = ms_y_prod * y_q;       // Q12.36
                P_scaled   = 27'sd24576 - $signed({1'b0, P[47:23]});     // Q13.14
                y_unscaled = $signed({1'b0, y_q}) * (P_scaled);      // Q5.27
                y_shifted  = y_unscaled >> 14;

                if (y_shifted < 32'sd0)
                    y_d = 16'd0;
                else if (y_shifted > 32'sd65535)
                    y_d = 16'hFFFF;
                else
                    y_d = y_shifted[15:0];

                count_d    = 0;
                next_state = WRITE;
            end

            WRITE: begin
                wr_en_a       = 1'b1;
                wr_en_b       = 1'b1;
                addr_odd_d    = output_base_addr + count_q;
                addr_even_d   = output_base_addr + count_q + 1;
                temp_val_odd  = ($signed(x_mem[count_q >> 1][DATA_WIDTH*2-1:DATA_WIDTH]) * $signed({1'b0, y_q}) * $signed(M_output)) >>> S_NORM;
                temp_val_even = ($signed(x_mem[count_q >> 1][DATA_WIDTH-1:0]) * $signed({1'b0, y_q}) * $signed(M_output)) >>> S_NORM;
                
                if (temp_val_odd > 8'sd127)
                    wr_data_a = 8'h7F;
                else if (temp_val_odd < -8'sd127)
                    wr_data_a = 8'h81;
                else
                    wr_data_a = temp_val_odd[7:0];

                if (temp_val_even > 8'sd127)
                    wr_data_b = 8'h7F;
                else if (temp_val_even < -8'sd127)
                    wr_data_b = 8'h81;
                else
                    wr_data_b = temp_val_even[7:0];

                count_d = count_q + 2;

                if (count_q == N_EMBD - 2)
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