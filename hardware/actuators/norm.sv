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
    logic ms_en, ms_clr;
    (* USE_DSP = "yes" *) logic signed [23:0] ms_odd_q, ms_even_q;
    logic signed [31:0] ms_overall_d, ms_overall_q;
    logic [47:0] ms_scaled;
    logic [19:0] ms_real_d, ms_real_q;   // Q10.10
    logic [$clog2(N_EMBD):0] count_d, count_q;
    logic [ADDR_WIDTH-1:0] addr_odd_d, addr_odd_q;
    logic [ADDR_WIDTH-1:0] addr_even_d, addr_even_q;
    logic signed [47:0] product;                // Q6.26
    logic [15:0] y_d, y_q;                      // Q3.13
    logic [47:0] ms_y_prod_d, ms_y_prod_q;
    logic [47:0] P;                             // Q12.36 (P = ms_real * y^2)
    logic signed [26:0] P_scaled_d, P_scaled_q; // Q13.14
    logic signed [47:0] y_unscaled;             // Q21.27
    logic signed [31:0] y_shifted_d, y_shifted_q;
    logic [31:0] raw_idx_d, raw_idx_q;
    logic signed [23:0] rescaled_d, rescaled_q;
    logic signed [31:0] temp_val_odd_d, temp_val_odd_q;
    logic signed [31:0] temp_val_even_d, temp_val_even_q;


    assign addr_a = addr_odd_d;
    assign addr_b = addr_even_d;

    // FSM states
    typedef enum logic [3:0] {
        IDLE, ADD, LUT0, LUT1, LUT2, LUT3,
        RSQRT1, RSQRT2, RSQRT3, RSQRT4,
        RESCALE, WRITE1, WRITE2, DONE
    } state_t;

    state_t curr_state, next_state;

    // FSM sequential logic
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            curr_state      <= IDLE;
            ms_odd_q        <= 0;
            ms_even_q       <= 0;
            ms_overall_q    <= 0;
            ms_real_q       <= 0;
            count_q         <= 0;
            addr_odd_q      <= 0;
            addr_even_q     <= 0;
            y_q             <= 0;
            ms_y_prod_q     <= 0;
            P_scaled_q      <= 0;
            y_shifted_q     <= 0;
            raw_idx_q       <= 0;
            rescaled_q      <= 0;
            temp_val_odd_q  <= 0;
            temp_val_even_q <= 0;
        end else begin
            curr_state      <= next_state;
            ms_overall_q    <= ms_overall_d;
            ms_real_q       <= ms_real_d;
            count_q         <= count_d;
            addr_odd_q      <= addr_odd_d;
            addr_even_q     <= addr_even_d;
            y_q             <= y_d;
            ms_y_prod_q     <= ms_y_prod_d;
            P_scaled_q      <= P_scaled_d;
            y_shifted_q     <= y_shifted_d;
            raw_idx_q       <= raw_idx_d;
            rescaled_q      <= rescaled_d;
            temp_val_odd_q  <= temp_val_odd_d;
            temp_val_even_q <= temp_val_even_d;

            if (ms_clr) begin
                ms_odd_q  <= 0;
                ms_even_q <= 0;
            end else if (ms_en) begin
                ms_odd_q  <= ms_odd_q + $signed(rd_data_a) * $signed(rd_data_a);
                ms_even_q <= ms_even_q + $signed(rd_data_b) * $signed(rd_data_b);
            end
        end
    end

    // FSM combinational logic
    always_comb begin
        
        next_state      = curr_state;
        ms_overall_d    = ms_overall_q;
        ms_real_d       = ms_real_q;
        count_d         = count_q;
        addr_odd_d      = addr_odd_q;
        addr_even_d     = addr_even_q;
        y_d             = y_q;
        ms_y_prod_d     = ms_y_prod_q;
        P_scaled_d      = P_scaled_q;
        y_shifted_d     = y_shifted_q;
        raw_idx_d       = raw_idx_q;
        rescaled_d      = rescaled_q;
        temp_val_odd_d  = temp_val_odd_q;
        temp_val_even_d = temp_val_even_q;

        done        = 1'b0;
        wr_en_a     = 1'b0;
        wr_en_b     = 1'b0;
        idx         = 0;
        wr_data_a   = 0;
        wr_data_b   = 0;
        x_mem_we    = 1'b0;
        x_mem_addr  = 0;
        x_mem_wdata = 0;

        ms_en         = 1'b0;
        ms_clr        = 1'b0;
        ms_scaled     = 0;
        product       = 0;
        P             = 0;
        y_unscaled    = 0;

        case (curr_state)

            IDLE: begin
                if (start) begin
                    count_d      = 0;
                    ms_clr       = 1'b1;
                    ms_overall_d = 0;
                    addr_odd_d   = input_base_addr;
                    addr_even_d  = input_base_addr + 1;
                    next_state   = ADD;
                end
            end

            ADD: begin
                x_mem_we    = 1'b1;
                x_mem_addr  = count_q >> 1;
                x_mem_wdata = {rd_data_a, rd_data_b};
                ms_en       = 1'b1;
                count_d     = count_q + 2;

                if (count_q < N_EMBD - 2) begin
                    addr_odd_d  = addr_odd_q + 2;
                    addr_even_d = addr_even_q + 2;
                    next_state  = ADD;
                end else begin
                    next_state  = LUT0;
                end
            end

            LUT0: begin
                ms_overall_d = ms_odd_q + ms_even_q;
                next_state   = LUT1;
            end

            LUT1: begin
                ms_scaled  = ms_overall_q * M_ms_real;
                ms_real_d  = ms_scaled >>> (S_MS_REAL + $clog2(N_EMBD));
                next_state = LUT2;
            end

            LUT2: begin
                product    = ms_real_q * M_idx;
                raw_idx_d  = product >>> S_NORM;
                next_state = LUT3;
            end

            LUT3: begin
                idx        = (raw_idx_q >= 32'd31) ? 5'd31 : raw_idx_q[4:0];
                y_d        = lut_output;            // Q3.13
                next_state = RSQRT1;
            end

            RSQRT1: begin
                ms_y_prod_d = ms_real_q * y_q;
                next_state  = RSQRT2;
            end

            RSQRT2: begin
                P          = ms_y_prod_q * y_q;       // Q12.36
                P_scaled_d = 27'sd24576 - $signed({1'b0, P[47:23]});     // Q13.14
                next_state = RSQRT3;
            end

            RSQRT3: begin
                y_unscaled  = $signed({1'b0, y_q}) * (P_scaled_q);      // Q5.27
                y_shifted_d = y_unscaled >> 14;
                next_state  = RSQRT4;
            end

            RSQRT4: begin
                if (y_shifted_q < 32'sd0)
                    y_d = 16'd0;
                else if (y_shifted_q > 32'sd65535)
                    y_d = 16'hFFFF;
                else
                    y_d = y_shifted_q[15:0];

                count_d    = 0;
                next_state = RESCALE;
            end

            RESCALE: begin
                rescaled_d = $signed({1'b0, y_q}) * $signed(M_output);
                next_state = WRITE1;
            end

            WRITE1: begin
                temp_val_odd_d  = (($signed(x_mem[count_q >> 1][DATA_WIDTH*2-1:DATA_WIDTH]) * rescaled_q) + (1 << (S_NORM - 1))) >>> S_NORM;
                temp_val_even_d = (($signed(x_mem[count_q >> 1][DATA_WIDTH-1:0]) * rescaled_q) + (1 << (S_NORM - 1))) >>> S_NORM;
                next_state      = WRITE2;
            end

            WRITE2: begin
                wr_en_a     = 1'b1;
                wr_en_b     = 1'b1;
                addr_odd_d  = output_base_addr + count_q;
                addr_even_d = output_base_addr + count_q + 1;

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

                count_d = count_q + 2;

                if (count_q == N_EMBD - 2)
                    next_state = DONE;
                else 
                    next_state = WRITE1;
            end

            DONE: begin
                done       = 1'b1;
                next_state = IDLE;
            end
            
            default: next_state = IDLE;

        endcase
    end
    
endmodule