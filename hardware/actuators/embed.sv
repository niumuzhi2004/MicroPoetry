import addr_pkg::*;

module embed #(
    parameter int VOCAB_SIZE = 3005,
    parameter int BLOCK_SIZE = 96,
    parameter int N_EMBD     = 64,
    parameter int ADDR_WIDTH = 16,
    parameter int DATA_WIDTH = 8
) (
    input  logic clk,
    input  logic rst_n,

    // control logic 
    input  logic start,
    output logic done,
    input  logic [$clog2(VOCAB_SIZE)-1:0] token_id,
    input  logic [$clog2(BLOCK_SIZE)-1:0] pos_id,

    // weight ROM port
    input  logic [DATA_WIDTH-1:0] wte_data,
    input  logic [DATA_WIDTH-1:0] wpe_data,
    output logic [$clog2(VOCAB_SIZE*N_EMBD)-1:0] wte_addr,
    output logic [$clog2(BLOCK_SIZE*N_EMBD)-1:0] wpe_addr,

    // scratchpad port
    output logic wr_en,
    output logic [ADDR_WIDTH-1:0] wr_addr,
    output logic [DATA_WIDTH-1:0] wr_data
);

    logic [$clog2(N_EMBD):0] count_d, count_q;
    logic [$clog2(VOCAB_SIZE*N_EMBD)-1:0] wte_addr_d, wte_addr_q;
    logic [$clog2(BLOCK_SIZE*N_EMBD)-1:0] wpe_addr_d, wpe_addr_q;
    logic [ADDR_WIDTH-1:0] wr_addr_d, wr_addr_q;

    assign wte_addr = wte_addr_q;
    assign wpe_addr = wpe_addr_q;
    assign wr_addr  = wr_addr_q;

    typedef enum logic [1:0] {
        IDLE  = 2'b00,
        READ  = 2'b01,
        WRITE = 2'b10,
        DONE  = 2'b11
    } state_t;

    state_t curr_state, next_state;
    
    // FSM sequential logic
    always_ff @(posedge clk) begin
        if (~rst_n) begin
            curr_state <= IDLE;
            count_q    <= 0;
            wte_addr_q <= 0;
            wpe_addr_q <= 0;
            wr_addr_q  <= 0;
        end else begin
            curr_state <= next_state;
            count_q    <= count_d;
            wte_addr_q <= wte_addr_d;
            wpe_addr_q <= wpe_addr_d;
            wr_addr_q  <= wr_addr_d;
        end
    end

    // FSM combinational logic 
    always_comb begin

        next_state = curr_state;
        count_d    = count_q;
        wte_addr_d = wte_addr_q;
        wpe_addr_d = wpe_addr_q;
        wr_addr_d  = wr_addr_q;

        done    = 1'b0;
        wr_en   = 1'b0;
        wr_data = 0;

        case (curr_state)

            IDLE: begin
                count_d = 0;
                if (start) begin
                    wte_addr_d = token_id * N_EMBD;
                    wpe_addr_d = pos_id * N_EMBD;
                    wr_addr_d  = X_EMBD_BASE_ADDR;
                    next_state = READ;
                end
            end

            READ: next_state = WRITE;

            WRITE: begin
                wr_en      = 1'b1;
                wr_data    = wpe_data_q + wte_data_q;
                count_d    = count_q + 1;

                if (count_q < N_EMBD-1) begin
                    wte_addr_d = wte_addr_q + 1;
                    wpe_addr_d = wpe_addr_q + 1;
                    wr_addr_d  = wr_addr_q + 1;
                    next_state = READ;
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