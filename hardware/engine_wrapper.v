`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08/22/2026 04:21:25 PM
// Design Name: 
// Module Name: engine_wrapper
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module engine_wrapper #(
    parameter integer C_S_AXI_ADDR_WIDTH = 9,
    parameter integer C_S_AXI_DATA_WIDTH = 32,

    parameter integer LAYER_NUM = 4,
    parameter integer N_HEAD = 4,
    parameter integer BLOCK_SIZE = 96,
    parameter integer DATA_WIDTH = 8,
    parameter integer ADDR_WIDTH = 16,
    parameter integer N_EMBD = 64,
    parameter integer ZE_CHARS = 1848,
    parameter integer N_TEMPLATE = 4,
    parameter integer POEM_LEN = 56,
    parameter integer PROGRAM_LEN = 115,
    parameter integer INSTR_WIDTH = 12,
    parameter integer TITLE_SIZE = 12,
    parameter integer VOCAB_SIZE = 3005
) (
    input wire S_AXI_ACLK,
    input wire S_AXI_ARESETN,
    input wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR,
    input wire S_AXI_AWVALID,
    output wire S_AXI_AWREADY,
    input wire [C_S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA,
    input wire [3:0] S_AXI_WSTRB,
    input wire S_AXI_WVALID,
    output wire S_AXI_WREADY,
    input wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR,
    input wire S_AXI_ARVALID,
    output wire S_AXI_ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0] S_AXI_RDATA,
    output wire [1:0] S_AXI_RRESP,
    output wire S_AXI_RVALID,
    input wire S_AXI_RREADY,
    output wire [1:0] S_AXI_BRESP,
    output wire S_AXI_BVALID,
    input wire S_AXI_BREADY
);

    engine_top #(
        .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH),
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH),
        .LAYER_NUM(LAYER_NUM),
        .N_HEAD(N_HEAD),
        .BLOCK_SIZE(BLOCK_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .N_EMBD(N_EMBD),
        .ZE_CHARS(ZE_CHARS),
        .N_TEMPLATE(N_TEMPLATE),
        .POEM_LEN(POEM_LEN),
        .PROGRAM_LEN(PROGRAM_LEN),
        .INSTR_WIDTH(INSTR_WIDTH),
        .TITLE_SIZE(TITLE_SIZE),
        .VOCAB_SIZE(VOCAB_SIZE)
    ) engine_top_inst (
        .S_AXI_ACLK(S_AXI_ACLK),
        .S_AXI_ARESETN(S_AXI_ARESETN),
        .S_AXI_AWADDR(S_AXI_AWADDR),
        .S_AXI_AWVALID(S_AXI_AWVALID),
        .S_AXI_AWREADY(S_AXI_AWREADY),
        .S_AXI_WDATA(S_AXI_WDATA),
        .S_AXI_WSTRB(S_AXI_WSTRB),
        .S_AXI_WVALID(S_AXI_WVALID),
        .S_AXI_WREADY(S_AXI_WREADY),
        .S_AXI_ARADDR(S_AXI_ARADDR),
        .S_AXI_ARVALID(S_AXI_ARVALID),
        .S_AXI_ARREADY(S_AXI_ARREADY),
        .S_AXI_RDATA(S_AXI_RDATA),
        .S_AXI_RRESP(S_AXI_RRESP),
        .S_AXI_RVALID(S_AXI_RVALID),
        .S_AXI_RREADY(S_AXI_RREADY),
        .S_AXI_BRESP(S_AXI_BRESP),
        .S_AXI_BVALID(S_AXI_BVALID),
        .S_AXI_BREADY(S_AXI_BREADY)
    );
endmodule
