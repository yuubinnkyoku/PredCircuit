module weight_mac_dual_orientation_top #(
    parameter integer N = 8,
    parameter integer DATA_W = 14,
    parameter integer WEIGHT_W = 14,
    parameter integer ACC_W = 40,
    parameter integer ADDR_W = $clog2(N)
) (
    input  wire                       clk,
    input  wire                       rst,
    input  wire                       wr_en,
    input  wire [ADDR_W-1:0]          wr_row,
    input  wire [ADDR_W-1:0]          wr_col,
    input  wire signed [WEIGHT_W-1:0] wr_data,
    input  wire                       transpose,
    input  wire [ADDR_W-1:0]          rd_row,
    input  wire                       in_valid,
    input  wire                       clear_acc,
    input  wire                       last,
    input  wire [N*DATA_W-1:0]        data_in,
    output wire signed [ACC_W-1:0]    acc_out,
    output wire                       out_valid,
    output wire [63:0]                mac_active_cycles,
    output wire [63:0]                mac_count
);
    wire [N*WEIGHT_W-1:0] weight_bus;

    weight_source_dual_orientation #(
        .N(N), .WEIGHT_W(WEIGHT_W), .ADDR_W(ADDR_W)
    ) weights (
        .clk(clk), .wr_en(wr_en), .wr_row(wr_row), .wr_col(wr_col),
        .wr_data(wr_data), .transpose(transpose), .rd_row(rd_row),
        .weight_out(weight_bus)
    );

    shared_mac_array #(
        .P(N), .DATA_W(DATA_W), .WEIGHT_W(WEIGHT_W), .ACC_W(ACC_W)
    ) mac (
        .clk(clk), .rst(rst), .in_valid(in_valid), .clear_acc(clear_acc),
        .last(last), .data_in(data_in), .weight_in(weight_bus),
        .acc_out(acc_out), .out_valid(out_valid),
        .mac_active_cycles(mac_active_cycles), .mac_count(mac_count)
    );
endmodule
