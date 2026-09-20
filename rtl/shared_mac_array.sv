// Shared matrix MAC array for the first PredCircuit synthesis sweep.
//
// This block intentionally contains no PC-ALM-specific state.  The same
// physical multiplier/accumulator array is meant to serve W*h and W^T*c
// phases, so P can be swept independently of the fixed V=8 element lanes.
module shared_mac_array #(
    parameter integer P = 8,
    parameter integer DATA_W = 14,
    parameter integer WEIGHT_W = 14,
    parameter integer ACC_W = 40
) (
    input  wire                         clk,
    input  wire                         rst,
    input  wire                         in_valid,
    input  wire                         clear_acc,
    input  wire                         last,
    input  wire [P*DATA_W-1:0]          data_in,
    input  wire [P*WEIGHT_W-1:0]        weight_in,
    output reg  signed [ACC_W-1:0]      acc_out,
    output reg                          out_valid,
    output reg  [63:0]                  mac_active_cycles,
    output reg  [63:0]                  mac_count
);
    integer i;
    reg signed [ACC_W-1:0] lane_sum;
    reg signed [ACC_W-1:0] product_ext;
    reg signed [DATA_W-1:0] data_lane;
    reg signed [WEIGHT_W-1:0] weight_lane;
    reg signed [DATA_W+WEIGHT_W-1:0] product_lane;

    // P independent signed products are summed before the accumulator.
    // Keeping P as the only scaling parameter makes the first P={8,16,32,64}
    // synthesis sweep directly interpretable.
    always @* begin
        lane_sum = {ACC_W{1'b0}};
        product_ext = {ACC_W{1'b0}};
        data_lane = {DATA_W{1'b0}};
        weight_lane = {WEIGHT_W{1'b0}};
        product_lane = {(DATA_W+WEIGHT_W){1'b0}};
        for (i = 0; i < P; i = i + 1) begin
            data_lane = $signed(data_in[i*DATA_W +: DATA_W]);
            weight_lane = $signed(weight_in[i*WEIGHT_W +: WEIGHT_W]);
            product_lane = data_lane * weight_lane;
            product_ext = {{(ACC_W-(DATA_W+WEIGHT_W)){product_lane[DATA_W+WEIGHT_W-1]}}, product_lane};
            lane_sum = lane_sum + product_ext;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            acc_out <= {ACC_W{1'b0}};
            out_valid <= 1'b0;
            mac_active_cycles <= 64'd0;
            mac_count <= 64'd0;
        end else begin
            out_valid <= 1'b0;
            if (in_valid) begin
                mac_active_cycles <= mac_active_cycles + 64'd1;
                mac_count <= mac_count + P;
                if (clear_acc)
                    acc_out <= lane_sum;
                else
                    acc_out <= acc_out + lane_sum;
                if (last)
                    out_valid <= 1'b1;
            end
        end
    end
endmodule
