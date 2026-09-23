// Width-N weight source with explicit W and W^T copies.
// A scalar write keeps both orientations coherent.  A read exposes one
// complete row per cycle, selected from W or W^T, for a P=N MAC array.
module weight_source_dual_orientation #(
    parameter integer N = 8,
    parameter integer WEIGHT_W = 14,
    parameter integer ADDR_W = $clog2(N)
) (
    input  wire                         clk,
    input  wire                         wr_en,
    input  wire [ADDR_W-1:0]            wr_row,
    input  wire [ADDR_W-1:0]            wr_col,
    input  wire signed [WEIGHT_W-1:0]   wr_data,
    input  wire                         transpose,
    input  wire [ADDR_W-1:0]            rd_row,
    output reg  [N*WEIGHT_W-1:0]        weight_out
);
    reg signed [WEIGHT_W-1:0] w  [0:N*N-1];
    reg signed [WEIGHT_W-1:0] wt [0:N*N-1];
    integer k;

    always @(posedge clk) begin
        if (wr_en) begin
            w [wr_row*N + wr_col] <= wr_data;
            wt[wr_col*N + wr_row] <= wr_data;
        end
    end

    always @* begin
        weight_out = {(N*WEIGHT_W){1'b0}};
        for (k = 0; k < N; k = k + 1) begin
            if (transpose)
                weight_out[k*WEIGHT_W +: WEIGHT_W] = wt[rd_row*N + k];
            else
                weight_out[k*WEIGHT_W +: WEIGHT_W] = w[rd_row*N + k];
        end
    end
endmodule
