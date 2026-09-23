// Width-N single-copy weight source.  The same physical register matrix
// exposes either a row (W) or a column (W^T).  Synthesis measures the mux /
// routing price of transpose access without duplicating logical weights.
module weight_source_single_copy #(
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
    reg signed [WEIGHT_W-1:0] w [0:N*N-1];
    integer k;

    always @(posedge clk) begin
        if (wr_en)
            w[wr_row*N + wr_col] <= wr_data;
    end

    always @* begin
        weight_out = {(N*WEIGHT_W){1'b0}};
        for (k = 0; k < N; k = k + 1) begin
            if (transpose)
                weight_out[k*WEIGHT_W +: WEIGHT_W] = w[k*N + rd_row];
            else
                weight_out[k*WEIGHT_W +: WEIGHT_W] = w[rd_row*N + k];
        end
    end
endmodule
