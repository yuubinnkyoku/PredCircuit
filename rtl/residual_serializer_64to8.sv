`timescale 1ns/1ps
`default_nettype none

// One-vector elastic serializer for the width-64 PC-ALM design point.
// Accepts one 64-scalar residual vector atomically when ready and emits eight
// consecutive 8-scalar words.  This is deliberately only one vector deep: for
// a shared <=64-MAC pool, a new dense 64x64 residual vector cannot be produced
// in fewer than 64 cycles, while this block drains in 8 cycles.
module residual_serializer_64to8 #(
    parameter int DATA_W = 12,
    parameter int IN_LANES = 64,
    parameter int OUT_LANES = 8
) (
    input  logic clk,
    input  logic rst_n,
    input  logic in_valid,
    output logic in_ready,
    input  logic signed [IN_LANES*DATA_W-1:0] residual_in,
    output logic out_valid,
    input  logic out_ready,
    output logic signed [OUT_LANES*DATA_W-1:0] residual_out,
    output logic [$clog2(IN_LANES/OUT_LANES)-1:0] chunk_index
);
    localparam int CHUNKS = IN_LANES / OUT_LANES;
    localparam int CHUNK_W = OUT_LANES * DATA_W;
    localparam int IDX_W = $clog2(CHUNKS);

    logic signed [IN_LANES*DATA_W-1:0] buffer_q;
    logic [IDX_W-1:0] index_q;
    logic busy_q;

    initial begin
        if ((IN_LANES % OUT_LANES) != 0)
            $error("IN_LANES must be divisible by OUT_LANES");
    end

    assign in_ready = !busy_q;
    assign out_valid = busy_q;
    assign chunk_index = index_q;
    assign residual_out = buffer_q[index_q*CHUNK_W +: CHUNK_W];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            buffer_q <= '0;
            index_q <= '0;
            busy_q <= 1'b0;
        end else begin
            if (in_valid && in_ready) begin
                buffer_q <= residual_in;
                index_q <= '0;
                busy_q <= 1'b1;
            end else if (out_valid && out_ready) begin
                if (index_q == CHUNKS-1) begin
                    index_q <= '0;
                    busy_q <= 1'b0;
                end else begin
                    index_q <= index_q + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
