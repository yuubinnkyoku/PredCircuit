`timescale 1ns/1ps
`default_nettype none

// Streamed PC-ALM front end for the width-64 design point.
// One residual scalar/cycle is packed into 8-lane words and issued directly to
// the shared dual engine.  The dual address advances only on a packed-word issue.
module pcalm_stream_frontend #(
    parameter integer DATA_W = 12,
    parameter integer LANES = 8,
    parameter integer DEPTH = 992,
    parameter integer ADDR_W = 10
) (
    input  wire clk,
    input  wire rst_n,
    input  wire clear_all,
    input  wire signed [DATA_W-1:0] residual_in,
    input  wire residual_valid,
    output wire residual_ready,
    output wire signed [LANES*DATA_W-1:0] credit_q,
    output wire signed [LANES*DATA_W-1:0] dual_q,
    output wire [LANES-1:0] dual_saturated,
    output wire [LANES-1:0] credit_saturated,
    output wire result_valid,
    output wire clear_busy,
    output reg  [ADDR_W-1:0] issue_addr,
    output reg  [31:0] scalar_accept_count,
    output reg  [31:0] word_issue_count
);
    wire signed [LANES*DATA_W-1:0] packed_residual;
    wire packed_valid;
    wire packed_ready;
    wire scalar_ready_i;

    // dual_engine_8lane has II=1 and no downstream ready input.  It accepts a
    // word whenever enabled and not clearing, so this is the exact ready signal.
    assign packed_ready = !clear_busy;
    assign residual_ready = scalar_ready_i && !clear_busy;

    residual_packer_1to8 #(
        .WIDTH(DATA_W), .LANES(LANES)
    ) u_packer (
        .clk(clk), .rst_n(rst_n),
        .in_data(residual_in),
        .in_valid(residual_valid && !clear_busy),
        .in_ready(scalar_ready_i),
        .out_data(packed_residual),
        .out_valid(packed_valid),
        .out_ready(packed_ready)
    );

    dual_engine_8lane #(
        .DATA_W(DATA_W), .DEPTH(DEPTH), .ADDR_W(ADDR_W), .LANES(LANES)
    ) u_dual (
        .clk(clk), .rst_n(rst_n),
        .enable(packed_valid && packed_ready),
        .clear_all(clear_all),
        .addr(issue_addr),
        .residual_q(packed_residual),
        .dual_q(dual_q), .credit_q(credit_q),
        .dual_saturated(dual_saturated),
        .credit_saturated(credit_saturated),
        .clear_busy(clear_busy), .result_valid(result_valid)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            issue_addr <= '0;
            scalar_accept_count <= 0;
            word_issue_count <= 0;
        end else begin
            if (clear_all && !clear_busy) begin
                issue_addr <= '0;
                scalar_accept_count <= 0;
                word_issue_count <= 0;
            end else begin
                if (residual_valid && residual_ready)
                    scalar_accept_count <= scalar_accept_count + 1;
                if (packed_valid && packed_ready) begin
                    word_issue_count <= word_issue_count + 1;
                    if (issue_addr == DEPTH-1)
                        issue_addr <= '0;
                    else
                        issue_addr <= issue_addr + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
