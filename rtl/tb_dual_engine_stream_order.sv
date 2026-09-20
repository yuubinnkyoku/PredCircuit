`timescale 1ns/1ps
`default_nettype none

// Ordering test for the shared dual engine under a scheduler-like II=1 stream.
// It checks the PC-ALM pre_dual_energy contract: the emitted credit must use
// lambda_t, while the same transaction writes lambda_{t+1}.  Consecutive words
// are issued without bubbles, and the next sweep starts immediately after the
// previous sweep, so this also exercises the natural same-address reuse gap.
module tb_dual_engine_stream_order;
    localparam int DATA_W = 12;
    localparam int LANES = 8;
    localparam int DEPTH = 16;
    localparam int ADDR_W = 4;

    logic clk = 0;
    logic rst_n = 0;
    logic enable = 0;
    logic clear_all = 0;
    logic [ADDR_W-1:0] addr = '0;
    logic signed [LANES*DATA_W-1:0] residual_q = '0;
    logic signed [LANES*DATA_W-1:0] dual_q;
    logic signed [LANES*DATA_W-1:0] credit_q;
    logic [LANES-1:0] dual_saturated;
    logic [LANES-1:0] credit_saturated;
    logic clear_busy;
    logic result_valid;
    integer i;
    integer seen;

    dual_engine_8lane #(.DATA_W(DATA_W), .DEPTH(DEPTH), .ADDR_W(ADDR_W), .LANES(LANES)) dut (
        .clk, .rst_n, .enable, .clear_all, .addr, .residual_q,
        .dual_q, .credit_q, .dual_saturated, .credit_saturated,
        .clear_busy, .result_valid
    );

    always #5 clk = ~clk;

    task automatic set_residual(input logic signed [DATA_W-1:0] value);
        integer lane;
        begin
            for (lane = 0; lane < LANES; lane = lane + 1)
                residual_q[lane*DATA_W +: DATA_W] = value;
        end
    endtask

    task automatic check_result(
        input logic signed [DATA_W-1:0] expected_old_dual,
        input logic signed [DATA_W-1:0] expected_credit
    );
        integer lane;
        begin
            if (!result_valid) $fatal(1, "missing streamed result");
            for (lane = 0; lane < LANES; lane = lane + 1) begin
                if ($signed(dual_q[lane*DATA_W +: DATA_W]) !== expected_old_dual)
                    $fatal(1, "lane %0d old dual=%0d expected=%0d", lane,
                           $signed(dual_q[lane*DATA_W +: DATA_W]), expected_old_dual);
                if ($signed(credit_q[lane*DATA_W +: DATA_W]) !== expected_credit)
                    $fatal(1, "lane %0d credit=%0d expected=%0d", lane,
                           $signed(credit_q[lane*DATA_W +: DATA_W]), expected_credit);
            end
            if (|dual_saturated || |credit_saturated)
                $fatal(1, "unexpected saturation in ordering test");
        end
    endtask

    initial begin
        repeat (2) @(posedge clk);
        rst_n <= 1;
        @(posedge clk);
        clear_all <= 1;
        @(posedge clk);
        clear_all <= 0;
        wait (clear_busy);
        wait (!clear_busy);
        @(posedge clk);

        // Sweep 1: lambda_0=0, r=256.  Credit must be 256 (old lambda+r),
        // while lambda_1=round(237*256/256)=237 is written back.
        set_residual(12'sd256);
        enable <= 1;
        for (i = 0; i < DEPTH; i = i + 1) begin
            addr <= i[ADDR_W-1:0];
            @(posedge clk);
            if (i > 0) begin
                #1 check_result(12'sd0, 12'sd256);
            end
        end
        enable <= 0;
        @(posedge clk);
        #1 check_result(12'sd0, 12'sd256);

        // Start sweep 2 with no scheduler bubble beyond the pipeline drain.
        // Every address must now expose lambda_1=237.  With r=64 the credit is
        // 301; observing lambda_2 here instead would be a post-dual ordering bug.
        set_residual(12'sd64);
        enable <= 1;
        for (i = 0; i < DEPTH; i = i + 1) begin
            addr <= i[ADDR_W-1:0];
            @(posedge clk);
            if (i > 0) begin
                #1 check_result(12'sd237, 12'sd301);
            end
        end
        enable <= 0;
        @(posedge clk);
        #1 check_result(12'sd237, 12'sd301);

        $display("PASS: II=1 streamed dual engine preserves pre-dual credit ordering");
        $finish;
    end
endmodule

`default_nettype wire
