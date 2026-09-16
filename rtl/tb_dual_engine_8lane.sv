`timescale 1ns/1ps
`default_nettype none

module tb_dual_engine_8lane;
    localparam int DATA_W = 12;
    localparam int LANES = 8;
    localparam int DEPTH = 124;
    localparam int ADDR_W = 7;

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

    dual_engine_8lane #(.DATA_W(DATA_W), .DEPTH(DEPTH), .ADDR_W(ADDR_W), .LANES(LANES)) dut (
        .clk, .rst_n, .enable, .clear_all, .addr, .residual_q,
        .dual_q, .credit_q, .dual_saturated, .credit_saturated,
        .clear_busy, .result_valid
    );

    always #5 clk = ~clk;

    task automatic set_all_residual(input logic signed [DATA_W-1:0] value);
        begin
            for (i = 0; i < LANES; i = i + 1)
                residual_q[i*DATA_W +: DATA_W] = value;
        end
    endtask

    task automatic expect_all(
        input logic signed [DATA_W-1:0] expected_dual,
        input logic signed [DATA_W-1:0] expected_credit
    );
        begin
            if (!result_valid) $fatal(1, "expected result_valid");
            for (i = 0; i < LANES; i = i + 1) begin
                if ($signed(dual_q[i*DATA_W +: DATA_W]) !== expected_dual)
                    $fatal(1, "lane %0d dual=%0d expected=%0d", i, $signed(dual_q[i*DATA_W +: DATA_W]), expected_dual);
                if ($signed(credit_q[i*DATA_W +: DATA_W]) !== expected_credit)
                    $fatal(1, "lane %0d credit=%0d expected=%0d", i, $signed(credit_q[i*DATA_W +: DATA_W]), expected_credit);
            end
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

        // First update at address 0 from lambda=0 with residual=256.
        // round((253*0 + 237*256)/256) = 237; credit reports old lambda+r = 256.
        addr <= 0;
        set_all_residual(12'sd256);
        enable <= 1;
        @(posedge clk);
        enable <= 0;
        set_all_residual('0);
        @(posedge clk);
        #1 expect_all(12'sd0, 12'sd256);

        // Leave a bubble so the write commits, then read/update address 0 again with r=0.
        @(posedge clk);
        addr <= 0;
        set_all_residual('0);
        enable <= 1;
        @(posedge clk);
        enable <= 0;
        @(posedge clk);
        #1 expect_all(12'sd237, 12'sd237);

        // The second update is round(253*237/256)=234 and must persist.
        @(posedge clk);
        addr <= 0;
        enable <= 1;
        @(posedge clk);
        enable <= 0;
        @(posedge clk);
        #1 expect_all(12'sd234, 12'sd234);

        $display("PASS: shared dual engine clear/read/update/write persistence");
        $finish;
    end
endmodule

`default_nettype wire
