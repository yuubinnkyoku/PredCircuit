`timescale 1ns/1ps
`default_nettype none

module tb_pcalm_dual_update;
    localparam int DATA_W = 12;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic enable = 1'b0;
    logic clear_dual = 1'b0;
    logic mode_pcalm = 1'b0;
    logic signed [DATA_W-1:0] residual_q = '0;
    logic signed [DATA_W-1:0] dual_q;
    logic signed [DATA_W-1:0] credit_q;
    logic dual_saturated;
    logic credit_saturated;

    pcalm_dual_update #(
        .DATA_W(DATA_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .clear_dual(clear_dual),
        .mode_pcalm(mode_pcalm),
        .residual_q(residual_q),
        .dual_q(dual_q),
        .credit_q(credit_q),
        .dual_saturated(dual_saturated),
        .credit_saturated(credit_saturated)
    );

    always #5 clk = ~clk;

    task automatic tick;
        begin
            @(negedge clk);
            enable = 1'b1;
            @(posedge clk);
            #1;
            enable = 1'b0;
        end
    endtask

    task automatic check_value(
        input string label,
        input integer got,
        input integer expected
    );
        begin
            if (got !== expected) begin
                $display("FAIL %s: got %0d expected %0d", label, got, expected);
                $fatal(1);
            end
        end
    endtask

    task automatic check_bit(
        input string label,
        input logic got,
        input logic expected
    );
        begin
            if (got !== expected) begin
                $display("FAIL %s: got %0b expected %0b", label, got, expected);
                $fatal(1);
            end
        end
    endtask

    initial begin
        // Asynchronous reset.
        #2;
        rst_n = 1'b0;
        #4;
        rst_n = 1'b1;
        #1;
        check_value("reset dual", $signed(dual_q), 0);

        // sPC mode: dual is bypassed and credit = rho*r = r for rho=1.
        mode_pcalm = 1'b0;
        residual_q = 12'sd256; // +1.0 in Q3.8
        #1;
        check_value("spc credit", $signed(credit_q), 256);
        tick();
        check_value("spc dual stays zero", $signed(dual_q), 0);

        // PC-ALM mode. The credit visible before a tick uses the pre-dual lambda.
        mode_pcalm = 1'b1;
        residual_q = 12'sd256;
        #1;
        check_value("pcalm initial pre-dual credit", $signed(credit_q), 256);
        tick();
        // round(0.925 * 256) = 237 raw Q3.8 units.
        check_value("first dual update", $signed(dual_q), 237);
        check_value("credit after first update", $signed(credit_q), 493);
        check_bit("first dual not saturated", dual_saturated, 1'b0);

        tick();
        // round(0.99 * 237 + 0.925 * 256) = 471.
        check_value("second dual update", $signed(dual_q), 471);
        check_value("credit after second update", $signed(credit_q), 727);

        // Disabled update must hold lambda.
        residual_q = 12'sd100;
        enable = 1'b0;
        @(posedge clk);
        #1;
        check_value("disabled update holds dual", $signed(dual_q), 471);

        // Explicit minibatch/reset boundary for dual state.
        clear_dual = 1'b1;
        tick();
        clear_dual = 1'b0;
        check_value("clear dual", $signed(dual_q), 0);

        // Positive saturation: first step remains representable, second clips.
        residual_q = 12'sd2000;
        tick();
        check_value("positive large first dual", $signed(dual_q), 1850);
        check_bit("positive first dual not saturated", dual_saturated, 1'b0);
        check_bit("positive credit saturated", credit_saturated, 1'b1);
        tick();
        check_value("positive dual saturation", $signed(dual_q), 2047);
        check_bit("positive dual saturation flag", dual_saturated, 1'b1);

        // Switching to sPC must bypass the stale physical lambda immediately,
        // before the clock edge clears the register.
        mode_pcalm = 1'b0;
        residual_q = 12'sd100;
        #1;
        check_value("immediate spc bypass", $signed(credit_q), 100);
        tick();
        check_value("mode switch clears physical dual", $signed(dual_q), 0);

        // Negative saturation is symmetric.
        mode_pcalm = 1'b1;
        residual_q = -12'sd2000;
        tick();
        check_value("negative large first dual", $signed(dual_q), -1850);
        tick();
        check_value("negative dual saturation", $signed(dual_q), -2048);
        check_bit("negative dual saturation flag", dual_saturated, 1'b1);

        $display("PASS tb_pcalm_dual_update");
        $finish;
    end
endmodule

`default_nettype wire
