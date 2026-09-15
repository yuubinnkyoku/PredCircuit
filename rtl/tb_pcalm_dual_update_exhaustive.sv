`timescale 1ns/1ps
`default_nettype none

module tb_pcalm_dual_update_exhaustive;
    localparam int DATA_W = 12;
    localparam int DATA_MIN = -(1 <<< (DATA_W-1));
    localparam int DATA_MAX = (1 <<< (DATA_W-1)) - 1;

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

    integer expected_dual;
    integer expected_credit;
    integer magnitude;
    integer residual_i;
    integer dual_i;
    integer checked = 0;

    pcalm_dual_update #(.DATA_W(DATA_W)) dut (
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

    always #1 clk = ~clk;

    function automatic integer round_away_from_zero(input integer value);
        begin
            if (value >= 0) begin
                round_away_from_zero = (value + 128) >>> 8;
            end else begin
                magnitude = -value;
                round_away_from_zero = -((magnitude + 128) >>> 8);
            end
        end
    endfunction

    function automatic integer sat12(input integer value);
        begin
            if (value > DATA_MAX)
                sat12 = DATA_MAX;
            else if (value < DATA_MIN)
                sat12 = DATA_MIN;
            else
                sat12 = value;
        end
    endfunction

    task automatic load_dual(input integer value);
        begin
            // Exhaustive arithmetic checking intentionally seeds the physical
            // state directly.  Keep mode_pcalm asserted before doing so: if a
            // clock edge races with this task while mode_pcalm=0, the DUT's
            // sequential sPC-clear branch can overwrite the injected state.
            mode_pcalm = 1'b1;
            enable = 1'b0;
            clear_dual = 1'b0;
            @(negedge clk);
            dut.dual_reg = value;
            #0;
        end
    endtask

    task automatic check_pair(input integer d, input integer r);
        integer expected_scaled;
        integer expected_rounded;
        integer expected_dual_sat;
        integer expected_credit_sat;
        begin
            load_dual(d);
            residual_q = r;
            #0;

            expected_credit = sat12(d + r);
            expected_credit_sat = ((d + r) > DATA_MAX) || ((d + r) < DATA_MIN);
            if ($signed(credit_q) !== expected_credit || credit_saturated !== expected_credit_sat) begin
                $display("FAIL credit d=%0d r=%0d got=%0d/%0b exp=%0d/%0d",
                         d, r, $signed(credit_q), credit_saturated,
                         expected_credit, expected_credit_sat);
                $fatal(1);
            end

            expected_scaled = 253*d + 237*r;
            expected_rounded = round_away_from_zero(expected_scaled);
            expected_dual = sat12(expected_rounded);
            expected_dual_sat = (expected_rounded > DATA_MAX) || (expected_rounded < DATA_MIN);

            // Compare the combinational next-state arithmetic directly.  The
            // ordinary smoke test separately verifies the sequential register,
            // enable, clear and sPC-mode behavior.  Avoiding a clock per pair
            // also makes all 2^24 arithmetic combinations practical in CI.
            if ($signed(dut.dual_next) !== expected_dual ||
                dut.dual_saturated_next !== expected_dual_sat) begin
                $display("FAIL dual d=%0d r=%0d scaled=%0d rounded=%0d got=%0d/%0b exp=%0d/%0d",
                         d, r, expected_scaled, expected_rounded,
                         $signed(dut.dual_next), dut.dual_saturated_next,
                         expected_dual, expected_dual_sat);
                $fatal(1);
            end
            checked = checked + 1;
        end
    endtask

    initial begin
        // Reset and verify immediate sPC bypass independently of stored lambda.
        #1;
        rst_n = 1'b0;
        #2;
        rst_n = 1'b1;
        @(negedge clk);
        mode_pcalm = 1'b1;
        dut.dual_reg = 1234;
        #0;
        mode_pcalm = 1'b0;
        residual_q = -321;
        #0;
        if ($signed(credit_q) !== -321) $fatal(1, "sPC bypass mismatch");

        // Exhaust all 2^24 signed 12-bit (lambda,residual) combinations.
        // This checks rounding, both saturation boundaries and pre-dual credit.
        mode_pcalm = 1'b1;
        for (dual_i = DATA_MIN; dual_i <= DATA_MAX; dual_i = dual_i + 1) begin
            for (residual_i = DATA_MIN; residual_i <= DATA_MAX; residual_i = residual_i + 1) begin
                check_pair(dual_i, residual_i);
            end
        end

        $display("PASS tb_pcalm_dual_update_exhaustive checked=%0d", checked);
        $finish;
    end
endmodule

`default_nettype wire
