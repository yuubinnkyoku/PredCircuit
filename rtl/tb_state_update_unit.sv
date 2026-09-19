`timescale 1ns/1ps
`default_nettype none

module tb_state_update_unit;
    reg  signed [14:0] state_in;
    reg  signed [13:0] update_in;
    wire signed [14:0] state_out;
    wire saturated;
    wire zero_step;

    integer errors;

    state_update_unit #(
        .STATE_W(15),
        .UPDATE_W(14),
        .UPDATE_TO_STATE_SHIFT(-1)
    ) dut (
        .state_in(state_in),
        .update_in(update_in),
        .state_out(state_out),
        .saturated(saturated),
        .zero_step(zero_step)
    );

    task check;
        input signed [14:0] s;
        input signed [13:0] u;
        input signed [14:0] expected;
        input expected_sat;
        input expected_zero;
        begin
            state_in = s;
            update_in = u;
            #1;
            if ((state_out !== expected) || (saturated !== expected_sat) || (zero_step !== expected_zero)) begin
                $display("FAIL s=%0d u=%0d got=%0d sat=%0b zero=%0b expected=%0d sat=%0b zero=%0b",
                         s, u, state_out, saturated, zero_step, expected, expected_sat, expected_zero);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0;
        // update has one extra fractional bit: +/-1 is a half-state-LSB tie.
        check(15'sd100, 14'sd0,   15'sd100, 1'b0, 1'b1);
        check(15'sd100, 14'sd1,   15'sd99,  1'b0, 1'b0);
        check(15'sd100, -14'sd1,  15'sd101, 1'b0, 1'b0);
        check(15'sd100, 14'sd2,   15'sd99,  1'b0, 1'b0);
        check(15'sd100, -14'sd2,  15'sd101, 1'b0, 1'b0);
        check(15'sd100, 14'sd3,   15'sd98,  1'b0, 1'b0);
        check(15'sd100, -14'sd3,  15'sd102, 1'b0, 1'b0);
        check(15'sd16383, -14'sd2, 15'sd16383, 1'b1, 1'b1);
        check(-15'sd16384, 14'sd2, -15'sd16384, 1'b1, 1'b1);

        if (errors != 0) begin
            $fatal(1, "state_update_unit: %0d failures", errors);
        end
        $display("PASS state_update_unit directed arithmetic tests");
        $finish;
    end
endmodule

`default_nettype wire
