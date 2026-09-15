`timescale 1ns/1ps
`default_nettype none

module spc_credit #(
    parameter int DATA_W = 12,
    parameter int COEF_FRAC = 12,
    parameter int signed RHO_Q = 4096
) (
    input  logic signed [DATA_W-1:0] residual_q,
    output logic signed [DATA_W-1:0] credit_q,
    output logic                         credit_saturated
);
    localparam int WIDE_W = DATA_W + 34;
    localparam logic signed [DATA_W-1:0] DATA_MAX = {1'b0, {(DATA_W-1){1'b1}}};
    localparam logic signed [DATA_W-1:0] DATA_MIN = {1'b1, {(DATA_W-1){1'b0}}};

    logic signed [WIDE_W-1:0] credit_scaled;
    logic signed [WIDE_W-1:0] credit_rounded;

    function automatic logic signed [WIDE_W-1:0] round_shift_nearest(
        input logic signed [WIDE_W-1:0] value
    );
        logic signed [WIDE_W:0] magnitude;
        logic signed [WIDE_W:0] rounded_magnitude;
        begin
            if (COEF_FRAC == 0) begin
                round_shift_nearest = value;
            end else if (value >= 0) begin
                round_shift_nearest =
                    (value + ({{(WIDE_W-1){1'b0}}, 1'b1} <<< (COEF_FRAC-1))) >>> COEF_FRAC;
            end else begin
                magnitude = -$signed(value);
                rounded_magnitude =
                    (magnitude + ({{WIDE_W{1'b0}}, 1'b1} <<< (COEF_FRAC-1))) >>> COEF_FRAC;
                round_shift_nearest = -$signed(rounded_magnitude[WIDE_W-1:0]);
            end
        end
    endfunction

    always_comb begin
        credit_scaled = $signed(residual_q) * RHO_Q;
        credit_rounded = round_shift_nearest(credit_scaled);
        credit_saturated = 1'b0;
        if (credit_rounded > $signed(DATA_MAX)) begin
            credit_q = DATA_MAX;
            credit_saturated = 1'b1;
        end else if (credit_rounded < $signed(DATA_MIN)) begin
            credit_q = DATA_MIN;
            credit_saturated = 1'b1;
        end else begin
            credit_q = credit_rounded[DATA_W-1:0];
        end
    end
endmodule

`default_nettype wire
