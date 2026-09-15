`timescale 1ns/1ps
`default_nettype none

// Eight-lane shared PC-ALM dual engine for the depth-32,width-8,batch-4 point.
// 8 banks x 124 entries = 992 persistent lambda values. One address selects
// one entry in every bank, so a full dual sweep takes exactly 124 cycles.
module dual_engine_8lane #(
    parameter int DATA_W = 12,
    parameter int DEPTH = 124,
    parameter int ADDR_W = 7
) (
    input  logic clk,
    input  logic rst_n,
    input  logic enable,
    input  logic clear_all,
    input  logic [ADDR_W-1:0] addr,
    input  logic signed [DATA_W-1:0] residual_q [0:7],
    output logic signed [DATA_W-1:0] dual_q [0:7],
    output logic signed [DATA_W-1:0] credit_q [0:7],
    output logic [7:0] dual_saturated,
    output logic [7:0] credit_saturated
);
    localparam int DYADIC_FRAC = 8;
    localparam int WIDE_W = DATA_W + DYADIC_FRAC + 3;
    localparam logic signed [DATA_W-1:0] DATA_MAX = {1'b0, {(DATA_W-1){1'b1}}};
    localparam logic signed [DATA_W-1:0] DATA_MIN = {1'b1, {(DATA_W-1){1'b0}}};

    logic signed [DATA_W-1:0] dual_mem [0:7][0:DEPTH-1];
    logic signed [DATA_W-1:0] dual_next [0:7];
    logic signed [WIDE_W-1:0] dual_scaled [0:7];
    logic signed [WIDE_W-1:0] dual_rounded [0:7];
    logic signed [WIDE_W-1:0] credit_wide [0:7];
    logic [7:0] dual_sat_next;
    integer i, j;

    function automatic logic signed [WIDE_W-1:0] round_shift_dyadic(
        input logic signed [WIDE_W-1:0] value
    );
        logic signed [WIDE_W:0] magnitude;
        logic signed [WIDE_W:0] rounded_magnitude;
        begin
            if (value >= 0) begin
                round_shift_dyadic = (value + (1 <<< (DYADIC_FRAC-1))) >>> DYADIC_FRAC;
            end else begin
                magnitude = -$signed(value);
                rounded_magnitude = (magnitude + (1 <<< (DYADIC_FRAC-1))) >>> DYADIC_FRAC;
                round_shift_dyadic = -$signed(rounded_magnitude[WIDE_W-1:0]);
            end
        end
    endfunction

    always_comb begin
        for (i = 0; i < 8; i = i + 1) begin
            dual_q[i] = dual_mem[i][addr];
            // lambda' = (253*lambda + 237*r)/256, using shifts/adds only.
            dual_scaled[i] =
                ($signed(dual_q[i]) <<< 8) - ($signed(dual_q[i]) <<< 1) - $signed(dual_q[i])
                + ($signed(residual_q[i]) <<< 8) - ($signed(residual_q[i]) <<< 4)
                - ($signed(residual_q[i]) <<< 1) - $signed(residual_q[i]);
            dual_rounded[i] = round_shift_dyadic(dual_scaled[i]);
            credit_wide[i] = $signed(dual_q[i]) + $signed(residual_q[i]);

            dual_sat_next[i] = 1'b0;
            if (dual_rounded[i] > $signed(DATA_MAX)) begin
                dual_next[i] = DATA_MAX;
                dual_sat_next[i] = 1'b1;
            end else if (dual_rounded[i] < $signed(DATA_MIN)) begin
                dual_next[i] = DATA_MIN;
                dual_sat_next[i] = 1'b1;
            end else begin
                dual_next[i] = dual_rounded[i][DATA_W-1:0];
            end

            credit_saturated[i] = 1'b0;
            if (credit_wide[i] > $signed(DATA_MAX)) begin
                credit_q[i] = DATA_MAX;
                credit_saturated[i] = 1'b1;
            end else if (credit_wide[i] < $signed(DATA_MIN)) begin
                credit_q[i] = DATA_MIN;
                credit_saturated[i] = 1'b1;
            end else begin
                credit_q[i] = credit_wide[i][DATA_W-1:0];
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 8; i = i + 1)
                for (j = 0; j < DEPTH; j = j + 1)
                    dual_mem[i][j] <= '0;
            dual_saturated <= '0;
        end else if (clear_all) begin
            for (i = 0; i < 8; i = i + 1)
                for (j = 0; j < DEPTH; j = j + 1)
                    dual_mem[i][j] <= '0;
            dual_saturated <= '0;
        end else if (enable && addr < DEPTH) begin
            for (i = 0; i < 8; i = i + 1)
                dual_mem[i][addr] <= dual_next[i];
            dual_saturated <= dual_sat_next;
        end
    end
endmodule

`default_nettype wire
