`timescale 1ns/1ps
`default_nettype none

// Fair one-element credit core used to isolate the incremental hardware cost
// of the PC-ALM dual state.  Both variants expose the same ports and use the
// same residual + effective-dual credit/saturation path.  PCALM=0 compiles
// away the dual recurrence and effective_dual is identically zero.
module credit_core_common #(
    parameter int DATA_W = 12,
    parameter bit PCALM = 1'b1
) (
    input  logic                      clk,
    input  logic                      rst_n,
    input  logic                      enable,
    input  logic                      clear_dual,
    input  logic signed [DATA_W-1:0] residual_q,
    output logic signed [DATA_W-1:0] dual_q,
    output logic signed [DATA_W-1:0] credit_q,
    output logic                      dual_saturated,
    output logic                      credit_saturated
);
    localparam int DYADIC_FRAC = 8;
    localparam int WIDE_W = DATA_W + DYADIC_FRAC + 3;
    localparam logic signed [DATA_W-1:0] DATA_MAX = {1'b0, {(DATA_W-1){1'b1}}};
    localparam logic signed [DATA_W-1:0] DATA_MIN = {1'b1, {(DATA_W-1){1'b0}}};

    logic signed [DATA_W-1:0] dual_reg;
    logic signed [DATA_W-1:0] dual_next;
    logic signed [DATA_W-1:0] effective_dual;
    logic dual_saturated_next;
    logic signed [WIDE_W-1:0] dual_ext;
    logic signed [WIDE_W-1:0] residual_ext;
    logic signed [WIDE_W-1:0] dual_update_scaled;
    logic signed [WIDE_W-1:0] dual_update_rounded;
    logic signed [WIDE_W-1:0] credit_wide;

    function automatic logic signed [WIDE_W-1:0] round_shift_dyadic(
        input logic signed [WIDE_W-1:0] value
    );
        logic signed [WIDE_W:0] magnitude;
        logic signed [WIDE_W:0] rounded_magnitude;
        begin
            if (value >= 0) begin
                round_shift_dyadic =
                    (value + ({{(WIDE_W-1){1'b0}}, 1'b1} <<< (DYADIC_FRAC-1)))
                    >>> DYADIC_FRAC;
            end else begin
                magnitude = -$signed(value);
                rounded_magnitude =
                    (magnitude + ({{WIDE_W{1'b0}}, 1'b1} <<< (DYADIC_FRAC-1)))
                    >>> DYADIC_FRAC;
                round_shift_dyadic = -$signed(rounded_magnitude[WIDE_W-1:0]);
            end
        end
    endfunction

    always_comb begin
        effective_dual = PCALM ? dual_reg : '0;
        dual_ext = {{(WIDE_W-DATA_W){effective_dual[DATA_W-1]}}, effective_dual};
        residual_ext = {{(WIDE_W-DATA_W){residual_q[DATA_W-1]}}, residual_q};

        // Hardware-co-designed leaky PC-ALM recurrence:
        // lambda' = (253*lambda + 237*r) / 256.
        dual_update_scaled =
            (dual_ext <<< 8) - (dual_ext <<< 1) - dual_ext
            + (residual_ext <<< 8) - (residual_ext <<< 4)
            - (residual_ext <<< 1) - residual_ext;
        dual_update_rounded = round_shift_dyadic(dual_update_scaled);

        // Identical credit path in both compiled variants.  With PCALM=0,
        // effective_dual=0 and this is exactly sPC credit=r for rho=1.
        credit_wide = residual_ext + dual_ext;

        dual_saturated_next = 1'b0;
        if (dual_update_rounded > $signed(DATA_MAX)) begin
            dual_next = DATA_MAX;
            dual_saturated_next = PCALM;
        end else if (dual_update_rounded < $signed(DATA_MIN)) begin
            dual_next = DATA_MIN;
            dual_saturated_next = PCALM;
        end else begin
            dual_next = dual_update_rounded[DATA_W-1:0];
        end

        credit_saturated = 1'b0;
        if (credit_wide > $signed(DATA_MAX)) begin
            credit_q = DATA_MAX;
            credit_saturated = 1'b1;
        end else if (credit_wide < $signed(DATA_MIN)) begin
            credit_q = DATA_MIN;
            credit_saturated = 1'b1;
        end else begin
            credit_q = credit_wide[DATA_W-1:0];
        end
    end

    generate
        if (PCALM) begin : gen_pcalm
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    dual_reg <= '0;
                    dual_saturated <= 1'b0;
                end else if (clear_dual) begin
                    dual_reg <= '0;
                    dual_saturated <= 1'b0;
                end else if (enable) begin
                    dual_reg <= dual_next;
                    dual_saturated <= dual_saturated_next;
                end
            end
        end else begin : gen_spc
            always_comb begin
                dual_reg = '0;
                dual_saturated = 1'b0;
            end
        end
    endgenerate

    assign dual_q = dual_reg;
endmodule

module credit_core_spc #(
    parameter int DATA_W = 12
) (
    input logic clk,
    input logic rst_n,
    input logic enable,
    input logic clear_dual,
    input logic signed [DATA_W-1:0] residual_q,
    output logic signed [DATA_W-1:0] dual_q,
    output logic signed [DATA_W-1:0] credit_q,
    output logic dual_saturated,
    output logic credit_saturated
);
    credit_core_common #(.DATA_W(DATA_W), .PCALM(1'b0)) impl (.*);
endmodule

module credit_core_pcalm #(
    parameter int DATA_W = 12
) (
    input logic clk,
    input logic rst_n,
    input logic enable,
    input logic clear_dual,
    input logic signed [DATA_W-1:0] residual_q,
    output logic signed [DATA_W-1:0] dual_q,
    output logic signed [DATA_W-1:0] credit_q,
    output logic dual_saturated,
    output logic credit_saturated
);
    credit_core_common #(.DATA_W(DATA_W), .PCALM(1'b1)) impl (.*);
endmodule

`default_nettype wire
