`default_nettype none

module pcalm_dual_update #(
    parameter int DATA_W = 12,
    parameter int COEF_FRAC = 12,
    // Q*.COEF_FRAC constants for alpha=0.925, retain=(1-leak)=0.99, rho=1.
    parameter int signed ALPHA_Q = 3789,
    parameter int signed RETAIN_Q = 4055,
    parameter int signed RHO_Q = 4096
) (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         enable,
    input  logic                         clear_dual,
    input  logic                         mode_pcalm,
    input  logic signed [DATA_W-1:0]    residual_q,
    output logic signed [DATA_W-1:0]    dual_q,
    output logic signed [DATA_W-1:0]    credit_q,
    output logic                         dual_saturated,
    output logic                         credit_saturated
);
    // Coefficients are 32-bit signed parameters, so DATA_W+32 bits are enough
    // for one product. Two guard bits cover the following additions.
    localparam int WIDE_W = DATA_W + 34;
    localparam logic signed [DATA_W-1:0] DATA_MAX = {1'b0, {(DATA_W-1){1'b1}}};
    localparam logic signed [DATA_W-1:0] DATA_MIN = {1'b1, {(DATA_W-1){1'b0}}};

    logic signed [DATA_W-1:0] dual_reg;
    logic signed [DATA_W-1:0] dual_next;
    logic signed [DATA_W-1:0] effective_dual;

    logic signed [WIDE_W-1:0] dual_update_scaled;
    logic signed [WIDE_W-1:0] credit_scaled;
    logic signed [WIDE_W-1:0] dual_update_rounded;
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
        // sPC is the alpha=0 / no-dual special case. Bypass lambda
        // combinationally as soon as mode_pcalm is low; do not wait for the
        // next clock edge that clears the physical dual register.
        effective_dual = mode_pcalm ? dual_reg : '0;

        // Keep lambda and residual in the same DATA_W fixed-point format.
        // Multiplication by the Q*.COEF_FRAC constants adds COEF_FRAC
        // fractional bits; round once after the complete affine expression.
        dual_update_scaled =
            $signed(effective_dual) * RETAIN_Q + $signed(residual_q) * ALPHA_Q;
        credit_scaled =
            $signed(residual_q) * RHO_Q + ($signed(effective_dual) <<< COEF_FRAC);

        dual_update_rounded = round_shift_nearest(dual_update_scaled);
        credit_rounded = round_shift_nearest(credit_scaled);

        dual_saturated = 1'b0;
        if (dual_update_rounded > $signed(DATA_MAX)) begin
            dual_next = DATA_MAX;
            dual_saturated = mode_pcalm;
        end else if (dual_update_rounded < $signed(DATA_MIN)) begin
            dual_next = DATA_MIN;
            dual_saturated = mode_pcalm;
        end else begin
            dual_next = dual_update_rounded[DATA_W-1:0];
        end

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

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dual_reg <= '0;
        end else if (clear_dual || !mode_pcalm) begin
            dual_reg <= '0;
        end else if (enable) begin
            dual_reg <= dual_next;
        end
    end

    assign dual_q = dual_reg;

endmodule

`default_nettype wire
