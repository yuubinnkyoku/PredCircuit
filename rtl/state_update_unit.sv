`default_nettype none

// Multiplier-free state update for the current PC-ALM hardware point:
//   h_next = sat_state(h - q_update(g_h))
// with effective state step eta_eff = 1.
//
// STATE_W and UPDATE_W may use different fractional precision. UPDATE_TO_STATE_SHIFT
// is the binary-point conversion applied before subtraction:
//   > 0 : arithmetic left shift update into state units
//   < 0 : round-to-nearest, ties away from zero, then arithmetic right shift
module state_update_unit #(
    parameter integer STATE_W = 15,
    parameter integer UPDATE_W = 14,
    parameter integer UPDATE_TO_STATE_SHIFT = -1
) (
    input  wire signed [STATE_W-1:0] state_in,
    input  wire signed [UPDATE_W-1:0] update_in,
    output reg  signed [STATE_W-1:0] state_out,
    output reg                         saturated,
    output reg                         zero_step
);
    localparam integer SHIFT_ABS = (UPDATE_TO_STATE_SHIFT < 0) ? -UPDATE_TO_STATE_SHIFT : UPDATE_TO_STATE_SHIFT;
    localparam integer WORK_W = STATE_W + UPDATE_W + SHIFT_ABS + 3;

    reg signed [WORK_W-1:0] state_ext;
    reg signed [WORK_W-1:0] update_ext;
    reg signed [WORK_W-1:0] update_state_units;
    reg signed [WORK_W-1:0] magnitude;
    reg signed [WORK_W-1:0] rounded_magnitude;
    reg signed [WORK_W-1:0] candidate;
    reg signed [WORK_W-1:0] state_max;
    reg signed [WORK_W-1:0] state_min;

    always @* begin
        state_ext = {{(WORK_W-STATE_W){state_in[STATE_W-1]}}, state_in};
        update_ext = {{(WORK_W-UPDATE_W){update_in[UPDATE_W-1]}}, update_in};

        if (UPDATE_TO_STATE_SHIFT > 0) begin
            update_state_units = update_ext <<< UPDATE_TO_STATE_SHIFT;
        end else if (UPDATE_TO_STATE_SHIFT < 0) begin
            // Symmetric round-to-nearest, ties away from zero.  Work on magnitude
            // so signed arithmetic right-shift does not bias negative values.
            magnitude = (update_ext < 0) ? -update_ext : update_ext;
            rounded_magnitude = (magnitude + ({{(WORK_W-1){1'b0}}, 1'b1} <<< (SHIFT_ABS-1))) >>> SHIFT_ABS;
            update_state_units = (update_ext < 0) ? -rounded_magnitude : rounded_magnitude;
        end else begin
            update_state_units = update_ext;
        end

        candidate = state_ext - update_state_units;
        state_max = ({{(WORK_W-1){1'b0}}, 1'b1} <<< (STATE_W-1)) - 1;
        state_min = -({{(WORK_W-1){1'b0}}, 1'b1} <<< (STATE_W-1));
        saturated = 1'b0;

        if (candidate > state_max) begin
            state_out = {1'b0, {(STATE_W-1){1'b1}}};
            saturated = 1'b1;
        end else if (candidate < state_min) begin
            state_out = {1'b1, {(STATE_W-1){1'b0}}};
            saturated = 1'b1;
        end else begin
            state_out = candidate[STATE_W-1:0];
        end

        zero_step = (state_out == state_in);
    end
endmodule

`default_nettype wire
