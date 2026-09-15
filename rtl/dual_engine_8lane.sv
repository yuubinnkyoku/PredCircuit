`timescale 1ns/1ps
`default_nettype none

// Eight-lane shared PC-ALM dual engine for the depth-32,width-8,batch-4 point.
// 124 words x 8 lanes x 12 bits = 992 persistent lambda values.
//
// The eight lanes are packed into one 96-bit memory word. Reads are synchronous
// and the read/compute/write path is pipelined with initiation interval 1, so a
// 124-address sweep needs 124 issue cycles plus one drain cycle. This structure
// avoids an asynchronous 8-bank read mux and is intentionally RAM-friendly.
// clear_all performs a 124-cycle sequential zero fill; the memory has no reset.
module dual_engine_8lane #(
    parameter int DATA_W = 12,
    parameter int DEPTH = 124,
    parameter int ADDR_W = 7,
    parameter int LANES = 8
) (
    input  logic clk,
    input  logic rst_n,
    input  logic enable,
    input  logic clear_all,
    input  logic [ADDR_W-1:0] addr,
    input  logic signed [DATA_W-1:0] residual_q [0:LANES-1],
    output logic signed [DATA_W-1:0] dual_q [0:LANES-1],
    output logic signed [DATA_W-1:0] credit_q [0:LANES-1],
    output logic [LANES-1:0] dual_saturated,
    output logic [LANES-1:0] credit_saturated,
    output logic clear_busy,
    output logic result_valid
);
    localparam int DYADIC_FRAC = 8;
    localparam int WIDE_W = DATA_W + DYADIC_FRAC + 3;
    localparam int WORD_W = LANES * DATA_W;
    localparam logic signed [DATA_W-1:0] DATA_MAX = {1'b0, {(DATA_W-1){1'b1}}};
    localparam logic signed [DATA_W-1:0] DATA_MIN = {1'b1, {(DATA_W-1){1'b0}}};

    // One packed word contains all lanes for an address. No reset on this array.
    logic [WORD_W-1:0] dual_mem [0:DEPTH-1];
    logic [WORD_W-1:0] read_word;
    logic [WORD_W-1:0] write_word;
    logic signed [DATA_W-1:0] residual_d [0:LANES-1];
    logic signed [DATA_W-1:0] dual_next [0:LANES-1];
    logic signed [WIDE_W-1:0] dual_ext [0:LANES-1];
    logic signed [WIDE_W-1:0] residual_ext [0:LANES-1];
    logic signed [WIDE_W-1:0] dual_scaled [0:LANES-1];
    logic signed [WIDE_W-1:0] dual_rounded [0:LANES-1];
    logic signed [WIDE_W-1:0] credit_wide [0:LANES-1];
    logic [LANES-1:0] dual_sat_next;
    logic [LANES-1:0] credit_sat_next;
    logic [ADDR_W-1:0] addr_d;
    logic [ADDR_W-1:0] clear_addr;
    logic valid_d;
    integer i;

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
        write_word = '0;
        dual_sat_next = '0;
        credit_sat_next = '0;
        for (i = 0; i < LANES; i = i + 1) begin
            dual_q[i] = $signed(read_word[i*DATA_W +: DATA_W]);
            dual_ext[i] = {{(WIDE_W-DATA_W){dual_q[i][DATA_W-1]}}, dual_q[i]};
            residual_ext[i] = {{(WIDE_W-DATA_W){residual_d[i][DATA_W-1]}}, residual_d[i]};

            // lambda' = (253*lambda + 237*r)/256, shifts/adds only.
            dual_scaled[i] =
                (dual_ext[i] <<< 8) - (dual_ext[i] <<< 1) - dual_ext[i]
                + (residual_ext[i] <<< 8) - (residual_ext[i] <<< 4)
                - (residual_ext[i] <<< 1) - residual_ext[i];
            dual_rounded[i] = round_shift_dyadic(dual_scaled[i]);
            credit_wide[i] = dual_ext[i] + residual_ext[i];

            if (dual_rounded[i] > $signed(DATA_MAX)) begin
                dual_next[i] = DATA_MAX;
                dual_sat_next[i] = 1'b1;
            end else if (dual_rounded[i] < $signed(DATA_MIN)) begin
                dual_next[i] = DATA_MIN;
                dual_sat_next[i] = 1'b1;
            end else begin
                dual_next[i] = dual_rounded[i][DATA_W-1:0];
            end

            if (credit_wide[i] > $signed(DATA_MAX)) begin
                credit_q[i] = DATA_MAX;
                credit_sat_next[i] = 1'b1;
            end else if (credit_wide[i] < $signed(DATA_MIN)) begin
                credit_q[i] = DATA_MIN;
                credit_sat_next[i] = 1'b1;
            end else begin
                credit_q[i] = credit_wide[i][DATA_W-1:0];
            end
            write_word[i*DATA_W +: DATA_W] = dual_next[i];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            clear_busy <= 1'b0;
            clear_addr <= '0;
            valid_d <= 1'b0;
            result_valid <= 1'b0;
            dual_saturated <= '0;
            credit_saturated <= '0;
        end else if (clear_all && !clear_busy) begin
            clear_busy <= 1'b1;
            clear_addr <= '0;
            valid_d <= 1'b0;
            result_valid <= 1'b0;
            dual_saturated <= '0;
            credit_saturated <= '0;
        end else if (clear_busy) begin
            dual_mem[clear_addr] <= '0;
            valid_d <= 1'b0;
            result_valid <= 1'b0;
            dual_saturated <= '0;
            credit_saturated <= '0;
            if (clear_addr == DEPTH-1) begin
                clear_busy <= 1'b0;
                clear_addr <= '0;
            end else begin
                clear_addr <= clear_addr + 1'b1;
            end
        end else begin
            // Retire the previous synchronous read/update.
            result_valid <= valid_d;
            if (valid_d) begin
                dual_mem[addr_d] <= write_word;
                dual_saturated <= dual_sat_next;
                credit_saturated <= credit_sat_next;
            end

            // Issue the next read. The residual and address travel with it.
            valid_d <= enable && (addr < DEPTH);
            if (enable && (addr < DEPTH)) begin
                read_word <= dual_mem[addr];
                addr_d <= addr;
                for (i = 0; i < LANES; i = i + 1)
                    residual_d[i] <= residual_q[i];
            end
        end
    end
endmodule

`default_nettype wire
