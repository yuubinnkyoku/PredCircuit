`timescale 1ns/1ps
`default_nettype none

// Eight-lane shared PC-ALM dual engine for the depth-32,width-8,batch-4 point.
// 124 words x 8 lanes x 12 bits = 992 persistent lambda values.
//
// The eight lanes are packed into one 96-bit memory word. Reads are synchronous
// and the read/compute/write path is pipelined with initiation interval 1, so a
// 124-address sweep needs 124 issue cycles plus one drain cycle. clear_all uses
// the same single write port for a 124-cycle zero fill. The RAM process itself
// contains no reset or control-register assignments, keeping RAM inference clean.
//
// External lane ports are packed vectors rather than unpacked-array ports because
// the Yosys 0.33 Verilog frontend used by CI does not accept unpacked arrays in
// module ports. Lane i occupies [i*DATA_W +: DATA_W].
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
    input  logic signed [LANES*DATA_W-1:0] residual_q,
    output logic signed [LANES*DATA_W-1:0] dual_q,
    output logic signed [LANES*DATA_W-1:0] credit_q,
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

    logic [WORD_W-1:0] dual_mem [0:DEPTH-1];
    logic [WORD_W-1:0] read_word;
    logic [WORD_W-1:0] write_word;
    logic [WORD_W-1:0] ram_write_data;
    logic [ADDR_W-1:0] ram_write_addr;
    logic ram_write_en;
    logic ram_read_en;
    logic signed [DATA_W-1:0] residual_d [0:LANES-1];
    logic signed [DATA_W-1:0] dual_lane [0:LANES-1];
    logic signed [DATA_W-1:0] credit_lane [0:LANES-1];
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
        dual_q = '0;
        credit_q = '0;
        dual_sat_next = '0;
        credit_sat_next = '0;
        for (i = 0; i < LANES; i = i + 1) begin
            dual_lane[i] = $signed(read_word[i*DATA_W +: DATA_W]);
            dual_ext[i] = {{(WIDE_W-DATA_W){dual_lane[i][DATA_W-1]}}, dual_lane[i]};
            residual_ext[i] = {{(WIDE_W-DATA_W){residual_d[i][DATA_W-1]}}, residual_d[i]};
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
                credit_lane[i] = DATA_MAX;
                credit_sat_next[i] = 1'b1;
            end else if (credit_wide[i] < $signed(DATA_MIN)) begin
                credit_lane[i] = DATA_MIN;
                credit_sat_next[i] = 1'b1;
            end else begin
                credit_lane[i] = credit_wide[i][DATA_W-1:0];
            end
            dual_q[i*DATA_W +: DATA_W] = dual_lane[i];
            credit_q[i*DATA_W +: DATA_W] = credit_lane[i];
            write_word[i*DATA_W +: DATA_W] = dual_next[i];
        end

        ram_write_en = clear_busy || valid_d;
        ram_write_addr = clear_busy ? clear_addr : addr_d;
        ram_write_data = clear_busy ? '0 : write_word;
        ram_read_en = enable && !clear_busy && (addr < DEPTH);
    end

    always_ff @(posedge clk) begin
        if (ram_write_en)
            dual_mem[ram_write_addr] <= ram_write_data;
        if (ram_read_en)
            read_word <= dual_mem[addr];
    end

    always_ff @(posedge clk) begin
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
            result_valid <= valid_d;
            if (valid_d) begin
                dual_saturated <= dual_sat_next;
                credit_saturated <= credit_sat_next;
            end

            valid_d <= enable && (addr < DEPTH);
            if (enable && (addr < DEPTH)) begin
                addr_d <= addr;
                for (i = 0; i < LANES; i = i + 1)
                    residual_d[i] <= $signed(residual_q[i*DATA_W +: DATA_W]);
            end
        end
    end
endmodule

`default_nettype wire