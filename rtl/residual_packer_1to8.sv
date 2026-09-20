// Pack a scalar residual stream into 8-lane words for the shared PC-ALM dual engine.
//
// The producer may supply one scalar every cycle.  A separate output register
// holds each completed word, so assembling the next word does not overwrite a
// stalled output word.  Backpressure is required only if a completed output
// word remains unconsumed when the next group of eight scalars completes.
module residual_packer_1to8 #(
    parameter integer WIDTH = 12,
    parameter integer LANES = 8
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire signed [WIDTH-1:0]      in_data,
    input  wire                         in_valid,
    output wire                         in_ready,
    output reg  signed [LANES*WIDTH-1:0] out_data,
    output reg                          out_valid,
    input  wire                         out_ready
);
    localparam integer COUNT_W = $clog2(LANES);

    reg signed [LANES*WIDTH-1:0] assemble_q;
    reg [COUNT_W-1:0] count_q;

    // A new scalar is safe unless it would complete a word while the previous
    // completed word is still blocked.  This permits continuous 1 scalar/cycle
    // input when the consumer accepts one 8-lane word at least every 8 cycles.
    wire completing = (count_q == LANES-1);
    assign in_ready = !completing || !out_valid || out_ready;

    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            assemble_q <= '0;
            count_q    <= '0;
            out_data   <= '0;
            out_valid  <= 1'b0;
        end else begin
            if (out_valid && out_ready)
                out_valid <= 1'b0;

            if (in_valid && in_ready) begin
                assemble_q[count_q*WIDTH +: WIDTH] <= in_data;
                if (completing) begin
                    // Nonblocking assignment means explicitly splice the
                    // just-arriving final lane into the completed word.
                    out_data <= assemble_q;
                    out_data[(LANES-1)*WIDTH +: WIDTH] <= in_data;
                    out_valid <= 1'b1;
                    count_q <= '0;
                end else begin
                    count_q <= count_q + 1'b1;
                end
            end
        end
    end
endmodule
