`timescale 1ns/1ps
`default_nettype none

module tb_residual_serializer_64to8;
    localparam int DATA_W = 12;
    localparam int IN_LANES = 64;
    localparam int OUT_LANES = 8;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic in_valid;
    logic in_ready;
    logic signed [IN_LANES*DATA_W-1:0] residual_in;
    logic out_valid;
    logic out_ready;
    logic signed [OUT_LANES*DATA_W-1:0] residual_out;
    logic [2:0] chunk_index;
    integer i;
    integer seen;
    integer cycles_since_first;

    residual_serializer_64to8 dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_ready(in_ready),
        .residual_in(residual_in), .out_valid(out_valid), .out_ready(out_ready),
        .residual_out(residual_out), .chunk_index(chunk_index)
    );

    always #5 clk = ~clk;

    task automatic load_vector(input integer base);
        begin
            for (i = 0; i < IN_LANES; i = i + 1)
                residual_in[i*DATA_W +: DATA_W] = $signed(base + i);
        end
    endtask

    // Sample each beat on the falling edge, before the following rising edge
    // advances index_q.  This avoids a testbench race that previously skipped
    // chunk 0 even though the serializer itself had emitted it correctly.
    task automatic check_vector(input integer base);
        integer lane;
        integer expected;
        begin
            seen = 0;
            while (seen < 8) begin
                if (out_valid) begin
                    if (chunk_index !== seen[2:0])
                        $fatal(1, "chunk order mismatch: got %0d expected %0d", chunk_index, seen);
                    for (lane = 0; lane < OUT_LANES; lane = lane + 1) begin
                        expected = base + seen*OUT_LANES + lane;
                        if ($signed(residual_out[lane*DATA_W +: DATA_W]) !== expected)
                            $fatal(1, "payload mismatch chunk=%0d lane=%0d got=%0d expected=%0d",
                                   seen, lane, $signed(residual_out[lane*DATA_W +: DATA_W]), expected);
                    end
                    seen = seen + 1;
                end
                if (seen < 8) @(negedge clk);
            end
        end
    endtask

    initial begin
        in_valid = 1'b0;
        out_ready = 1'b1;
        residual_in = '0;
        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        if (!in_ready) $fatal(1, "serializer not ready after reset");
        load_vector(0);
        in_valid = 1'b1;
        @(negedge clk);
        in_valid = 1'b0;
        check_vector(0);
        @(negedge clk);
        if (!in_ready) $fatal(1, "serializer did not drain after 8 beats");

        // A shared 64-MAC engine needs >=64 cycles for the next 64x64 vector.
        // After the eight service beats above, model the remaining gap.
        cycles_since_first = 0;
        while (cycles_since_first < 55) begin
            @(negedge clk);
            if (!in_ready) $fatal(1, "unexpected backlog during producer gap");
            cycles_since_first = cycles_since_first + 1;
        end

        if (!in_ready) $fatal(1, "second vector would stall producer");
        load_vector(512);
        in_valid = 1'b1;
        @(negedge clk);
        in_valid = 1'b0;
        check_vector(512);

        // Backpressure must hold both payload and chunk index bit-exactly.
        @(negedge clk);
        if (!in_ready) $fatal(1, "serializer not ready for backpressure test");
        load_vector(1024);
        in_valid = 1'b1;
        @(negedge clk);
        in_valid = 1'b0;
        out_ready = 1'b0;
        repeat (3) begin
            @(negedge clk);
            if (!out_valid || chunk_index !== 0)
                $fatal(1, "backpressure failed to hold first chunk");
        end
        out_ready = 1'b1;
        check_vector(1024);

        $display("PASS: 64-to-8 serializer preserves order and drains each width64 burst in 8 beats");
        $finish;
    end
endmodule

`default_nettype wire
