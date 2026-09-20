`timescale 1ns/1ps
`default_nettype none
module tb_pcalm_stream_frontend;
    localparam DATA_W=12;
    localparam LANES=8;
    localparam DEPTH=16;
    localparam ADDR_W=4;
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0, clear_all=0;
    reg signed [DATA_W-1:0] residual_in=0;
    reg residual_valid=0;
    wire residual_ready;
    wire signed [LANES*DATA_W-1:0] credit_q, dual_q;
    wire [LANES-1:0] dual_saturated, credit_saturated;
    wire result_valid, clear_busy;
    wire [ADDR_W-1:0] issue_addr;
    wire [31:0] scalar_accept_count, word_issue_count;
    integer i, lane, results, stalls;
    integer expected;

    pcalm_stream_frontend #(.DATA_W(DATA_W),.LANES(LANES),.DEPTH(DEPTH),.ADDR_W(ADDR_W)) dut (
        .clk(clk),.rst_n(rst_n),.clear_all(clear_all),
        .residual_in(residual_in),.residual_valid(residual_valid),.residual_ready(residual_ready),
        .credit_q(credit_q),.dual_q(dual_q),.dual_saturated(dual_saturated),
        .credit_saturated(credit_saturated),.result_valid(result_valid),.clear_busy(clear_busy),
        .issue_addr(issue_addr),.scalar_accept_count(scalar_accept_count),.word_issue_count(word_issue_count)
    );

    task send64;
        input integer base;
        begin
            stalls=0;
            for (i=0;i<64;i=i+1) begin
                @(negedge clk);
                residual_in = base+i;
                residual_valid = 1;
                if (!residual_ready) stalls=stalls+1;
                while (!residual_ready) @(negedge clk);
                @(posedge clk);
            end
            @(negedge clk); residual_valid=0;
            if (stalls != 0) begin $display("FAIL producer stalls=%0d",stalls); $fatal; end
        end
    endtask

    task collect8_first_sweep;
        input integer base;
        begin
            results=0;
            while (results<8) begin
                @(negedge clk);
                if (result_valid) begin
                    for (lane=0;lane<LANES;lane=lane+1) begin
                        expected=base+results*8+lane;
                        if ($signed(credit_q[lane*DATA_W +: DATA_W]) !== expected) begin
                            $display("FAIL credit word=%0d lane=%0d got=%0d expected=%0d",results,lane,$signed(credit_q[lane*DATA_W +: DATA_W]),expected); $fatal;
                        end
                    end
                    results=results+1;
                end
            end
        end
    endtask

    initial begin
        repeat(3) @(posedge clk); rst_n=1;
        @(negedge clk); clear_all=1;
        @(posedge clk); @(negedge clk); clear_all=0;
        wait(clear_busy); wait(!clear_busy);
        // First 64 scalars occupy addresses 0..7.  Clearing guarantees lambda=0,
        // so pre-update-lambda credit must equal the residual exactly.
        fork
            send64(17);
            collect8_first_sweep(17);
        join
        if (scalar_accept_count !== 64) begin $display("FAIL scalar count %0d",scalar_accept_count); $fatal; end
        if (word_issue_count !== 8) begin $display("FAIL word count %0d",word_issue_count); $fatal; end
        if (issue_addr !== 8) begin $display("FAIL issue addr %0d",issue_addr); $fatal; end
        if (|dual_saturated || |credit_saturated) begin $display("FAIL unexpected saturation"); $fatal; end
        $display("PASS streamed frontend: 64 scalars, 8 words, zero producer stalls, pre-dual credit exact");
        $finish;
    end
endmodule
`default_nettype wire
