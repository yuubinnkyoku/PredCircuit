`timescale 1ns/1ps
module tb_residual_packer_1to8;
    localparam WIDTH=12;
    localparam LANES=8;
    reg clk=0, rst_n=0;
    reg signed [WIDTH-1:0] in_data;
    reg in_valid;
    wire in_ready;
    wire signed [LANES*WIDTH-1:0] out_data;
    wire out_valid;
    reg out_ready;
    integer sent, received, lane;
    integer expected_base;

    residual_packer_1to8 #(.WIDTH(WIDTH), .LANES(LANES)) dut (
        .clk(clk), .rst_n(rst_n), .in_data(in_data), .in_valid(in_valid),
        .in_ready(in_ready), .out_data(out_data), .out_valid(out_valid),
        .out_ready(out_ready)
    );
    always #5 clk=~clk;

    task check_word;
        input integer base;
        begin
            for (lane=0; lane<LANES; lane=lane+1)
                if ($signed(out_data[lane*WIDTH +: WIDTH]) !== base+lane) begin
                    $display("FAIL lane=%0d got=%0d expected=%0d", lane,
                             $signed(out_data[lane*WIDTH +: WIDTH]), base+lane);
                    $fatal(1);
                end
        end
    endtask

    initial begin
        in_data=0; in_valid=0; out_ready=1;
        repeat(3) @(posedge clk); rst_n=1;

        // 32 scalars on 32 consecutive cycles: no producer stall is allowed.
        sent=0; received=0; expected_base=0;
        while (sent < 32) begin
            @(negedge clk);
            if (!in_ready) begin
                $display("FAIL unexpected producer stall at scalar %0d", sent);
                $fatal(1);
            end
            in_valid=1; in_data=sent;
            @(posedge clk);
            sent=sent+1;
            if (out_valid && out_ready) begin
                check_word(expected_base);
                expected_base=expected_base+8;
                received=received+1;
            end
        end
        @(negedge clk); in_valid=0;
        repeat(4) begin
            @(posedge clk);
            if (out_valid && out_ready) begin
                check_word(expected_base);
                expected_base=expected_base+8;
                received=received+1;
            end
        end
        if (received != 4) begin
            $display("FAIL received %0d words expected 4", received); $fatal(1);
        end

        // Hold one completed word under backpressure and ensure it is stable.
        out_ready=0;
        for (sent=100; sent<108; sent=sent+1) begin
            @(negedge clk);
            if (!in_ready) begin $display("FAIL early stall"); $fatal(1); end
            in_valid=1; in_data=sent;
            @(posedge clk);
        end
        @(negedge clk); in_valid=0;
        if (!out_valid) begin $display("FAIL missing blocked word"); $fatal(1); end
        check_word(100);
        repeat(3) begin
            @(posedge clk);
            if (!out_valid) begin $display("FAIL valid dropped under backpressure"); $fatal(1); end
            check_word(100);
        end
        @(negedge clk); out_ready=1;
        @(posedge clk);

        $display("PASS residual_packer_1to8 continuous stream and backpressure");
        $finish;
    end
endmodule
