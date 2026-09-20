`timescale 1ns/1ps
module tb_shared_mac_array;
    localparam integer P = 8;
    localparam integer DW = 14;
    localparam integer WW = 14;
    localparam integer AW = 40;

    reg clk = 0;
    reg rst = 1;
    reg in_valid = 0;
    reg clear_acc = 0;
    reg last = 0;
    reg [P*DW-1:0] data_in = 0;
    reg [P*WW-1:0] weight_in = 0;
    wire signed [AW-1:0] acc_out;
    wire out_valid;
    wire [63:0] mac_active_cycles;
    wire [63:0] mac_count;

    integer i;

    shared_mac_array #(.P(P), .DATA_W(DW), .WEIGHT_W(WW), .ACC_W(AW)) dut (
        .clk(clk), .rst(rst), .in_valid(in_valid), .clear_acc(clear_acc),
        .last(last), .data_in(data_in), .weight_in(weight_in),
        .acc_out(acc_out), .out_valid(out_valid),
        .mac_active_cycles(mac_active_cycles), .mac_count(mac_count)
    );

    always #5 clk = ~clk;

    task load_uniform;
        input signed [DW-1:0] d;
        input signed [WW-1:0] w;
        begin
            for (i = 0; i < P; i = i + 1) begin
                data_in[i*DW +: DW] = d;
                weight_in[i*WW +: WW] = w;
            end
        end
    endtask

    initial begin
        repeat (2) @(posedge clk);
        rst <= 0;

        // First chunk: 8 * (3 * -2) = -48.
        @(negedge clk);
        load_uniform(14'sd3, -14'sd2);
        in_valid = 1; clear_acc = 1; last = 0;
        @(posedge clk); #1;
        if ($signed(acc_out) !== -40'sd48) $fatal(1, "first accumulation mismatch: %0d", $signed(acc_out));

        // Second chunk: 8 * (-4 * 5) = -160; accumulated result -208.
        @(negedge clk);
        load_uniform(-14'sd4, 14'sd5);
        clear_acc = 0; last = 1;
        @(posedge clk); #1;
        if ($signed(acc_out) !== -40'sd208) $fatal(1, "final accumulation mismatch: %0d", $signed(acc_out));
        if (!out_valid) $fatal(1, "last did not raise out_valid");
        if (mac_active_cycles !== 64'd2) $fatal(1, "active-cycle counter mismatch: %0d", mac_active_cycles);
        if (mac_count !== 64'd16) $fatal(1, "MAC counter mismatch: %0d", mac_count);

        // Bubble must preserve accumulator and counters.
        @(negedge clk);
        in_valid = 0; clear_acc = 0; last = 0;
        @(posedge clk); #1;
        if ($signed(acc_out) !== -40'sd208) $fatal(1, "bubble changed accumulator");
        if (out_valid) $fatal(1, "out_valid persisted across bubble");
        if (mac_active_cycles !== 64'd2 || mac_count !== 64'd16) $fatal(1, "bubble changed counters");

        $display("PASS shared_mac_array P=%0d acc=%0d active_cycles=%0d mac_count=%0d", P, $signed(acc_out), mac_active_cycles, mac_count);
        $finish;
    end
endmodule
