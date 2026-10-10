// ABOUTME: Drives rvlab's tinytpu_wdma against a TL-UL memory with random response delays.
// ABOUTME: Checks copy, im2col gather and strided write-back against straightforward reference loops.

module tb_tinytpu_dma;
  localparam int WB = 128;
  localparam int AW = 16;
  localparam int MEM_BYTES = 1 << 16;
  localparam int BUFW = 1024;

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;

  // ------------------------------------------------------------ descriptor
  logic start = 1'b0;
  logic [1:0]  mode;
  logic [31:0] src;
  logic [AW-1:0] dst_word;
  logic [1:0]  dst_region;
  logic [16:0] len;
  logic [15:0] rows, g_h, g_w, g_cw, g_wo, g_oy0, g_ox0, wb_cols, wb_grp;
  logic [7:0]  g_k, g_stride;
  logic [3:0]  g_pad;
  logic [5:0]  g_t0, g_t1;
  logic [31:0] wb_stride, wb_gstride;
  logic busy, done, err;

  logic          wr_en;
  logic [1:0]    wr_region;
  logic [AW-1:0] wr_addr;
  logic [WB-1:0] wr_data;
  logic [AW-1:0] rd_addr;
  logic [WB-1:0] rd_data;

  tlul_pkg::tl_h2d_t tl_h2d;
  tlul_pkg::tl_d2h_t tl_d2h;

  tinytpu_wdma #(.WORD_BITS(WB), .DST_AW(AW)) dut (
    .clk_i(clk), .rst_ni(rst_n),
    .start_i(start), .mode_i(mode), .src_addr_i(src), .dst_word_i(dst_word),
    .dst_region_i(dst_region), .len_i(len), .rows_i(rows),
    .g_h_i(g_h), .g_w_i(g_w), .g_cw_i(g_cw), .g_wo_i(g_wo),
    .g_k_i(g_k), .g_stride_i(g_stride), .g_pad_i(g_pad), .g_t0_i(g_t0), .g_t1_i(g_t1),
    .g_oy0_i(g_oy0), .g_ox0_i(g_ox0),
    .wb_cols_i(wb_cols), .wb_stride_i(wb_stride), .wb_grp_stride_i(wb_gstride), .wb_grp_i(wb_grp),
    .busy_o(busy), .done_o(done), .err_o(err),
    .wr_en_o(wr_en), .wr_region_o(wr_region), .wr_addr_o(wr_addr), .wr_data_o(wr_data),
    .rd_addr_o(rd_addr), .rd_data_i(rd_data),
    .tl_o(tl_h2d), .tl_i(tl_d2h)
  );

  // --------------------------------------------- buffers seen by the DMA
  logic [WB-1:0] buf_mem [BUFW];          // whichever region it writes
  logic [WB-1:0] out_mem [BUFW];          // the result buffer it reads
  logic [1:0]    wr_region_seen;
  always_ff @(posedge clk) begin
    if (wr_en) begin
      buf_mem[$clog2(BUFW)'(wr_addr)] <= wr_data;
      wr_region_seen   <= wr_region;
    end
    rd_data <= out_mem[$clog2(BUFW)'(rd_addr)];
  end

  // ---------------------------------------------------- TL-UL memory model
  // Always ready; answers each request after 1..4 cycles, one at a time,
  // which is all the adapter ever has in flight.
  logic [7:0] mem [MEM_BYTES];
  logic       pend = 1'b0;
  int         wait_n;
  logic [31:0] p_addr, p_data;
  logic        p_we;
  int unsigned rng = 32'h1234_5678;
  function automatic int unsigned rnd();
    rng = rng * 1103515245 + 12345;
    return rng >> 16;
  endfunction

  assign tl_d2h.a_ready  = !pend;
  assign tl_d2h.d_valid  = pend && (wait_n == 0);
  assign tl_d2h.d_opcode = p_we ? tlul_pkg::AccessAck : tlul_pkg::AccessAckData;
  assign tl_d2h.d_data   = p_data;
  assign tl_d2h.d_error  = 1'b0;
  assign tl_d2h.d_param  = '0;
  assign tl_d2h.d_size   = 2'd2;
  assign tl_d2h.d_source = '0;
  assign tl_d2h.d_sink   = '0;
  assign tl_d2h.d_user   = '0;

  int writes_seen = 0;
  always_ff @(posedge clk) begin
    if (!pend && tl_h2d.a_valid) begin
      pend   <= 1'b1;
      wait_n <= int'(rnd() % 4);
      p_addr <= tl_h2d.a_address;
      p_we   <= (tl_h2d.a_opcode == tlul_pkg::PutFullData);
      if (tl_h2d.a_opcode == tlul_pkg::PutFullData) begin
        for (int b = 0; b < 4; b++) mem[(tl_h2d.a_address + b) % MEM_BYTES] <= tl_h2d.a_data[8*b +: 8];
        writes_seen <= writes_seen + 1;
      end else begin
        for (int b = 0; b < 4; b++) p_data[8*b +: 8] <= mem[(tl_h2d.a_address + b) % MEM_BYTES];
      end
    end else if (pend) begin
      if (wait_n == 0) pend <= 1'b0;
      else             wait_n <= wait_n - 1;
    end
  end

  // ------------------------------------------------------------- helpers
  int errors = 0;

  task automatic go();
    @(negedge clk); start = 1'b1;
    @(negedge clk); start = 1'b0;
    fork
      begin : wait_done
        while (!done) @(posedge clk);
      end
      begin : watchdog
        repeat (400000) @(posedge clk);
        $display("FAIL: DMA did not finish (mode %0d)", mode);
        $finish;
      end
    join_any
    disable fork;
    @(negedge clk);
    if (busy) begin $display("FAIL: busy after done"); errors++; end
  endtask

  task automatic clear_desc();
    mode = 0; src = 0; dst_word = 0; dst_region = 0; len = 0; rows = 0;
    g_h = 0; g_w = 0; g_cw = 0; g_wo = 0; g_oy0 = 0; g_ox0 = 0; g_k = 0; g_stride = 0;
    g_pad = 0; g_t0 = 0; g_t1 = 0; wb_cols = 0; wb_grp = 0; wb_stride = 0; wb_gstride = 0;
  endtask

  function automatic logic [WB-1:0] mem_word(int addr);
    logic [WB-1:0] w;
    for (int b = 0; b < WB / 8; b++) w[8*b +: 8] = mem[(addr + b) % MEM_BYTES];
    return w;
  endfunction

  // im2col reference: the same rows sw/export_head.py's im2col() builds.
  task automatic check_gather(int base, int H, int W, int cw, int k, int s, int pad,
                              int t0, int t1, int Wo, int oy0, int ox0, int nrows, int dst0);
    int oy = oy0, ox = ox0, word = dst0;
    for (int r = 0; r < nrows; r++) begin
      for (int t = t0; t < t1; t++) begin
        int iy = oy * s + t / k - pad;
        int ix = ox * s + t % k - pad;
        for (int c = 0; c < cw; c++) begin
          logic [WB-1:0] want;
          if (iy < 0 || ix < 0 || iy >= H || ix >= W) want = '0;
          else want = mem_word(base + ((iy * W + ix) * cw + c) * 16);
          if (buf_mem[word] !== want) begin
            if (errors < 5)
              $display("FAIL gather row %0d tap %0d word %0d: got %h want %h", r, t, c, buf_mem[word], want);
            errors++;
          end
          word++;
        end
      end
      ox++;
      if (ox == Wo) begin ox = 0; oy++; end
    end
  endtask

  task automatic run_gather(int H, int W, int cw, int k, int s, int pad, int t0, int t1,
                            int oy0, int ox0, int nrows, int dst0);
    int Wo = (W + 2 * pad - k) / s + 1;
    clear_desc();
    for (int i = 0; i < BUFW; i++) buf_mem[i] = {4{32'hDEAD_BEEF}};
    mode = 2'd1; src = 32'h0000_2000; dst_word = AW'(dst0); dst_region = 2'd0;
    g_h = 16'(H); g_w = 16'(W); g_cw = 16'(cw); g_wo = 16'(Wo);
    g_k = 8'(k); g_stride = 8'(s); g_pad = 4'(pad); g_t0 = 6'(t0); g_t1 = 6'(t1);
    g_oy0 = 16'(oy0); g_ox0 = 16'(ox0); rows = 16'(nrows);
    go();
    check_gather(32'h2000, H, W, cw, k, s, pad, t0, t1, Wo, oy0, ox0, nrows, dst0);
    // nothing past the last row was touched
    if (buf_mem[dst0 + nrows * (t1 - t0) * cw] !== {4{32'hDEAD_BEEF}}) begin
      $display("FAIL gather wrote past its rows"); errors++;
    end
  endtask

  initial begin
    for (int i = 0; i < MEM_BYTES; i++) mem[i] = 8'(rnd());
    for (int i = 0; i < BUFW; i++) out_mem[i] = {16'(rnd()), 16'(rnd()), 16'(rnd()), 16'(rnd()), 16'(rnd()), 16'(rnd()), 16'(rnd()), 16'(rnd())};
    clear_desc();
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    // -- copy: the original weight DMA, unchanged
    mode = 2'd0; src = 32'h0000_0100; dst_word = 16'd7; dst_region = 2'd1; len = 17'd5;
    go();
    for (int i = 0; i < 5; i++)
      if (buf_mem[7 + i] !== mem_word(32'h100 + 16 * i)) begin
        $display("FAIL copy word %0d", i); errors++;
      end
    if (wr_region_seen !== 2'd1) begin $display("FAIL copy region"); errors++; end

    // -- gather: a 3x3 stride-2 conv, a tap range, rows that wrap a line
    run_gather(7, 6, 2, 3, 2, 1, 2, 8, 0, 1, 10, 3);
    // -- every tap, stride 1, one word per pixel, from the middle of the grid
    run_gather(5, 9, 1, 3, 1, 1, 0, 9, 2, 7, 12, 0);
    // -- a 1x1 "conv": rows are pixels, nothing padded
    run_gather(4, 4, 3, 1, 1, 0, 0, 1, 0, 0, 16, 0);
    // -- the head's deep conv split: taps 5..8 of a 384-channel 3x3
    run_gather(5, 5, 24, 3, 2, 1, 5, 9, 1, 0, 3, 0);
    // -- a degenerate start finishes without moving anything
    clear_desc(); mode = 2'd1; rows = 16'd0; go();
    clear_desc(); mode = 2'd2; rows = 16'd0; go();

    // -- write-back: 7 rows of 2 words, row stride 96 B, groups of 3 rows
    //    1000 B apart -- the transposed conv's two-level copy
    begin
      logic [7:0] mem_before [MEM_BYTES];
      int base = 32'h8000, nrow = 7, ncol = 2, rs = 96, grp = 3, gs = 1008, w0 = 40;
      int touched [int];
      mem_before = mem;
      clear_desc();
      mode = 2'd2; src = 32'(base); dst_word = AW'(w0); rows = 16'(nrow); wb_cols = 16'(ncol);
      wb_stride = 32'(rs); wb_grp = 16'(grp); wb_gstride = 32'(gs);
      go();
      for (int r = 0; r < nrow; r++)
        for (int c = 0; c < ncol; c++) begin
          int at = base + (r / grp) * gs + (r % grp) * rs + 16 * c;
          logic [WB-1:0] want = out_mem[w0 + r * ncol + c];
          if (mem_word(at) !== want) begin
            if (errors < 5) $display("FAIL wb row %0d col %0d: got %h want %h", r, c, mem_word(at), want);
            errors++;
          end
          for (int b = 0; b < 16; b++) touched[at + b] = 1;
        end
      for (int a = 0; a < MEM_BYTES; a++)
        if (touched.exists(a) == 0 && mem[a] !== mem_before[a]) begin
          if (errors < 5) $display("FAIL wb wrote outside its rows at %h", a);
          errors++;
        end
      if (writes_seen != nrow * ncol * 4) begin
        $display("FAIL wb issued %0d writes, expected %0d", writes_seen, nrow * ncol * 4); errors++;
      end
    end

    // -- and a plain tile: grp 0 means one group
    begin
      int base = 32'hA000;
      clear_desc();
      mode = 2'd2; src = 32'(base); dst_word = 16'd0; rows = 16'd4; wb_cols = 16'd3;
      wb_stride = 32'd64; wb_grp = 16'd0; wb_gstride = 32'd0;
      go();
      for (int r = 0; r < 4; r++)
        for (int c = 0; c < 3; c++)
          if (mem_word(base + r * 64 + 16 * c) !== out_mem[r * 3 + c]) begin
            $display("FAIL wb tile row %0d col %0d", r, c); errors++;
          end
    end

    if (errors == 0) $display("tb_tinytpu_dma: PASS (copy, 4 gathers, 2 write-backs, degenerate starts)");
    else             $display("tb_tinytpu_dma: FAIL, %0d errors", errors);
    $finish;
  end
endmodule
