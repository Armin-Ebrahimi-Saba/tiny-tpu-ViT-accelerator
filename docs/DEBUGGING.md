> **Provenance.** This is the bring-up account from
> [Armin-Ebrahimi-Saba/RISC-V-ViT-accelerator](https://github.com/Armin-Ebrahimi-Saba/RISC-V-ViT-accelerator),
> a sibling project on the same rvlab SoC, brought here unchanged because the
> platform defects it describes are in the DDR3 path this project has not yet
> exercised. Where it says "our" it means that project. Of the three platform
> fixes it ends with, this tree's `rvlab/` submodule already carried the
> write-back fix from upstream (`rvlab` commit `ff12462`, July 2026); the
> `a_ready` fix and the prefetcher bypass switch were applied here, and the
> watchdog register it credits with finding the hang is in
> `rvlab/src/rtl/student/student_tl_watch.sv`. The distilled rules are in
> `LESSONS.md`; `report.md` §6.15 records what was taken and what was not.

# What was debugged, and how

This is the story of getting Depth-Anything V2 to run correctly on the FPGA,
told in the order it happened. It is written for someone who did not follow
the work — including someone learning hardware bring-up for the first time —
so every specialised term is explained the first time it appears.

The dead ends are kept in deliberately. Almost every wrong theory here was
plausible, was held with confidence, and was eventually disproved by a
measurement. Knowing *why* each was wrong is more useful than knowing the
answer.

The short version: **the accelerator and the model were never the problem.**
Three defects in the platform's DDR3 memory path — none of them ours — caused
every symptom, and each one hid the next.

---

## A few terms used throughout

- **SoC** — "system on chip": the CPU, memories, bus and our accelerator, all
  built into the FPGA.
- **TL-UL** — TileLink Uncached Lightweight, the bus protocol the SoC uses. A
  *master* (something that wants memory, e.g. the CPU) sends a *request* on the
  "A channel"; the *device* (e.g. memory) sends a *response* on the "D
  channel". Each channel has a `valid` signal from the sender and a `ready`
  signal from the receiver; a transfer happens only in a cycle where both are
  high. This valid/ready pair is a **handshake**.
- **`a_ready` / `d_ready`** — the "I can accept this" signal on the request
  and response channels respectively.
- **DDR3** — the 512 MB of external DRAM on the board. Slow and far away, so
  it sits behind a cache.
- **Cache** — a small fast on-chip memory (16 kB here) holding copies of
  recently-used DDR3 lines. A **line** is 32 bytes. **Direct-mapped** means
  each DDR3 address can live in exactly one cache slot (its **set**), chosen
  by some address bits; two addresses that differ only in the other bits
  compete for the same slot and are said to **alias**.
- **Write-back cache** — writes go into the cache line and are *not* sent to
  DDR3 immediately. The line is marked **dirty**. It reaches DDR3 only when
  **evicted** — pushed out to make room for another address in the same set.
  So "the cache acknowledged my write" says nothing about whether DDR3 has it.
- **JTAG** — the debug cable. It can load programs, halt the CPU, read
  registers, and read/write memory directly through **system bus access
  (sysbus)** without involving the CPU at all. That last property turns out to
  matter enormously.
- **Requantisation** — after a matrix multiply produces 32-bit results, they
  are scaled back down to 16-bit activations for the next layer.
- **Testbench** — a simulation harness that drives a hardware module with
  stimulus and checks its outputs. **RTL** is the hardware source code.

---

## 1. Early tooling failures (before any real bug)

Before the first inference could even be attempted, a run of host-side
tooling problems each looked like a hardware fault. Listed briefly because
each one cost real time and each is easy to hit again.

| What it looked like | What it was |
|---|---|
| "The 25 MB weight transfer takes 1–2 hours" | It takes 67 s. `print()` block-buffers when redirected to a file, so a finished transfer looked like a stalled one. Use `flush=True` or `python -u`. |
| The board ran the wrong program | The ELF path was relative; OpenOCD resolves it against *its own* directory, and `load_image` fails silently. Always pass absolute paths and check the returned text. |
| Program looked wedged, old output appeared | The console ring was cleared *before* `riscv set_mem_access sysbus` was issued, so the clear was silently discarded. Set the access mode first. |
| CPU never saw the "go" flag | The flag lived in DDR3. Once the debugger has written DDR3 over sysbus, the CPU stops seeing later debugger writes there (the cache holds a stale copy). Moved the flag to on-chip BRAM, where there is no cache. |
| Repeated exit code 144 | `pkill -f pattern` matched its *own* command line and killed the shell. Use `pgrep -f "[p]attern"` — the brackets stop the pattern matching itself. |
| Out of memory, 30 GB gone | Killing the simulator child (`xsimk`) left seven `flow` parents waiting forever. Kill the parent. Later, the reverse: `pkill -f "[x]simk"` left an `xsim --gui` window alive with a dead kernel behind it for 23 hours, looking hung. Kill `-f "[x]sim"` (no `k`) to get both. |

---

## 2. The console flooding that looked like a hang

**Symptom.** The program printed `[patch embedding]` and then nothing, for
minutes. Interpreted as "the CPU is slow at element-wise work" — an
estimate of 12 minutes was even given.

**Why that was wrong.** The arithmetic never supported it: roughly 110,000
element operations cannot take 36 billion cycles. Nobody divided the cycle
count by the work.

**What it was.** The program prints through a **hostio ring** — a 1 kB
circular buffer in BRAM that the host drains over JTAG at tens of bytes per
second. A program that prints faster than that fills the ring and then spins
in `obuf_putc` waiting for space. From outside, that is indistinguishable from
a hung program. A per-row diagnostic print left in from an earlier
investigation was flooding the ring.

**Fix.** Removed the print. Keep target-side printing minimal.

Note a later false conclusion this produced: after the print was removed,
patch embedding appeared to be reached "28× sooner" (117 M vs 3,359 M
cycles). That was a wrap artifact — `mcycle` is a 32-bit counter that wraps
every 85.9 s at 50 MHz. Both runs took about the same time.

---

## 3. The hang nobody could halt

This was the big one. The model **never reached `block 1/12`** in any run,
across the whole bring-up.

### The symptom, as measured

Reading the accelerator's debug registers over sysbus while the CPU ran:

    status=2  dbg=00002000 (state=IDLE, nothing outstanding)  cycles frozen
    hostio ring: widx == ridx (empty — the CPU is NOT stuck printing)

So the accelerator was idle and clean, the console was empty, and the CPU was
doing nothing observable.

### The theories that were wrong

1. **"It's a software infinite loop."** The same C runs on the host in 4
   seconds, end to end. The loop terminates.
2. **"It's stuck printing."** The ring was empty. (Earlier, a ring that
   *looked* jammed — `widx=18, ridx=19` — was created by my own attach:
   `openocd.start` resets the core, so the freshly restarted program wrote 18
   bytes of banner against a stale read index. Every PC reading taken by
   killing the runner to attach was a reading of a restarted program.)
3. **"The debugger can find the PC."** It could not, ever, and this was the
   structural fact that took longest to accept. Driving the halt request
   directly into the RISC-V debug module (bypassing OpenOCD's `halt`) returned
   `allhalted=0, anyrunning=1, anyunavail=0` — the debug module was talking to
   the CPU fine; the CPU simply could not stop. **A CPU that has issued a bus
   request that never gets answered cannot retire that instruction, and a CPU
   that cannot retire cannot enter debug mode.** There was no PC to read.
4. **"The cache drops responses (the `d_ready` defect, section 6)."** A real
   defect, and a skid buffer was built to work around it. It did not fix
   this hang. Worse, it *amplified* the real bug — see below.

### The instrument that found it

If the debugger cannot ask the CPU, put the answer in hardware where JTAG
can still read it. `student_tl_watch.sv` is a dozen flip-flops that snoop
the DDR3 bus port and latch the oldest request that has not been answered:
its address, whether it was a read or write, which master sent it, how many
are outstanding, and how long it has waited (a **saturating** counter, so a
value pinned at maximum is unambiguous). Exposed as two registers in the DDR3
control block, readable over sysbus on a wedged system. It drives nothing on
the bus, so it cannot disturb what it watches.

One line:

    ddr watchdog: addr=82045670 PutFullData source=71 outstanding=30
                  stalled=65535 cycles (SATURATED -- never answered)

A *write*, from the accelerator's write engine, to the activation arena, with
**30 transactions outstanding** on a port that should never hold more than
two. Requests were leaving and nothing was coming back.

### The cause

`rvlab_tlul_ddr.sv` — the module that wraps the DDR3 cache — built its bus
response starting from the *error responder* (a small block that answers
requests with an error while DDR3 is not yet calibrated) and overrode it when
the cache had a response:

    tl_o = err_resp_rsp;                  // a_ready comes from HERE
    if (cache_rsp.d_valid) tl_o = cache_rsp;

The error responder holds `a_ready` high whenever it is idle. Once
calibration completes, it is told never to accept anything — but its
`a_ready` was still what the bus saw. So whenever the cache was *not* ready,
the bus still saw "ready", handshaked the request away, and **nobody accepted
it**. No response could ever exist for it. The CPU wedged on the next such
request forever.

The skid buffer from section 6 made this systematic instead of intermittent:
its throttle deasserts the cache-side `a_ready` routinely, and every request
issued in those windows was swallowed.

### The fix, and its verification

Source `a_ready` from whichever module will actually accept:

    tl_o.a_ready = ctrl_calib_complete ? cache_rsp.a_ready : err_resp_rsp.a_ready;

Verified on hardware: the run went from never reaching block 1 to passing
patch embedding and all twelve transformer blocks.

> **Lesson.** When a response never arrives, check first whether the request
> was ever *accepted*. A mux that picks a response source must pick the
> matching `a_ready`, or the two halves of the handshake describe different
> modules.

---

## 4. Two mistakes I made while chasing section 3

**A "read-only" probe that killed the run.** To peek at live registers I
wrote a script using `with OpenOcd() as ocd:`. That context manager's
`__exit__` sends `shutdown` to OpenOCD. It terminated the debug server the
live run depended on, and the runner sat blocked on a dead socket — which I
then reported as "still waiting". When attaching to a session you must not
disturb, connect and close the socket by hand.

**A start condition that fired too early.** To let a simulation start
without a host writing the go flag, I let `main.c` also start when the weight
blob's header looked valid. The header is the *first* thing transferred, so
inference began against a blob still being written underneath it, and DDR3
read-back failed seconds later — reproducibly, and briefly mistaken for a
memory fault. Reverted. Only the flag means the transfer is complete.

**Adding a register shifted every register after it.** The watchdog
registers went into `ddr_ctrl.hjson` ahead of the existing `ctrl` register,
moving it from `+0x4` to `+0xc`. Software built against the old map wrote the
DDR3 reset bit into a read-only address, and DDR3 never came out of reset.
After any `.hjson` change, rebuild `libsys` *and* the program.

---

## 5. Tensors that existed but could not be found

With the hang gone, the run reached the transformer blocks and started
saying:

    dav2: tensor 'blk1.qkv.w' not found in blob
    dav2: 'blk6.fc1' has k=2143289344, expected 384

That second number is `0x7FC00000` — a floating-point NaN bit pattern read as
an integer. These were reads returning garbage.

### Ruling things out

- The tensors are in the file (checked by parsing the blob directory).
- A device-side checksum of all 24,871,428 bytes, read through the CPU's own
  path with the accelerator idle, matched the file. **DDR3 held the right
  bytes.**
- Making the lookup retry once: every miss failed *twice*. So not a transient
  glitch — the CPU was reading consistently wrong data under load.

### The cause

The 21 kB blob directory is scanned linearly on every lookup, through a 16 kB
cache, while the accelerator writes the activation arena. The arena at
`0x82000000` and the blob at `0x80000000` differ only in their upper address
bits, so **every set aliases**. Under that pressure, `rvlab_ddr_prefetch` —
a block between the cache and DDR3 that fetches lines ahead of time — was
returning the *alias partner's* line, another set's line, or zeros.

### Proof, and fix

`rvlab_ddr_alias_tb.sv` drives that exact pattern against the real cache and
prefetcher with a behavioural memory behind them (seconds, not hours). With
the prefetcher in the path: 65 of 256 reads wrong, e.g.
`got 820100a5 want 800100a5 <-- ARENA'S DATA`. With it bypassed: 256/256
correct. So the cache was fine and the prefetcher was not.

Fix: `USE_PREFETCH = 1'b0`. This is a **bypass, not a repair** — it costs
read bandwidth, and the prefetcher's logic looked correct on inspection, so the
fault was never localised inside it. On hardware, lookup failures went from 61
per run to 0.

---

## 6. The `d_ready` defect and the skid buffer

Discovered early, worked around, and worth recording because it is the defect
the accelerator's retry logic exists for.

`rvlab_ddr_block_cache` asserts its response `d_valid` for exactly one cycle
and **never looks at `d_ready`**. TL-UL requires a device to hold its response
until the receiver is ready. So if the bus is busy on that one cycle, the
response is gone. The CPU never exposes this — it issues one request at a time
and is always ready — but the accelerator, the SoC's first master with
multiple requests in flight, lost exactly one response per dirty eviction.
Measured: `accepted=444, responses=443`.

Proven by `rvlab_ddr_dready_tb.sv`, which holds `d_ready` low for 200 cycles
and then raises it: response lost.

Two workarounds, both in our own files:

- `student_gemm` remembers the read in flight and re-issues it if no response
  arrives within 2048 cycles. (Reads only — a lost *write* ack would still
  wedge the block; noted by review, not yet fixed.)
- `student_tl_rsp_hold`, a **skid buffer**: a tiny FIFO in front of the cache
  that always tells the cache "ready", takes every response the cycle it is
  offered, and holds it until the real receiver accepts. A credit counter
  throttles requests so the FIFO cannot overflow.

The skid buffer is correct — but see section 3 for how it amplified a
different defect until that one was fixed. A correct fix to one layer can make
a bug in another layer worse.

---

## 7. The output was wrong: one lost word

With lookups clean, the full inference completed and produced a depth map.
It was wrong: correlation **r = 0.19** against the host build, which itself
matches PyTorch at r = 0.9998. Same C source, different numbers.

I had described that map as "physically sensible" from its statistics before
looking at it. It was three circular blobs on black. Look at the picture.

### Narrowing by measurement

`dav2_accel_bigcheck` compares the accelerator against the CPU kernel at the
exact shapes the model uses, with every buffer in DDR3:

    82x384x384    0/31488 words differ
    82x384x1152   0/94464
    82x384x1536   0/125952
    82x1536x384   0/31488
    81x588x384    1/31104 words differ, first at 18904 (hw 0 sw -4690841)

Patch embedding lost **exactly one word**, deterministically — same index,
same values, six runs. That one word plausibly ruins everything: the
requantisation scale is derived from the largest magnitude in the result, so a
missing large value rescales the whole tensor, and patch embedding feeds all
twelve blocks.

A shape sweep showed **N is the trigger, not K**: N = 49, 65, 79, 81, 97 fail;
17, 33, 80, 82, 83 pass. The same failing shape **passes** in `student_gemm_tb`
against ideal memory, so the accelerator's logic was sound — the fault needed
the real cache's back-pressure.

Three more measurements, each ruling out a theory:

1. Evict the cache line and re-read: still 0. **Lost, not read stale.**
2. Decode every lost word's position: always a tile's *final* row. It is the
   write issued immediately before the next tile's first access evicts it.
3. Per-job counters in the accelerator's `dbg2` register, `{writes issued,
   writes acked}`: on the failing job, `issued == acked == nt × M`. The
   accelerator put the write on the bus and the bus acknowledged it.

That third one **exonerated the accelerator** — the theory that it was our
drain logic was wrong — and moved the fault into the cache.

### A test with no power

The aliasing testbench passed on the unfixed RTL. I then "fixed" the cache by
gating its data and dirty-bit lookups on the stall signal, and the test still
passed. Fix confirmed? No: it had passed *before* the fix too. A test that
passes with and without the change proves nothing.

The reason: the testbench's driver waits for each response before sending the
next request, so a stalled miss never has a second request queued behind it.
That is exactly the condition the defect needs. A CPU does not wait — it
presents the next request the moment the previous one is accepted.

Rewriting the driver to do the same (**pipelined**, not serialised) made the
test fail on unfixed RTL at a single word, `80004340` — whose write was
immediately followed by a write to the same set. Now the test had power.

### The cause

`rvlab_ddr_block_cache.sv` issued write-backs of dirty lines with:

    a_data: data_rdata_raw,     // the raw RAM output

`data_rdata_raw` is one cycle stale when the line being evicted was written
on the *immediately preceding* access. A **write-first forwarded** version,
`data_rdata` — which returns the just-written value instead of the RAM's old
contents — already existed and was already used for the front-end response.
It had never been applied to the back-end write-back. Two back-to-back misses
to the same set — write a line, then evict it — sent the *pre-write* contents
to DDR3.

That explains every observation: the lost word is always a tile's final write
(the one evicted next); `issued == acked` with the data gone (the cache acks,
then writes back the wrong bytes); and it needs real back-pressure to appear.

### Fix and result

One line: `a_data: data_rdata`. With the pipelined driver: 0 of 1024 wrong.
On hardware: every GEMM shape bit-exact, and the full inference output
**identical to the host build in all 15,876 pixels** — r = 1.000000 against
the host, r = 0.999836 against PyTorch.

This is a fix in platform RTL (RVLab code, not a vendored third-party
library), and worth upstreaming.

---

## 8. A finding I got wrong: "DDR3 loses 111 words"

Between sections 7's narrowing and its fix, you asked me to test the memory
directly. I wrote a random write-then-read test over the 64 MB arena. It
reported **111 of 65,536 words wrong**, identical across runs, every wrong
value being *another valid word*. Two controls — random access confined to
8 kB (no evictions) and sequential access — both passed. I committed this as
"only out-of-order eviction loses data" and built a theory on it.

**It was the checker.** The random sequence picks some addresses more than
once. The naive check compared each operation against the value *it* wrote,
not the *last* value written to that address. The number of repeated
addresses in the sequence is exactly 111. The "wrong" values were correct —
they were the later write. The controls isolated nothing: sequential access
never repeats an address, and the in-cache test compared immediately after
each write.

With a bitmap marking repeated addresses, DDR3 passes **0/65536** on every
pattern. The memory was never at fault. Retracted in the commit that fixed
section 7.

The simulation testbench had the last-writer check from the start; the
hardware test did not. Your instinct to test the memory was right — it just
needed a correct checker.

---

## 9. Tests that passed while testing nothing

Three times in this project a test reported success without exercising the
thing it existed to test. Each is the same mistake.

- **`STRICT_SERIAL`.** A change to the accelerator was declared
  "regression-clean" and shipped to hardware, where it deadlocked the block at
  boot. The testbench instantiated the module *without* the new parameter, so
  it defaulted off and the test ran the old path.
- **`student_gemm_ddrpath_tb`.** Defined `run_gemm` and `run_job` and called
  neither. Printed `PASSED (0 words checked)`. Caught by external review.
- **The stall-gating "fix"** in section 7, which passed a test that also
  passed unfixed.

The rule that would have caught all three: **make the test fail before you
make it pass.** Run it on the broken version first. If it passes there, it is
not testing what you think.

---

## 10. Current state

**Working, and verified on hardware.**

- Full Depth-Anything V2 Small inference runs on the FPGA at 126×126.
- Output is bit-exact with the host build (15,876/15,876 pixels), and matches
  the PyTorch reference at r = 0.9998 — the same as the host does.
- Four additional test images (gradient, checkerboard, shaded sphere,
  corridor), each bit-exact against the host.
- Frame time **93.6 s** (4.68 G cycles at 50 MHz), i.e. **0.0107 FPS**.
- Timing met: WNS +0.287 ns, no failing endpoints. All syn/pnr/bitstream
  reports checked; remaining warnings are advisories (async-reset flops
  blocking DSP register merging, DSP pipelining headroom not needed at 50 MHz).

**Three real defects fixed, none in the accelerator or the model:**

| Defect | Where | Fix |
|---|---|---|
| `a_ready` from the idle error responder; requests accepted by nobody | `rvlab_tlul_ddr.sv` | source `a_ready` from the module that will accept |
| write-back uses stale RAM output for a line written the previous cycle | `rvlab_ddr_block_cache.sv` | `a_data: data_rdata` |
| prefetcher returns aliased lines | `rvlab_ddr_prefetch.sv` | **bypassed**, not repaired |

Plus the cache's `d_ready` defect, **worked around** by the skid buffer and
the accelerator's retry rather than fixed at source.

**Not done.**

- The prefetcher is bypassed; read bandwidth is lower than it could be.
- `student_gemm`'s retry covers reads only; a lost write ack would still
  wedge it. Unlikely now that requests are never swallowed, but real.
- ~~`main.c` reports the frame time from two 32-bit `mcycle` reads~~ Fixed:
  it reads `mcycleh` too and prints seconds and FPS directly.
- ~~`sim_ddrmodel_xsim` needs a start condition~~ Fixed with a two-word token
  at `0x81F00000` that `ddr3_blk_model` places under `+dav2_autostart`. Each
  half is verified; the one full run was closed before they met, so a
  complete run is still owed.
- ~~`student_gemm_ddrpath_tb` fails wholesale~~ Fixed: the reference arrays
  were sized for M ≤ 8 and the shapes I added overflowed them, so the weights
  sent to memory were X. Now sized for the shapes run and guarded. All four
  shapes pass against the real cache.
- No profiling of the 93.6 s. The accelerator is ~36× faster than the CPU
  kernel on GEMM, so the rest is CPU-side float bookkeeping and console I/O.

**Instruments that made the difference**, most leverage first:

1. A hardware watchdog register readable over JTAG on a wedged core.
2. Accelerator-vs-CPU comparison at model shapes, in DDR3.
3. A testbench driver that pipelines requests the way a CPU does.
4. The negative control — every time.

---

## 11. Separating the image from the weights

Not a bug, but the last change to how the system is used, and it closed a
real limitation.

**The problem.** The exporter baked the input picture into the weight blob
as two tensors, so every new image cost the full 25 MB, ~66 s JTAG transfer.
Eight images meant eight weight loads. It also meant that without `torch` on
the machine — which there was not — no new picture could be made at all; the
first workaround was a script that rewrote the two image tensors inside a
blob copy from synthetic scenes drawn in pure Python.

**The change.** The image is now a separate 95 kB buffer at `0x81E00000`, in
the 8 MB gap between blob and arena. `dav2_engine.c` takes it through a
setter instead of a blob lookup. `main.c` prints `DAV2_READY` after the
weights are accepted and then loops: set the flag to *done*, wait for the
host to set it to *frame*, infer, print, repeat. The runner loads weights
once and then sends each image (~0.26 s) and collects each result. The host
oracle takes the same `.dav2img` file as a separate argument, so it sees the
identical bytes.

**Two things checked before trusting it.**

- The new preprocessing (`dav2_image.py`, PIL + numpy) had to match the
  exporter *exactly*, or every board result would disagree with PyTorch for
  reasons unrelated to the hardware. Rebuilding the demo picture from the
  exporter's saved RGB and comparing against the tensor still inside the old
  blob: 47628 of 47628 pixels equal and the same float32 scale, bit for bit.
- An old comment in `main.c` warned that the CPU had once kept seeing a stale
  DDR3 value after a debugger write. That observation predated the bus fixes,
  and both paths go through the same cache, so I ran it rather than adding
  an invalidate register pre-emptively. Fifteen frames later, all bit-exact,
  it has not recurred — it was almost certainly one of the swallowed-write
  bugs wearing a different hat.

**Result.** Three frames in one session: weights 66 s once, then 0.26 s +
93.7 s per frame, each 15876/15876 against the oracle. Then six photographs
from the Depth-Anything repository through one load, likewise. The per-image
cost fell from ~160 s to ~94 s, all of it inference.

**Input options beyond JTAG**, for whoever wants a camera. Ranked by effort:

1. *PC webcam through the existing loop* — no RTL. `dav2_image.from_file`
   accepts anything PIL opens; an OpenCV frame is one conversion away.
2. *UART* — the FTDI UART pins are in the XDC but the SoC has no UART block.
   Add one plus a driver and the board runs standalone after the one-time
   weight load; 95 kB at 3 Mbaud is under a second.
3. *OV7670 on a Pmod* — ~€5, moderate RTL, the classic student project,
   writing pixels to DDR3 through the existing `student_dma`.
4. *HDMI in* — pins exist, but TMDS decode plus 1080p→126 downscale is a
   large job. USB webcam is not viable: the board's USB is HID-only.

Whichever path: inference is ~94 s a frame, so "camera" means one depth map
per minute and a half. The input plumbing is not the bottleneck.
