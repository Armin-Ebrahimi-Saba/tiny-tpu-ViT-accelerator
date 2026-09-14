# Lessons for the next project

Portable rules from bringing a custom accelerator up on a RISC-V SoC with
external DRAM. Each is *symptom → cause → rule*. Everything specific to this
codebase has been cut; the narrative with the specifics is in `DEBUGGING.md`.

Terms: a **master** issues bus requests, a **device** answers them; each
channel handshakes with a `valid` from the sender and a `ready` from the
receiver. A **write-back cache** holds writes until it must **evict** the line.
**Sysbus** is debugger access to memory that bypasses the CPU.

---

## Finding bugs

**The debugger reported "running" and every register read came back empty.**
A CPU wedged on a bus request that will never be answered cannot retire the
instruction, and a CPU that cannot retire cannot enter debug mode. There is no
PC to read and there never will be.
→ *When the debugger structurally cannot answer, put the answer in hardware.*
A dozen flops snooping the bus port, latching the oldest unanswered request,
readable over sysbus, found in one line what days of software probing could
not — and an observer that drives nothing cannot disturb the bug it watches.

**A response never arrived.** The request was never accepted. A mux chose its
response from one module and its `a_ready` from another; the bus handshaked
the request away and nobody took it.
→ *Before assuming a response was lost, check the request was accepted.* The
two halves of a handshake must describe the same module.

**"The CPU is slow at this loop" — estimated at 12 minutes.** The CPU was
hung. 110,000 operations cannot take 36 billion cycles.
→ *Divide the cycle count by the work before believing any "slow" theory.*
And: silence is not slowness. Instrument.

**The program went quiet after printing.** Its console is a 1 kB ring drained
over the debug link at tens of bytes per second; anything printing faster
spins waiting for space and looks hung.
→ *Target-side printing is a hang generator.* Keep it minimal, and never
diagnose a hang by adding more of it.

**Every PC reading showed a freshly started program.** Attaching the debugger
reset the core. Every "stalled" reading was of a restart.
→ *Know what your attach does to the target before trusting a reading.*

**A "read-only" probe killed the run it was observing.** The context manager
sent `shutdown` on exit.
→ *When attaching to a session you must not disturb, open and close the socket
by hand.* Read the teardown of any helper you borrow.

## Tests

**A fix was declared regression-clean and deadlocked the hardware at boot.**
The testbench never enabled the new parameter; the test ran the old path.
**A testbench printed `PASSED (0 words checked)`.** It defined its stimulus
tasks and never called them.
**A candidate fix passed the test.** So did the unfixed code.
→ *Make the test fail before you make it pass.* Run it on the broken version
first. A test that passes with and without the change measures nothing, and a
green run from it manufactures confidence — worse than no test.

**The bug needed a second request queued behind a stalled one, and the
testbench never produced that.** The driver waited for each response before
sending the next request. A CPU does not.
→ *Drive testbenches pipelined, the way real masters do.* A serialised driver
cannot reach the timing windows where most bus bugs live.

**A memory test reported exactly 111 bad words, stable across runs, each
"wrong" value a valid word.** The random sequence repeated 111 addresses and
the checker compared against the earlier write.
→ *A memory test that ignores repeated addresses measures its own generator.*
Compare against the last write to each address. A number that is *exactly*
stable across runs is suspicious — real faults jitter.

**The wrong value was described as "physically sensible" from its
statistics.** It was three blobs on black.
→ *Look at the picture.*

**A small self-test passed and the full run was wrong.** The self-test used
one shape in on-chip RAM; the failure needed a specific shape in DRAM.
→ *Test at the real shapes, in the real memory.* Then vary one dimension at a
time; the failing set (N = 49, 65, 79, 81, 97) named the trigger.

## Buses and caches

**A device pulsed `valid` for one cycle and ignored `ready`.** Responses
vanished whenever the fabric was busy that cycle. The CPU never saw it: it is
single-outstanding and always ready.
→ *Grep the device for `d_ready` before building a pipelined master.* "It
works for the CPU" proves nothing about a block with several requests in
flight. A skid buffer in front of the device fixes it without touching the
device.

**The cache acknowledged the write. DRAM never got it.** Write-back caches
ack on arrival; the data reaches DRAM only on eviction.
→ *An ack from a cache proves nothing about memory.* Per-request counters
showing `issued == acked` exonerate the master, not the path below it.

**A write-back sent the line's contents from one cycle earlier.** The RAM
output is stale when the line was written on the immediately preceding cycle;
the write-first forwarding path existed for the read side and was never wired
to the eviction side.
→ *Every reader of a RAM that can be written the same cycle needs the same
forwarding.* Check the back-end path, not just the front.

**Two regions differing only in upper address bits collide in every cache
set.** A prefetcher returned the alias partner's line under that pressure.
→ *Know your address map's aliasing before placing large buffers.* And test
aliasing on purpose.

**A correct fix in one layer made a defect in another layer systematic.** The
skid buffer's throttle deasserted `a_ready` routinely; the broken mux swallowed
every request in those windows.
→ *Re-measure after every fix.* Improvement is not monotone.

**A stall watchdog keyed on bus idleness never fired.** The device was
livelocked — busy, making no progress.
→ *Key watchdogs on "no response reached the consumer", and count activity in
the window to tell livelock from deadlock.*

**Address drift during a stall was a real bug in our master.** Fixing it
changed nothing on the board.
→ *Fixing a real bug does not mean you fixed the bug.* Confirm the symptom
moved.

## RTL and tooling details

**A comparison never fired.** `cnt != SW'(8)` with `SW = 3` truncated 8 to 0.
→ Counters that must represent the full count need one more bit than the
index.

**1708 assertion failures on don't-care fields.** Bus FIFOs check the whole
struct for X, including fields a read never uses.
→ Drive every field, always.

**An index went stale on the last iteration of a partial tile.** Cleared on
one FSM path, not the other.
→ *Partial and last-iteration cases are where indices go stale.* Test a
partial *first* tile, not only a partial last one.

**A register block's back-pressure reached the CPU's instruction fetch
combinationally.** WNS went negative by 77 ps.
→ A socket added for address decode is also added to the critical path.
Register its host side by default.

**Adding a register shifted every register after it; stale software wrote a
control bit into a read-only address.**
→ After any register-map change, rebuild every consumer of the map.

**Two plusargs arrived as one.** The flow joined them into a single
`--testplusarg`; the first absorbed the name of the second.
→ Print the exact simulator command line once and read it.

**A whole subsystem silently compiled out.** Its `ifdef` guard was not
defined for the "fast" simulation target, which then ran a stub.
→ *Make every configuration announce itself* (`initial $display`) so a wrong
build is visible in the first line of the log.

**A 32-bit cycle counter wrapped every 86 s, and a drop in the reading looked
like a 28× speedup.**
→ Difference timestamps with wrap correction; never trust a single subtraction
across a long interval.

**`pkill -f pattern` killed the shell running it.** The pattern matched its
own command line. And killing the simulator child left the parent holding
30 GB; killing the parent left the GUI window alive with a dead kernel.
→ Use `pgrep -f "[p]attern"`; kill the top-level process; know the process
tree of your tools.

**Interactive helpers died under redirection; `print()` block-buffered for
50 minutes; a typo'd method name guarded by `hasattr` polled forever.**
→ Write non-interactive variants; `flush=True` or `python -u`; never guard a
call you believe exists.

**A 25 MB transfer with no progress output was indistinguishable from a
hang.**
→ Chunk long transfers and print rate and ETA.

**A completion poll was unbounded.**
→ Bound every wait. On timeout, print the state and fall back.

## Process

**Vary one thing.** Every useful result here came from a controlled A/B:
prefetcher in vs out; skid buffer in vs out; shape N vs N±1; random vs
sequential. The wasted hours came from reasoning about symptoms.

**The board can be the better instrument.** A deterministic three-minute
hardware reproduction beat a day-long simulation model that was never
faithful enough to trust.

**Add debug registers early**, before you need them. Counters of accepted
requests and received responses turned "the block is stuck" into
`accepted=444, responses=443`, which is a fact.

**Record the wrong diagnoses.** Each one here was held with confidence and
disproved by a measurement. The list of what was ruled out, and how, is what
stops the next person re-investigating it.

**When you retract, retract in the commit.** A wrong finding committed as a
result will be read as a result.
