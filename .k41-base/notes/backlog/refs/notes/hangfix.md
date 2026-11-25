# hangfix: first request after start hangs (queues idle, rank barrier waits)

Branch hy4-hangfix, commit dc680ce3 (core/device/aql.cpp, aql.h, tests/device_test.cpp).

Symptom: ~half of fresh starts (old 02:55 binary 5/9, rebuilt HEAD 28e35f5d without fix 2/2+) never
produce a token on the first request; watchdog: all queues idle; fatal after 60 s.

gdb (both rank threads): Engine::rank_loop -> tiering_dispatch -> Mover::begin_dispatch -> retire ->
rad_event_record -> aql barrier -> reserve -> full_nap. The step thread waits in RankBarrier.
Stream state (card 0 and 1 alike): wr = rd = 7142 (ring 4096), slot (7142 & 4095) needs 970263,
chain_issued 970265, host read of chain_ctr 970259. The kernels had all finished; the counter is
added to at agent scope in the card's L2 and the host reads it over the BAR, which sees the line
only after an L2 writeback -- an idle card never does it. A lap of non-dispatch packets after
the warm-up's last chained kernels lands the first step on those slots -> wait forever.
Introduced by 2baabc67 (slot_free, 2026-11-19). Not a spec/DFly2/config issue: any config whose
last pre-idle work is chained kernels can hit it (nospec passed 2/2 in my runs, by chance or not).

Fix: barriers (no argument slot) wait for ring room only; a kernel slot wait that the counter
read does not end puts a barrier-bit, system-release barrier with its own signal in front
(chain_flush) and takes its completion as proof (chain_seen = chain_issued at the flush).
Both cases log once ("a barrier goes into a ring slot ...", "a slot wait the chain counter's
read did not end ..."). Seen in a fixed run: read 75544 while 970265 issued -- the BAR read of the
counter can be very stale, so nothing should ever trust it for liveness without a fallback.
Test: device_test a_slot_wait_ends_when_the_hosts_read_of_the_chain_counter_lags_for_good
(aql::chain_read_lag simulates the stale read; without the flush it stalls like the server).
Scripts/logs: work/hangfix/ (try.sh, fresh.sh, inspect.py gdb script, logs/).
