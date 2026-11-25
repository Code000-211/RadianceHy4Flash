# T3 (expert prefetch at TP2) -- notes

Branch hy4-tp2-t3 = claude/peaceful-wright-hhqnhi + merge of hy4-xstream (X0 counters/trace) + 1 commit.
NOT merged into the integration branch: it loses. Switches (all default off / TP1 behaviour unchanged):
RADIANCE_XP_TP=1 opens the prefetch at tp>1; RADIANCE_XP_STUDY=1 predicts without copying and prints
precision/recall at exit; RADIANCE_XP_COPIES=N, RADIANCE_XP_MINUSE (0 = never rest), RADIANCE_XP_REST.
What exists: MoeFP8 describes sigmoid-router layers to the prefetch (f32 router + correction bias, score=1),
hyv4flash links xp_next over consecutive MoE layers, xp_logits_f32 + plan "direct" mode, rows up to 64,
up to 8 copies a layer by rows. Data: work/tp2t3/ (run.sh, ab.sh, dec.py, *.out, *.server.log).
Results: see the report; short: study passes only for DFly2 (top-8: precision 45.6 %, recall 38.5 %), nospec
precision 23-30 %; end to end the machinery alone costs 4-6 % (study arm) and copies add 2-5 % more.
Bit-identical greedy text + top-5 logprobs at c1 and c8 (9 sequences), coupled and nospec.
