# Hy4-flash: Expert-Residenz und Decode-Tempo bei gefuelltem KV-Cache (longctx)

Messung, keine Code-Aenderung. Binary-Snapshot `work/longctx/snap` (Kopie von `src/build/bin` + `radiance_home`),
Container `hy4-flash-iqr-v4-u3.rad`, TP2, 210 W, `work/u3/nospec.yaml` und `work/u3/coupled03.yaml` (unveraendert kopiert).
Skripte/Rohdaten: `work/longctx/` (`sweep.py` Treiber, `multi.sh`/`all.sh`/`rerun48.sh` Laeufe, `analyze2.py`, `mdtable.py`,
`res/*.json` Rohdaten inkl. aller /stats-Polls, `logs-*.log` Serverlogs, `tables.md`, `summary.txt`).

## Methode

- Pro Fall (Sessions N, Tiefe D) ein FRISCHER Server (60 s Start, eigene kurze Lock-Haltung je N), damit weder
  Prefix-Cache noch Heat-Historie eines Vorfalls den KV-Fuellstand verfaelschen. Prompts wie BetterBench-Prefill: C++-Quelltext
  aus dem Repo, eindeutiger Nonce je Session (`[review <nonce>]`), 3.05 Zeichen/Token (Prompts sind real 5-10 % kuerzer als D).
- Phase A: alle N Prompts parallel mit max_tokens=1 (Prefill). Phase B: dieselben Prompts, 256 Token, T 0.9, `ignore_eos`;
  Prefix-Resume (nearest) laesst alle N Sessions gleichzeitig decodieren (ttft < 1 s ausser N=8/D=2K: 6-9 s Rest-Prefill).
- Decode-Fenster: /stats alle 1 s gepollt; Deltas (`experts.routed/routed_res/promotions`, `link.h2d_bytes`) zwischen erstem und
  letztem Poll, in dem ALLE N Requests im Zustand decode sind (Prefill-Reste ausgeschlossen). Token = Summe `requests[].output`-Delta.
  tok/s = Summe der Per-Request-Decode-Raten ((gen-1)/(t_end - t_first)). "Hit" = `routed_res/routed` im Fenster (Rank 0).
- KV held = `kv_bytes_used` /stats (live + gecachte fertige Konversationen, in 4-Token-Bloecken von kv_latent_sparse = kv_tokens_used/4).
  "Lent" ist geschaetzt: residente Experten-Einheiten (`experts.tiers[vram].units`) minus Boden 14405 (alles verliehene zurueckgeholt,
  gemessen), mal 1.43 MiB/Einheit; Maximum 5076 Slots = 7.11 GiB. Die Logzeilen enthalten dazu nur die Startzahlen
  (5076 Slots / 7.11 GiB verleihbar, KV 7.33 GiB), keine pro-Request "start/stats"-Zeilen der Leihe; /stats ist die Quelle.
- Link-Anteil = tok/s x miss/tok x 66 us (66 us je verpasstem Expert je Schritt und Rank, experts-streaming.md §0.1). Bei coupled
  zaehlt miss/tok je akzeptiertem Token (Verify-Zeilen verworfener Entwuerfe inklusive).
- N=8 / "60K" ist nicht fahrbar: der KV-Pool traegt real nur ca. 393K Token (kv_bytes_total erreicht nie die 7.33 GiB, s. Befund 3);
  Sessions aus 8x55-60K verdraengen sich gegenseitig (B-ttft 30 s, Rest-Prefill einer Session). N=8 daher mit D=48K (8x45.8K = 367K).
  Ein 6. Prompt (dichter Code) lag bei D=60K mit 75.6K Token ueber max-model-len; `sweep.py` kuerzt solche Prompts jetzt auf Ziel (ADAPT).

## Ergebnis (frischer Server je Zelle)


### no-spec

| N | depth (prompt tok) | KV held GiB / 4-tok blocks (live+cached) | lent to experts GiB (est.) | resident units | hit | miss/tok | promo/1K tok | h2d MB/tok | tok/s agg | tok/s/stream | link share of time | 2xpromo/miss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2048 (1766) | 0.09 / 1327 | 6.7 | 19182 | 0.990 | 6.9 | 7023 | 20.8 | 70 | 69.9 | 3 % | 2.05 |
| 1 | 16384 (16506) | 0.27 / 4586 | 6.5 | 19049 | 0.993 | 4.4 | 932 | 2.7 | 69 | 69.2 | 2 % | 0.43 |
| 1 | 32768 (31413) | 0.47 / 8286 | 6.3 | 18893 | 0.994 | 4.1 | 8932 | 2.0 | 68 | 68.1 | 2 % | 4.41 |
| 1 | 60000 (56326) | 0.81 / 14513 | 5.9 | 18653 | 0.991 | 5.6 | 1317 | 3.9 | 67 | 67.3 | 3 % | 0.47 |
| 4 | 2048 (1786) | 0.29 / 4120 | 6.5 | 19058 | 0.992 | 4.4 | 2951 | 8.8 | 192 | 47.9 | 6 % | 1.36 |
| 4 | 16384 (16019) | 0.97 / 16553 | 5.8 | 18579 | 0.992 | 4.5 | 2624 | 7.4 | 192 | 47.9 | 6 % | 1.17 |
| 4 | 32768 (31906) | 1.84 / 32627 | 5.0 | 17957 | 0.984 | 8.7 | 4341 | 9.2 | 180 | 44.9 | 11 % | 1.00 |
| 4 | 60000 (58731) | 3.30 / 59380 | 3.5 | 16925 | 0.969 | 17.2 | 6689 | 19.7 | 163 | 40.7 | 19 % | 0.78 |
| 8 | 2048 (1872) | 0.58 / 8207 | 6.3 | 18901 | 0.994 | 3.2 | 2025 | 2.3 | 271 | 33.9 | 7 % | 1.28 |
| 8 | 16384 (15854) | 1.92 / 32799 | 4.9 | 17938 | 0.980 | 9.3 | 3053 | 9.0 | 275 | 34.4 | 17 % | 0.65 |
| 8 | 32768 (32366) | 3.69 / 65411 | 3.2 | 16682 | 0.977 | 11.2 | 4182 | 10.6 | 255 | 31.8 | 20 % | 0.75 |
| 8 | 48000 (45823) | 5.16 / 92373 | 1.6 | 15584 | 0.955 | 21.5 | 6282 | 18.7 | 225 | 28.1 | 32 % | 0.59 |

### DFly2 coupled03

| N | depth (prompt tok) | KV held GiB / 4-tok blocks (live+cached) | lent to experts GiB (est.) | resident units | hit | miss/tok | promo/1K tok | h2d MB/tok | tok/s agg | tok/s/stream | link share of time | 2xpromo/miss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2048 (1805) | 0.09 / 1357 | 4.4 | 17586 | 0.974 | 17.7 | 9972 | 26.4 | 113 | 113.3 | 12 % | 1.13 |
| 1 | 16384 (15168) | 0.26 / 4453 | 6.0 | 18702 | 0.990 | 7.7 | 4761 | 10.8 | 102 | 102.1 | 6 % | 1.24 |
| 1 | 32768 (31270) | 0.47 / 8277 | 5.8 | 18547 | 0.985 | 9.8 | 3284 | 10.7 | 110 | 109.9 | 8 % | 0.67 |
| 1 | 60000 (56364) | 0.81 / 14554 | 5.4 | 18306 | 0.975 | 15.8 | 20935 | 59.5 | 103 | 103.1 | 11 % | 2.66 |
| 4 | 2048 (1786) | 0.29 / 4082 | 4.3 | 17459 | 0.979 | 10.4 | 6213 | 18.0 | 216 | 53.9 | 17 % | 1.19 |
| 4 | 16384 (16019) | 0.97 / 16586 | 5.3 | 18234 | 0.975 | 11.5 | 4712 | 13.7 | 242 | 60.6 | 20 % | 0.82 |
| 4 | 32768 (31906) | 1.84 / 32600 | 4.5 | 17610 | 0.963 | 16.8 | 6336 | 14.2 | 234 | 58.5 | 26 % | 0.75 |
| 4 | 60000 (58731) | 3.30 / 59337 | 2.9 | 16514 | 0.939 | 31.0 | 5974 | 17.7 | 195 | 48.7 | 40 % | 0.39 |
| 8 | 2048 (1872) | 0.58 / 8145 | 5.7 | 18512 | 0.988 | 5.1 | 3043 | 8.7 | 310 | 38.7 | 11 % | 1.20 |
| 8 | 16384 (15854) | 1.92 / 32787 | 4.5 | 17598 | 0.945 | 19.4 | 3404 | 10.5 | 311 | 38.9 | 37 % | 0.35 |
| 8 | 32768 (32366) | 3.69 / 65313 | 2.7 | 16338 | 0.921 | 29.0 | 3749 | 11.4 | 265 | 33.1 | 47 % | 0.26 |
| 8 | 48000 (45823) | 5.16 / 92353 | 1.2 | 15294 | 0.887 | 40.9 | 3533 | 8.8 | 216 | 27.0 | 55 % | 0.17 |

Spalte "2xpromo/miss": untere Schranke des Anteils der Misses, die ein Latenz-Miss sind (jede Promotion folgt >= 2 Misses,
experts-streaming.md §0.4). Faellt sie mit der Tiefe, werden Misses zu Kapazitaets-Misses (Einheit passt gar nicht in den Slab).

## Befunde

1. **Residenz sinkt mit dem KV-Fuellstand, aber sanft.** Gesamtzahl resident 18.9K Einheiten (90 %) bei 2K, 15.3-15.6K (73-74 %) bei
   8x46K (5.2 GiB KV held; lent nur noch ~1.2-1.6 GiB). No-spec-Hit: 0.99 bis ca. 65K Token live (<= 1 GiB KV), 0.984 bei 128K,
   0.977 bei 260K, 0.969 bei 235K (N=4), 0.955 bei 367K. DFly2: faengt tiefer an (0.97-0.99 bei 1-4 Sessions kurz) und faellt
   frueher: bei N=8 0.988 (2K) -> 0.945 (16K, 128K live) -> 0.921 (32K) -> 0.887 (48K). Der Einbruch beginnt, sobald das KV
   >= 1-2 GiB haelt (lent < ~5 GiB). Bei 1 Session bleibt alles >= 0.975 (KV <= 0.8 GiB).
2. **Tempo (agg tok/s, Spitze -> tiefster Punkt):** no-spec N=1 70 -> 67 (-4 %), N=4 192 -> 163 (-15 %, 60K), N=8 275 -> 225 (-18 %, 48K).
   coupled N=1 113 -> 103 (-9 %), N=4 242 -> 195 (-20 %, 60K), N=8 311 -> 216 (-31 %, 48K); coupled liegt bei N=8/48K sogar unter no-spec
   (216 vs 225). Prefill-Zeit waechst mit Tiefe wie erwartet (nicht Gegenstand).
3. **Der KV-Pool ist real kleiner als beworben.** Bei 8 Sessions ~55-60K liegt `kv_tokens_used` bei 393K (kv_bytes_total 6.59e9 B von
   7.87e9 carved): ~1.2 GiB der "verleihbaren" Bloecke kommen nicht zurueck (Aktivierungs-Arena-Leihe 1.69 GiB / Slab-Reserven; Ursache
   nicht untersucht). 8 x 65536 passt in der Praxis nicht; Prefix-Evictions verdraengen laufende Sessions.
4. **Gecachte fertige Konversationen halten KV (Prefix-Cache an) und fressen die Leihe dauerhaft.** Kontrolllauf auf EINEM
   durchlaufenden Server (`res/nospec.json`, Faelle aufsteigend, gleiche Methode): nach den langen Sessions blieben ~400K Token KV
   gecacht (`kv_tokens_cached`), resident nur 14.3-14.4K (= Boden, 69 %), und N=8 bei nur 2K Kontext lief mit Hit 0.944, 214 tok/s
   (frisch 0.994 / 271 tok/s, -21 %). Die "99 % mit kurzen Prompts" gilt also nur bei leerem Prefix-Cache; ein warmer Server mit
   Verlauf sitzt dauerhaft auf dem Boden. (Ob/wann Cache-TTL oder Eviction die Leihe zurueckgibt: nicht gemessen.)
5. **link.h2d pro Token** (beide Ranks summiert, enthaelt Promotions + gestreamte Misses): no-spec 2-5 MB/tok kurz (Warm-up-Zellen
   bis 21 MB) -> 19-20 MB/tok bei N=4/60K und N=8/48K; coupled 9-14 MB/tok -> 18 MB (N=4/60K), 60 MB bei N=1/60K (Churn-Ausreisser,
   20.9K Promotions je 1K Token).
6. Promotions je 1K Token steigen mit der Tiefe (no-spec N=8: 2.0K -> 6.3K), also mehr Umlagerung bei kleinerem Slab; die untere
   Schranke des Latenz-Miss-Anteils faellt (N=8 no-spec 1.28 -> 0.59, coupled 1.20 -> 0.17).

## Verdikt: lohnen X2/X3 bei langem Kontext?

Schaetzung wie in experts-streaming.md: Gewinn = 1/(1 - Link-Anteil x f) - 1, f = Anteil der Misses, den X2/X3 entfernen
(Notiz: X2 20-40 %, X2+X3 30-50 % bei kurzem Kontext; bei Tiefe begrenzt durch den Latenz-Miss-Anteil, der in der Tabelle faellt).

| Fall | Link-Anteil | f konservativ / optimistisch | Gewinn X2+X3 |
|---|---|---|---|
| no-spec N=8, 2K | 7 % | 0.4 / 0.5 | +3 % / +4 % |
| no-spec N=8, 32K | 20 % | 0.3 / 0.5 | +6 % / +11 % |
| no-spec N=8, 48K | 32 % | 0.2 / 0.5 | +7 % / +19 % |
| no-spec N=4, 60K | 19 % | 0.2 / 0.5 | +4 % / +10 % |
| coupled N=8, 2K | 11 % | 0.4 / 0.5 | +5 % / +6 % |
| coupled N=8, 32K | 47 % | 0.2 / 0.4 | +11 % / +23 % |
| coupled N=8, 48K | 55 % | 0.15 / 0.4 | +9 % / +28 % |
| coupled N=4, 60K | 40 % | 0.2 / 0.4 | +9 % / +19 % |
| 1 Session, alle Tiefen | 2-12 % | 0.4 / 0.5 | <= +6 % |

**Urteil:** Ja, aber nur fuer Mehr-Session-Betrieb mit Spekulation und mit Einschraenkung. Die Residenz bricht bei <= 1 GiB KV nicht ein
(1 Session: nie relevant, X2/X3 <= +6 %). Ab ~1-2 GiB gehaltenem KV (>= 4 Sessions mit >= 16-32K, oder irgendein warmer Server mit
Prefix-Cache) steigt der Link-Anteil auf 20-55 % der Zeit, und dort sind X2/X3 mit +5-10 % (no-spec) bzw. +9-25 % (DFly2, N=8)
realistisch; die optimistische Obergrenze (+19-28 %) erreicht man nicht, weil mit der Tiefe ein wachsender Teil der Misses
Kapazitaets-Misses sind (Latenz-Miss-Schranke 0.17-0.6), die X2/X3 nicht beseitigen. Sie sind kein Ersatz fuer mehr Kapazitaet: der
groessere Hebel bei langem Kontext ist, das KV zu verkleinern (X6: `kv-cache-dtype i8`, KV-Host-Tier) oder den gecachten
Konversations-KV-Ballast freizugeben (Befund 4: -21 % schon bei 2K Kontext). Empfehlung: X2 (Admission) bauen, weil es auch im
kurzen Betrieb zaehlt; X3 nur nach X6/Cache-Pruefung und nur wenn Mehr-Session-DFly2 bei Tiefe das Produktziel ist.
