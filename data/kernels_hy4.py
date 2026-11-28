#!/usr/bin/env python3
"""data/kernels_hy4.rkb, the rad-kbench fixture of Hy4-flash's ops at real per-head geometry (WP8).

No container declares every shape: the gpu-mini container (ref/ harness, PLAN 1.1: the real heads,
latent, rope, window, indexer and Engram dims at n = 512) gives the graph's points, M 1..1024;
WP6b's hand-written MLA-with-sink cases give the real contexts; clones of the gpu-mini points give
what gpu-mini cannot declare: the TP2 / TP4 head shares (32 / 64 -> 16 / 8 heads of the absorbed and
non-absorbed attention, the gate and the kv_b projection), the real stream n = 3072 (iHC, the Engram
gate) and M = 6 (a verify of 1 + 5 drafts). Rebuild (host only, libref answers):

  rad-kbench -m gpu-mini.rad --kernels libref --max-ctx 4096 --max-tok 1024 --record gpumini.rkb
  python3 data/kernels_hy4.py raw gpumini.rkb raw_extra.rkb
  rad-kbench --fixture raw_extra.rkb --kernels libref --rerecord extra.rkb
  python3 data/kernels_hy4.py merge gpumini.rkb hy4_mla_sink.rkb extra.rkb data/kernels_hy4.rkb

hy4_mla_sink.rkb is WP6b's (notes/wp6b-mkfix.py, rerecorded). The vision rows (WP15):

  rad-kbench -m gpu-mini-vis.rad --kernels libref --max-ctx 4096 --max-tok 1024 --mm-max-patches 4096 --record vis.rkb
  python3 data/kernels_hy4.py vis data/kernels_hy4.rkb vis.rkb data/kernels_hy4.rkb

WP16.3 appended mla_project_kv points at 16 / 8 heads (`wp163`, `append` below).

WP16.2 appended the iHC junction
(rmsnorm_ihc_junction: the gpu-mini graph's points, n = 3072 clones, M = 6 and 16) and ihc_pre /
ihc_post at n = 3072 M = 16, from a recording of the graph with the junction declared:

  rad-kbench -m gpu-mini.rad --kernels libref --max-ctx 4096 --max-tok 1024 --record gpumini2.rkb
  python3 data/kernels_hy4.py junction gpumini2.rkb raw_junction.rkb
  rad-kbench --fixture raw_junction.rkb --kernels libref --rerecord junction.rkb
  python3 data/kernels_hy4.py append data/kernels_hy4.rkb junction.rkb appended.rkb
  rad-kbench --fixture appended.rkb --kernels libref --rerecord data/kernels_hy4.rkb

(The last rerecord matters: rad-kbench draws a weight by position, dtype and extents, so a case
recorded alone can draw different bytes inside the whole fixture.)

The DFly2 draft pass (WP13.7), recorded with the drafter declared and appended (no whole-fixture rerecord: the
appended cases replay against their recording as they are, and a rerecord would rewrite the junction rows' answers):

  rad-kbench -m gpu-mini.rad --kernels libref --max-ctx 4096 --max-tok 1024 --num-speculative-tokens 5 \
      --op dfly_sel --op row_topk --op mla_attn_sel_sink --record dfly.rkb
  python3 data/kernels_hy4.py dfly data/kernels_hy4.rkb dfly.rkb data/kernels_hy4.rkb

Check a card with
  rad-kbench --fixture data/kernels_hy4.rkb --kernels libr4d:libref [--bench]
"""
import copy, struct, sys
ENC = 568   # sizeof(RadEncoding)
class R:
    def __init__(s, b): s.b, s.p = b, 0
    def take(s, n): v = s.b[s.p:s.p + n]; s.p += n; return v
    def u8(s): return s.take(1)[0]
    def u32(s): return struct.unpack('<I', s.take(4))[0]
    def u64(s): return struct.unpack('<Q', s.take(8))[0]
    def i64(s): return struct.unpack('<q', s.take(8))[0]
    def f64(s): return struct.unpack('<d', s.take(8))[0]
    def str(s): return s.take(s.u32()).decode()
def load(path):
    r = R(open(path, 'rb').read())
    assert r.take(4) == b'RKB1'
    fx = dict(version=r.u32(), samples=r.u32(), source=r.str(), oracle=r.str(), recorded=r.str(), cases=[])
    for _ in range(r.u32()):
        c = dict(op=r.str(), seed=r.u64(), m=r.i64(), band=r.str(), params=[], opd=[])
        for _ in range(r.u32()):
            c['params'].append([r.str(), r.u32(), r.u64(), r.f64(), r.str()])
        for _ in range(r.u32()):
            o = dict(role=r.u32(), dtype=r.u32())
            o['shape'] = [r.i64() for _ in range(r.u32())]
            o['absent'] = r.u8(); o['fill'] = r.u32(); o['flags'] = r.u8()
            o['idx_max'] = r.i64(); o['idx_const'] = r.i64(); o['fill_chunk'] = r.i64()
            o['has_enc'] = r.u8()
            if o['has_enc']:
                o['enc'] = r.take(ENC); o['sel'] = [(r.u32(), r.i64(), r.i64()) for _ in range(r.u32())]
            o['have_ref'] = r.u8()
            if o['have_ref']:
                o['ref'] = (r.i64(), r.f64(), r.i64(), r.take(4 * r.u32()), None)
                n = r.u32(); o['sketch'] = r.take(8 * n); o['nsk'] = n
            c['opd'].append(o)
        fx['cases'].append(c)
    assert r.p == len(r.b)
    return fx
def save(fx, path):
    out = []
    P = lambda f, v: out.append(struct.pack(f, v))
    def s(x): b = x.encode(); P('<I', len(b)); out.append(b)
    out.append(b'RKB1'); P('<I', 6); P('<I', 1024); s(fx['source']); s(fx['oracle']); s(fx['recorded'])
    P('<I', len(fx['cases']))
    for c in fx['cases']:
        s(c['op']); P('<Q', c['seed']); P('<q', c['m']); s(c['band'])
        P('<I', len(c['params']))
        for k, kind, i, d, st in c['params']:
            s(k); P('<I', kind); P('<Q', i & 0xffffffffffffffff); P('<d', d); s(st)
        P('<I', len(c['opd']))
        for o in c['opd']:
            P('<I', o['role']); P('<I', o['dtype']); P('<I', len(o['shape']))
            for x in o['shape']: P('<q', x)
            P('<B', o['absent']); P('<I', o['fill']); P('<B', o['flags'])
            P('<q', o['idx_max']); P('<q', o['idx_const']); P('<q', o['fill_chunk'])
            P('<B', o['has_enc'])
            if o['has_enc']:
                out.append(o['enc']); P('<I', len(o['sel']))
                for a, b, c2 in o['sel']: P('<I', a); P('<q', b); P('<q', c2)
            hr = o.get('have_ref', 0)
            P('<B', hr)
            if hr:
                n, ss, nf, smp, _ = o['ref']; P('<q', n); P('<d', ss); P('<q', nf)
                P('<I', len(smp) // 4); out.append(smp); P('<I', o['nsk']); out.append(o['sketch'])
    open(path, 'wb').write(b''.join(out))
def par(c, k):
    for p in c['params']:
        if p[0] == k: return p[2] if p[1] == 1 else (p[3] if p[1] == 4 else p[4])
    return None
HY = ['ihc_pre', 'ihc_post', 'rmsnorm_ihc_junction', 'gate_mul', 'mla_project_kv', 'mla_kv_split', 'mla_attn_paged_sink', 'mla_attn_sel_sink',
      'attn_kv_sink', 'idx_pool_gated', 'idx_score', 'engram_norm', 'engram_gate', 'embed_lookup_q', 'quant_act_fp8',
      'dequant', 'qsa_select', 'qsa_work', 'qsa_tail_store', 'swiglu_clamped', 'router_sigmoid_topk']
HEADS = ['mla_project_kv', 'mla_kv_split', 'mla_attn_paged_sink', 'mla_attn_sel_sink', 'attn_kv_sink', 'gate_mul']
WIDE = ['ihc_pre', 'ihc_post', 'engram_gate']

def setp(c, k, v):
    for p in c['params']:
        if p[0] == k:
            if p[1] == 1: p[2] = v
            else: p[3] = float(v)
def fresh(c, salt, keep_encodings=False):
    c = copy.deepcopy(c)
    c['seed'] = (c['seed'] * 0x9E3779B97F4A7C15 + salt) & 0xffffffffffffffff
    for o in c['opd']:
        o['have_ref'] = 0; o.pop('ref', None)
        # Drawn plain, unless the caller keeps the container's encodings: a clone that resizes the
        # operand an encoding was recorded for must drop it, one that leaves it alone must not.
        if o['has_enc'] and not keep_encodings:
            o['has_enc'] = 0; o.pop('enc', None); o.pop('sel', None)
    return c

def heads_clone(c, H2, salt):
    H = par(c, 'heads') if c['op'] != 'gate_mul' else par(c, 'n') // 256
    d = fresh(c, salt)
    m = {H: H2, H * 448: H2 * 448, H * 256: H2 * 256}
    for o in d['opd']:
        sh = o['shape']
        if c['op'] == 'gate_mul' and len(sh) == 2: sh[1] = m.get(sh[1], sh[1])
        elif len(sh) == 3 and sh[1] == H: sh[1] = H2                              # q / k / v / out [M, H, d]
        elif o['role'] == 3 and len(sh) == 1 and sh[0] == H: sh[0] = H2           # sinks
        elif o['role'] == 3 and len(sh) == 2 and sh[0] == H * 448: sh[0] = H2 * 448   # kv_b
        elif c['op'] == 'mla_kv_split' and len(sh) == 2 and sh[1] == H * 448: sh[1] = H2 * 448
    if c['op'] == 'gate_mul': setp(d, 'n', H2 * 256)
    else: setp(d, 'heads', H2)
    d['band'] = f"{c['band']} heads {H2}"
    return d

def wide_clone(c, n2, salt):
    n = par(c, 'n'); hc = par(c, 'hc')
    d = fresh(c, salt)
    m = {n: n2, n * hc: n2 * hc}
    for o in d['opd']: o['shape'] = [m.get(x, x) if i > 0 or len(o['shape']) == 1 else x for i, x in enumerate(o['shape'])]
    setp(d, 'n', n2); d['band'] = f"{c['band']} n {n2}"
    return d

def m_clone(c, M2, salt):
    # The M rows are the only thing this touches, so the weights -- and the affine `fwht128` codes
    # the container recorded with them -- are still the container's: a quantised row handed a plain
    # weight answers something else than the reference (tests/kbench_recipe_test.py).
    M = c['m']; d = fresh(c, salt, keep_encodings=True)
    for o in d['opd']:
        if o['shape'] and o['shape'][0] == M and not (o['flags'] & 4) and o['role'] != 3: o['shape'][0] = M2
        if o['idx_max'] == M and (o['flags'] & 4): o['idx_max'] = M2
    d['m'] = M2; setp(d, 'M', M2); d['band'] = f"{c['band']} M {M2}"
    return d

def raw(src, out):
    fx = load(src)
    base = [c for c in fx['cases'] if c['op'] in HEADS + WIDE]
    new, salt = [], 1
    for c in base:
        if c['m'] not in (1, 8, 64, 1024): continue
        if c['op'] in HEADS:
            H = par(c, 'heads') if c['op'] != 'gate_mul' else par(c, 'n') // 256
            for H2 in (H // 2, H // 4) if H == 32 or H == 64 else ():
                if c['op'] == 'attn_kv_sink' and c['m'] == 1024 and H2 == 8: continue
                new.append(heads_clone(c, H2, salt)); salt += 1
        if c['op'] in WIDE and par(c, 'n') == 512:
            new.append(wide_clone(c, 3072, salt)); salt += 1
        if c['m'] == 8 and c['op'] in ('mla_project_kv', 'mla_attn_paged_sink', 'gate_mul', 'ihc_pre', 'ihc_post'):
            new.append(m_clone(c, 6, salt)); salt += 1
    seen, uniq = set(), []
    for c in new:
        k = (c['op'], repr(c['params']), repr([(o['role'], o['dtype'], o['shape']) for o in c['opd']]))
        if k not in seen: seen.add(k); uniq.append(c)
    new = uniq
    x = dict(fx, source='hy4-flash TP shards / real n / M 6 (WP8, clones of the gpu-mini graph)', cases=new)
    save(x, out); print(len(new), 'cases')

def wp163(src, out):
    """WP16.3's points of mla_project_kv: the TP2 / TP4 head shares (16 / 8) at the heights the
    gpu-mini graph only declares at 32 / 64 heads -- a verify (6), a full token tile (16), one past
    it (17, the mb law's step), 128 and a ragged prefill chunk (1000) -- cloned from the recorded
    16 / 8-head points and recorded again by libref:
      python3 data/kernels_hy4.py wp163 data/kernels_hy4.rkb raw163.rkb
      rad-kbench --fixture raw163.rkb --kernels libref --rerecord extra163.rkb
      python3 data/kernels_hy4.py append data/kernels_hy4.rkb extra163.rkb data/kernels_hy4.rkb"""
    fx = load(src)
    have = {(c['op'], par(c, 'mode'), par(c, 'heads'), c['m']) for c in fx['cases']}
    base = {}
    for c in fx['cases']:
        if c['op'] == 'mla_project_kv' and par(c, 'heads') in (8, 16) and c['m'] == 8:
            base[(par(c, 'mode'), par(c, 'heads'))] = c
    new, salt = [], 16300
    for (mode, H), c in sorted(base.items()):
        for M2 in (6, 16, 17, 128, 1000):
            if ('mla_project_kv', mode, H, M2) in have: continue
            new.append(m_clone(c, M2, salt)); salt += 1
    x = dict(fx, source='hy4-flash: mla_project_kv at the TP2 / TP4 head shares, WP16.3 heights', cases=new)
    save(x, out); print(len(new), 'cases')

def k41_prefill(src, out):
    """K-41's points of mla_attn_paged_sink: a 2048-row prefill chunk at the TP2 head shares (16 and
    32), which the recording only declares up to M 1024 -- the windowed pair (block 16) and the dense
    one (block 4), cloned from their M 1024 points and recorded again by libref:
      python3 data/kernels_hy4.py k41_prefill data/kernels_hy4.rkb raw_k41.rkb
      rad-kbench --fixture raw_k41.rkb --kernels libref --rerecord k41.rkb
      python3 data/kernels_hy4.py append data/kernels_hy4.rkb k41.rkb data/kernels_hy4.rkb"""
    fx = load(src)
    new, salt = [], 41000
    for c in fx['cases']:
        if c['op'] != 'mla_attn_paged_sink' or c['m'] != 1024: continue
        if par(c, 'heads') not in (16, 32): continue
        new.append(m_clone(c, 2048, salt)); salt += 1
    # The recording predates the row's optional `rope_pos` operand: the row declares it, the case must
    # carry it, so the clones hand it over absent (the geometry without positions), as opd_none() does.
    for c in new:
        if len(c['opd']) == 8:
            c['opd'].append(dict(role=0, dtype=11, shape=[c['m']], absent=1, fill=0, flags=0,
                                 idx_max=0, idx_const=-1, fill_chunk=0, has_enc=0))
    seen, uniq = set(), []
    for c in new:
        k = (c['op'], repr(c['params']), repr([(o['role'], o['dtype'], o['shape']) for o in c['opd']]))
        if k not in seen: seen.add(k); uniq.append(c)
    x = dict(fx, source='hy4-flash: mla_attn_paged_sink prefill chunk at M 2048, TP2 head shares (K-41)',
             cases=uniq)
    save(x, out); print(len(uniq), 'cases')

def append(base, extra, out):
    """base's cases, then extra's (recorded) ones that base does not have."""
    fx, ex = load(base), load(extra)
    key = lambda c: (c['op'], repr(c['params']), repr([(o['role'], o['dtype'], o['shape']) for o in c['opd']]))
    seen = {key(c) for c in fx['cases']}
    add = [c for c in ex['cases'] if key(c) not in seen and any(o.get('have_ref') for o in c['opd'])]
    fx['cases'] += add
    save(fx, out); print(len(add), 'cases added,', len(fx['cases']), '->', out)

def junction(src, out):
    """The junction's graph points (n = 512) as recorded, their n = 3072 clones at M 1, 8, 64, 1024,
    and M 6 / 16 clones of the n = 3072 M 8 points of the junction, ihc_pre and ihc_post."""
    fx = load(src)
    new, salt = [], 7001
    for c in fx['cases']:
        if c['op'] == 'rmsnorm_ihc_junction': new.append(fresh(c, salt)); salt += 1
    wide = []
    for c in fx['cases']:
        if c['op'] in ('rmsnorm_ihc_junction', 'ihc_pre', 'ihc_post') and par(c, 'n') == 512 and c['m'] in (1, 8, 64, 1024):
            w = wide_clone(c, 3072, salt); salt += 1
            if c['op'] == 'rmsnorm_ihc_junction': new.append(w)
            if c['m'] == 8: wide.append(w)
    for w in wide:
        for M2 in ((6, 16) if w['op'] == 'rmsnorm_ihc_junction' else (16,)):
            new.append(m_clone(w, M2, salt)); salt += 1
    seen, uniq = set(), []
    for c in new:
        k = (c['op'], repr(c['params']), repr([(o['role'], o['dtype'], o['shape']) for o in c['opd']]))
        if k not in seen: seen.add(k); uniq.append(c)
    x = dict(fx, source='hy4-flash iHC junction / ihc at n 3072 (WP16.2, the gpu-mini graph and clones)', cases=uniq)
    save(x, out); print(len(uniq), 'cases')

def merge(*paths):
    out = paths[-1]; cases = []
    for i, p in enumerate(paths[:-1]):
        fx = load(p)
        seen = set()
        for c in fx['cases']:
            if i == 0 and c['op'] not in HY: continue
            if not any(o.get('have_ref') for o in c['opd']): continue   # the oracle refused it at record
            k = (c['op'], repr(c['params']), repr([(o['role'], o['dtype'], o['shape']) for o in c['opd']]))
            if i == 0 and k in seen: continue      # the same point a layer later: one is enough
            seen.add(k); cases.append(c)
    head = dict(load(paths[0]), source='hy4-flash: gpu-mini graph + real geometry / TP shards (WP8)', cases=cases)
    save(head, out); print(len(cases), 'cases ->', out)

# WP15: the vision rows, from a recording of the gpu-mini-vis container with --mm-max-patches (the tower
# at the real head width 72, its merger, the newline lookup, the media rows' scatter, and the trunk's
# and the indexer's M-RoPE forms).
VIS = ('grid_embed_sf', 'conv2d_unfold', 'attn_dense', 'layernorm', 'gemm_nt_bias', 'scatter_rows')
def vis_case(c):
    if c['op'] in VIS: return True
    if c['op'] == 'embed_lookup': return par(c, 'n_vocab') == 1          # the newline row
    if c['op'] in ('rope', 'idx_pool_gated'):
        return any(p[0] == 'mode' and p[4] == 'imrope_gptj' for p in c['params'])
    return False

def vis(base, rec, out):
    """base fixture + the vision cases of `rec` (rad-kbench -m gpu-mini-vis.rad --mm-max-patches 4096
    --kernels libref --record rec), each distinct point once."""
    fx, r = load(base), load(rec)
    have = {(c['op'], repr(c['params']), repr([(o['role'], o['dtype'], o['shape']) for o in c['opd']]))
            for c in fx['cases']}
    add = []
    for c in r['cases']:
        if not vis_case(c) or not any(o.get('have_ref') for o in c['opd']): continue
        k = (c['op'], repr(c['params']), repr([(o['role'], o['dtype'], o['shape']) for o in c['opd']]))
        if k in have: continue
        have.add(k); add.append(c)
    fx = dict(fx, source=fx['source'] + ' + vision (WP15, gpu-mini-vis)', cases=fx['cases'] + add)
    save(fx, out); print(len(add), 'vision cases added ->', out, len(fx['cases']), 'in all')

# WP13.7: the DFly2 draft pass's own points, from a recording of the gpu-mini container with the drafter
# declared (rad-kbench --num-speculative-tokens 5): dfly_sel (blocks 514 over 512 recalled pages), the chain's
# row_topk over the vocabulary, and the draft attention at the table width 514.
def dfly_case(c):
    if c['op'] in ('dfly_sel', 'row_topk'): return True
    return c['op'] == 'mla_attn_sel_sink' and par(c, 'blocks') == 514

def dfly(base, rec, out):
    """base fixture + the draft pass's cases of `rec`, each distinct point once."""
    fx, r = load(base), load(rec)
    key = lambda c: (c['op'], repr(c['params']), repr([(o['role'], o['dtype'], o['shape']) for o in c['opd']]))
    have = {key(c) for c in fx['cases']}
    add = []
    for c in r['cases']:
        if not dfly_case(c) or not any(o.get('have_ref') for o in c['opd']) or key(c) in have: continue
        have.add(key(c)); add.append(c)
    fx = dict(fx, source=fx['source'] + ' + DFly2 draft pass (WP13.7)', cases=fx['cases'] + add)
    save(fx, out); print(len(add), 'draft cases added ->', out, len(fx['cases']), 'in all')

# G-3 (levers-general.md L3): the fused rows' cases, written by hand (the gpu-mini graph declares none of
# them), recorded by libref (the fused ops run their chain there) and appended:
#   python3 data/kernels_hy4.py g3 raw_g3.rkb
#   rad-kbench --fixture raw_g3.rkb --kernels libref --rerecord g3.rkb
#   python3 data/kernels_hy4.py append data/kernels_hy4.rkb g3.rkb data/kernels_hy4.rkb
BF16, F32, I32, I8 = 3, 1, 11, 8
IN_, OUT_, W_ = 0, 1, 3
PINT, PSTR, PF64 = 1, 2, 4
def mkcase(op, M, params, opds, band, seed):
    c = dict(op=op, seed=seed, m=M, band=band, params=[], opd=[])
    for k, v in params:
        if isinstance(v, str): c['params'].append([k, PSTR, 0, 0.0, v])
        elif isinstance(v, float): c['params'].append([k, PF64, 0, v, ''])
        else: c['params'].append([k, PINT, v, 0.0, ''])
    for role, dt, shape, *rest in opds:
        c['opd'].append(dict(role=role, dtype=dt, shape=list(shape), absent=1 if rest and rest[0] == 'absent' else 0,
                             fill=0, flags=0, idx_max=0, idx_const=-1, fill_chunk=0, has_enc=0))
    return c
def g3a_cases(base):
    """swiglu_clamped_had_quant_i8g: the routed arm's rows (decode: 1..8 tokens x top_k 8 slots; a 128-column
    group count of the TP2 / TP4 widths 768 / 384 and a narrow row), with and without the bf16 plane."""
    out = []
    salt = base
    for n, lim in ((768, 10.0), (768, 0.75), (384, 0.75), (128, 0.5), (256, 10.0)):
        for M in (1, 8, 17, 64, 65, 200):
            for plane in (False, True):
                if n != 768 and M not in (8, 65): continue
                p = [('M', M), ('n', n), ('limit', lim), ('group', 128), ('dtype', 'bf16')]
                ops = [(IN_, BF16, (M, 2 * n)), (OUT_, I8, (M, n)), (OUT_, F32, (M, n // 128)),
                       (OUT_, BF16, (M, n)) if plane else (OUT_, BF16, (M, n), 'absent')]
                out.append(mkcase('swiglu_clamped_had_quant_i8g', M, p, ops,
                                  f"g3a swiglu_clamped_had_quant_i8g M={M} n={n} limit={lim}{' +plane' if plane else ''}", salt)); salt += 1
    return out
def g3c_cases(base):
    """mla_expand_gate_had_quant_i8g at the TP2 / TP4 head shares (16 / 8 sparse, 32 / 16 sliding heads) and the
    decode-class heights (1, 6 a verify, 8, 16, 17, 64), with and without the bf16 plane."""
    out, salt = [], base
    for H in (16, 8, 32):
        for M in (1, 6, 8, 16, 17, 64):
            for plane in (False, True):
                if H != 16 and M not in (1, 17, 64): continue
                p = [('M', M), ('heads', H), ('k_dim', 192), ('v_dim', 256), ('latent_dim', 512), ('group', 128), ('dtype', 'bf16')]
                ops = [(IN_, BF16, (M, H, 512)), (W_, BF16, (H * 448, 512)), (IN_, BF16, (M, H * 256)),
                       (OUT_, I8, (M, H * 256)), (OUT_, F32, (M, H * 2)),
                       (OUT_, BF16, (M, H * 256)) if plane else (OUT_, BF16, (M, H * 256), 'absent')]
                out.append(mkcase('mla_expand_gate_had_quant_i8g', M, p, ops,
                                  f"g3c mla_expand_gate_had_quant_i8g M={M} H={H}{' +plane' if plane else ''}", salt)); salt += 1
    return out
def g3b_cases(src, base):
    """rmsnorm_ihc_junction with the next norm's rotated int8 codes (`group` 128, xq / xs operands): the recorded
    tail junction points (gpu-mini n 512, the n 3072 clones) at M 1, 6, 8, 16, 64, with and without the addend."""
    fx = load(src)
    out, salt = [], base
    for c in fx['cases']:
        if c['op'] != 'rmsnorm_ihc_junction' or c['m'] not in (1, 6, 8, 16, 64): continue
        if par(c, 'tail') == 0 or par(c, 'n') % 128: continue
        d = fresh(c, salt); salt += 1
        M, n = c['m'], par(c, 'n')
        d['params'].append(['group', PINT, 128, 0.0, ''])
        for dt, shape in ((I8, [M, n]), (F32, [M, n // 128])):
            d['opd'].append(dict(role=OUT_, dtype=dt, shape=shape, absent=0, fill=0, flags=0, idx_max=0,
                                 idx_const=-1, fill_chunk=0, has_enc=0))
        d['band'] = c['band'] + ' +codes'
        out.append(d)
    return out
def g3(out, src='data/kernels_hy4.rkb'):
    cases = g3a_cases(0x93a00000) + g3b_cases(src, 0x93b00000) + g3c_cases(0x93c00000)
    fx = dict(version=6, samples=1024, source='hy4-flash G-3 fused launches (levers-general L3)', oracle='libref',
              recorded='2026-11-22', cases=cases)
    save(fx, out); print(len(cases), 'cases ->', out)

# T-04: the Hy4 TP2 shapes the fixture does not hold after T-05's re-record from the container (the
# container declares the TP1 geometry -- 32 sparse / 64 sliding heads -- and M steps that jump 16 -> 32),
# and the chunk heights `gemm_nt_q` is launched at (its sweep stops at 512). Written by hand, recorded by
# libref, appended. The source is the fixture as it was before T-04, never the fixture itself: the clones
# it writes are trunk cases too, so reading back what was appended would clone the clones.
#   git show ac66d3a0:data/kernels_hy4.rkb > base_hy4.rkb
#   python3 data/kernels_hy4.py t04 raw_t04.rkb base_hy4.rkb
#   rad-kbench --fixture raw_t04.rkb --kernels libref --rerecord t04.rkb
#   python3 data/kernels_hy4.py append base_hy4.rkb t04.rkb data/kernels_hy4.rkb
# The trunk's own rows (docs/ARCHITECTURES.md, `hyv4flash`: one layer's chain), which D-10 wants at the
# verify heights: the embed / engram lines, the junction, the attention linears and their norms, the
# indexer, the MLA absorb / sink / sel / expand, the MoE router and the lm_head. The M clone keeps the
# encodings the container recorded with a case's operands -- it resizes the extents a launch at the new
# height has and nothing else, so the weights are still the container's, and a quantised row handed plain
# weights answers something else than the reference (tests/kbench_recipe_test.py). Those extents are not
# all M: `T04_ROW_AXIS` holds the row axes that sit elsewhere in an operand and `T04_M_RULE` the two ops
# whose other extents their own reference derives from the height and the sequences the batch holds, so a
# clone is the row a launch at that height really runs, not the M 8 operands M tall. The routed arm is not
# here because it is not in this fixture at all: gpu-mini declares no expert arm, kernels_moe.rkb is
# Qwen's 256-expert geometry, and a Hy4 case for `moe_gemm_q`'s rows needs a container that declares the
# routed arm (its u5-trellis expert weights), not a clone. D-10's 47 % gate_up is a recording of its own.
TRUNK = ('embed_lookup', 'cast', 'engram_norm', 'engram_gate', 'grid_embed_sf', 'iota', 'sub_i32',
         'ihc_pre', 'ihc_post', 'rmsnorm', 'rmsnorm_ihc_junction', 'gemm_nt_q', 'gemm_nt', 'gemm_nt_f32',
         'gemm_nt_bias', 'rope', 'rope_rows', 'kv_store', 'gate_mul', 'mla_kv_split', 'attn_kv_sink',
         'mla_project_kv', 'mla_attn_paged_sink', 'mla_attn_sel_sink', 'mla_expand_gate',
         'mla_expand_gate_had_quant_i8g', 'had_quant_act_i8g', 'swiglu_clamped_had_quant_i8g',
         'router_sigmoid_topk', 'gather_rows', 'qsa_work', 'idx_pool_gated', 'qsa_tail_store', 'idx_score',
         'qsa_select', 'logits_gemm')

def t04a_sliding(src):
    """mla_attn_paged_sink: the 30 sliding layers at TP2 (32 heads a rank, window 512), M = 1, `stable` 0
    and 1 -- the shape K-14's 19 us `merge_sink` bracket is about, which no recorded case reaches (the
    container's sliding geometry is 64 heads). Its chained form is replayed with `--chain-forms`."""
    out, salt = [], 0xa04a0000
    for c in load(src)['cases']:
        if c['op'] != 'mla_attn_paged_sink' or par(c, 'window') != 512: continue
        if par(c, 'heads') != 64 or c['m'] != 1: continue
        out.append(heads_clone(c, 32, salt)); salt += 1
    return out

def t04b_sel(src):
    """mla_attn_sel_sink: the 12 sparse layers at TP2 (16 heads a rank), M = 1 and 8, over the recorded
    512-page cache. The table is a real `qsa_select` one when timed: `timing_input` draws this op's table
    as each row's own distinct, ascending page ids (block order at M = 1, a sorted draw out of the page
    range otherwise), not one shuffled run over the whole operand. Every other index table keeps its
    permutation."""
    out, salt = [], 0xb04b0000
    for c in load(src)['cases']:
        if c['op'] != 'mla_attn_sel_sink' or par(c, 'heads') != 32 or c['m'] not in (1, 8): continue
        out.append(heads_clone(c, 16, salt)); salt += 1
    return out

# The row axes that do not sit in shape[0], by op: `(operand, axis, ratio)`, and whether the operand's
# index range follows the axis. A cu histogram is M + 1 long (the `flags & 4` rule in `t04_clone`), the
# Engram grid's and the rope's position table are [components, M], and the indexer's pooling reads one
# row per M * ratio -- one place for the rule instead of a branch per op. An op whose operands are not all
# M tall names a rule of its own, `(d, c, M2)`, because no single axis describes it: `kv_store` and
# `qsa_work` are sized by their reference's derivation, below.
T04_ROW_AXIS = {'grid_embed_sf': (2, 1, False, False), 'rope': (1, 1, False, True),
                'rope_rows': (1, 1, False, True), 'idx_pool_gated': (0, 0, True, False)}

def t04_kv_store(d, c, M2):
    """`kv_store`: the cache the store scatters into is `ceil(M / block_size) + 2` blocks deep and the
    slot table is drawn over every slot of it -- the reference's own derivation, `ref_kv_store_shape`, and
    the depth a launch at this height has. A clone that keeps the M 8 cache scatters a whole height's
    tokens onto the handful of slots that height needed."""
    bs = int(par(c, 'block_size'))
    nb = (M2 + bs - 1) // bs + 2
    for o in d['opd']:
        if o['absent'] or not o['shape']: continue
        if len(o['shape']) == 4: o['shape'][0] = nb          # the cache: [blocks, kv heads, block, row]
        elif len(o['shape']) == 1: o['idx_max'] = nb * bs     # the slot table: an id a token, over it all

def t04_qsa_work(d, c, M2):
    """`qsa_work`: its operands are sized by two different counts, neither of them M. The work list holds
    `ceil(M / ratio) + seqs` groups -- the arch's `work` at a graph declared for this height
    (`hyv4flash_common.h`) -- so the stage, the page list and the position list follow it, and its rows are
    one a sequence, `min(seqs, M)` of them, so the tail, the slot index, the block table and the counts
    follow those (`ref_qsa_work_shape` derives both). `m_clone` makes every one of them M tall: a tail ring,
    a block table and a count at a sequence count no launch has."""
    r, seqs = int(par(c, 'ratio')), int(par(c, 'seqs'))
    work, S = (M2 + r - 1) // r + seqs, min(seqs, M2)
    s0, w0 = min(seqs, c['m']), int(par(c, 'work'))
    n, ring = int(par(c, 'n')), int(par(c, 'ring'))
    for o, s in zip(d['opd'], c['opd']):
        sh = tuple(s['shape'])
        if s['absent'] or not sh: continue
        if s['role'] == 0 and sh in ((c['m'], n), (c['m'],)): continue   # k and pos: M tall, already moved
        if sh == (s0 + 1,): o['shape'] = [S + 1]                # cu: one entry a sequence, and the 0
        elif sh == (s0, 1, ring, n): o['shape'] = [S, 1, ring, n]   # the tail: a ring row a sequence
        elif sh == (s0, 1): o['shape'] = [S, 1]; o['idx_max'] = S  # the slot index, one a sequence
        elif sh == (s0, 2): o['shape'] = [S, 2]                 # the block table
        elif sh == (s0,): o['shape'] = [S]                      # the count: one a sequence
        elif sh == (w0 * r, n): o['shape'] = [work * r, n]      # the stage: a row a group of the list
        elif sh == (w0,): o['shape'] = [work]                   # the page and position lists
        else: raise SystemExit(f'qsa_work: operand {s["shape"]} is not one of the shapes this rule knows')
    setp(d, 'work', work)

T04_M_RULE = {'kv_store': t04_kv_store, 'qsa_work': t04_qsa_work}

def t04_clone(c, M2, salt):
    """`m_clone`, plus the row axes that do not sit in shape[0] (`T04_ROW_AXIS`) and the two ops sized by
    their reference's derivation rather than by M (`T04_M_RULE`). `m_clone` moves shape[0] and leaves the
    rest, which is the right clone only for a row whose every operand has M first."""
    d = m_clone(c, M2, salt)
    for o in d['opd']:
        if o['flags'] & 4 and o['shape'] and o['shape'][0] == c['m'] + 1: o['shape'][0] = M2 + 1
    rule = T04_M_RULE.get(c['op'])
    if rule is not None:
        rule(d, c, M2); return d
    rule = T04_ROW_AXIS.get(c['op'])
    if rule is not None:
        idx, axis, ratio, follows = rule
        o = d['opd'][idx]
        o['shape'][axis] = M2 * par(c, 'ratio') if ratio else M2
        if follows: o['idx_max'] = o['shape'][axis]
    return d

def t04c_verify(src):
    """every trunk row at the c8 verify heights M = 24 / 40 / 48 (D-10), cloned from that row's M = 8 case
    so it keeps the band it has a decode-class case in (a row whose cases start at 65 is prefill-class and
    is left)."""
    fx, out, salt = load(src), [], 0xc04c0000
    for c in fx['cases']:
        if c['op'] not in TRUNK or c['m'] != 8: continue
        for M2 in (24, 40, 48):
            out.append(t04_clone(c, M2, salt)); salt += 1
    return out

def t04d_gemm_q(src):
    """gemm_nt_q at the pipelined chunk heights M = 1024 / 2048 / 3072 (`kPipeMinRows` and the 4096-row
    chunk's halves) at the trunk's two dense planes (K = 1536, the absorb's rank, and 8192, d_model): the
    recorded sweep stops at 512, so a chunk's GEMM is timed at a band no case reaches. The expert planes
    (K = 13824, 16384) keep their recorded sweep at the heights: a full-height expert GEMM is 150 GFLOP
    of reference per case, and the card-less gate replays every case. q_a's own shape is `t04e` below:
    every recorded case is N 3072, so the Verify pair's N 1024 / K 3072 had no case at any M."""
    fx, out, salt = load(src), [], 0xd04d0000
    for c in fx['cases']:
        if c['op'] != 'gemm_nt_q' or c['m'] != 512 or par(c, 'K') not in (1536, 8192): continue
        for M2 in (1024, 2048, 3072):
            out.append(m_clone(c, M2, salt)); salt += 1
    return out

def t04e_gemm_q_a(src):
    """gemm_nt_q at q_a's shape (N 1024, K 3072), M 2048: the Verify pair K-51 and audit F7.1 name,
    `(2048, 1024, 3072)` against `(2048, 3072, 8192)`, and no case had N 1024 or K 3072 at any M (every
    recorded `gemm_nt_q` case is the trunk's N 3072 planes). Cloned from the K 1536 dense case with the
    container's affine `fwht128` encoding kept: an encoding names the group and the transform, not the
    extents, and 3072 is 24 groups of 128, so it still describes the 1024 x 3072 weight. The container
    files the trunk's dense planes as i8a8h, so the case reaches the i8a8 rows at q_a's shape, not q_a's
    own u5-trellis row (`gemm_iqr_tiled`), which no fixture holds (K-06, K-51)."""
    N, K, M = 3072, 1536, 512
    base = None
    for c in load(src)['cases']:
        if (c['op'] == 'gemm_nt_q' and c['m'] == M and par(c, 'N') == N and par(c, 'K') == K
                and par(c, 'dtype') == 'i8a8h'):
            base = c
            break
    if base is None:
        raise SystemExit(f'no gemm_nt_q N {N} K {K} M {M} case to clone q_a from')
    d = fresh(base, 0xe04e0000, keep_encodings=True)
    # The one operand axis each shape is, by its exact extents: the weight and its group scales, the
    # activation and its group scales, the output. A layout the container changed fails here loudly
    # instead of cloning an operand by its position.
    axes = {(M, K): (2048, 3072), (M, K // 128): (2048, 3072 // 128),
            (N, K): (1024, 3072), (N, K // 128): (1024, 3072 // 128),
            (M, N): (2048, 1024)}
    for o in d['opd']:
        if not o['shape']:
            continue
        if tuple(o['shape']) not in axes:
            raise SystemExit(f'gemm_nt_q operand of shape {o["shape"]} is not one of q_a\'s axes')
        o['shape'] = list(axes[tuple(o['shape'])])
        if o['has_enc']:
            # The selected planes' extents in their own elements move with the weight: a kernel reads
            # them to lay the planes out, and a recorded extent that does not match the operand is a
            # drop at rerecord ("its plane holds 4718592 elements and the reader wants 3145728").
            o['sel'] = [(p, *axes[(e0, e1)]) for p, e0, e1 in o['sel']]
    d['m'] = 2048
    setp(d, 'M', 2048); setp(d, 'N', 1024); setp(d, 'K', 3072)
    d['band'] = f"{base['band']} q_a M 2048"
    return [d]

def t04(out, src):
    cases = (t04a_sliding(src) + t04b_sel(src) + t04c_verify(src) + t04d_gemm_q(src)
             + t04e_gemm_q_a(src))
    fx = dict(version=6, samples=1024, source='hy4-flash TP2 shapes and c8 verify heights (T-04, D-10)',
              oracle='libref', recorded='2026-11-28', cases=cases)
    save(fx, out); print(len(cases), 'cases ->', out)

if __name__ == '__main__':
    {'raw': raw, 'merge': merge, 'vis': vis, 'wp163': wp163, 'junction': junction, 'append': append, 'dfly': dfly,
     'g3': g3, 't04': t04, 'k41_prefill': k41_prefill}[sys.argv[1]](*sys.argv[2:])
