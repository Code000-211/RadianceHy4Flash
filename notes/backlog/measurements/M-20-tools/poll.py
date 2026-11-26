import json, sys, time, urllib.request
out = open(sys.argv[1], 'a'); port = sys.argv[2] if len(sys.argv) > 2 else '8041'
ok = False; dead = 0
def find(o, k):
    if isinstance(o, dict):
        if k in o: return o[k]
        for v in o.values():
            r = find(v, k)
            if r is not None: return r
    if isinstance(o, list):
        for v in o:
            r = find(v, k)
            if r is not None: return r
while True:
    try:
        o = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{port}/stats', timeout=5).read())
        e = find(o, 'experts') or {}
        v = {t['name']: t['units'] for t in e.get('tiers', [])}
        out.write(json.dumps({'ns': time.monotonic_ns(), 'vram': v.get('vram'), 'pinned': v.get('pinned'),
                              'flex': e.get('flex_bytes'), 'cap': e.get('flex_capacity'), 'routed': e.get('routed'),
                              'kv_used': o.get('kv_bytes_used'), 'kv_total': o.get('kv_bytes_total')}) + '\n'); out.flush()
        ok = True; dead = 0
    except Exception:
        dead += 1
        if (ok and dead > 60) or dead > 1800: break
    time.sleep(0.5)
