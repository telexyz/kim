# https://www.webucator.com/article/python-clocks-explained
import time

RECORD_TIMESPENT = False
RECORD_CUDA_TIMESPENT = False
started_at = time.time()
fw_ts, fw_cn = {}, {}
bw_ts, bw_cn = {}, {}

def record_timespent(op, timespent, forward=True):
    ts, cn = (fw_ts, fw_cn) if forward else (
        bw_ts, bw_cn)
    k = op.__class__.__name__
    try:
        ts[k] += timespent
        cn[k] += 1
    except KeyError:
        ts[k] = timespent
        cn[k] = 1

# Other timespents log
other_ts, other_cn = {}, {}
def record_other_timespent(name, ts):
    try: 
        other_ts[name] += ts
        other_cn[name] += 1
    except KeyError:
        other_ts[name] = ts
        other_cn[name] = 1

# CUDA timespents log
cu_ts, cu_cn, cu_caller = {}, {}, {}
def record_cu_timespent(name, caller, caller_class, ts):
    try: 
        cu_ts[name] += ts
        cu_cn[name] += 1
    except KeyError:
        cu_ts[name] = ts
        cu_cn[name] = 1

    if caller is None: return
    kaller = f"{caller}.{caller_class.__name__}"
    try: x = cu_caller[name]
    except KeyError:
        cu_caller[name] = {}
        x = cu_caller[name]
    try: x[kaller] += 1
    except KeyError: x[kaller] = 1

def print_cuda_timespents(ops):
    sec = 1_000_000_000.0 # 10^9 nanoseconds
    total = sum(cu_ts.values())/sec
    if total == 0: return
    print(f"\nCUDA            CALL  x   AVG  = TIME     %\n- - - - - - - - - - - - - - - - - - - - - -")
    for k, v in sorted(cu_ts.items(), key=lambda x: -x[1]):
        if cu_cn[k] == 0: continue
        v /= sec
        print(f"{k:14s} {cu_cn[k]:5d}  {v/cu_cn[k]:.5f}  {v:3.4f}    {round(v*100/total):2d}")
    print(f"- - - - - - - - - - - - - - - - - - - - - -\nTOTAL                          {total:3.4f}  100%\n\n")

    return
    
    stats = {}
    for k, v in sorted(cu_cn.items(), key=lambda x: x[0]):
        print(f"\n{k:15s} {cu_cn[k]:15d}\n- - - - - - - - - - - - - - - -")
        try:
            for c, t in sorted(cu_caller[k].items(), key=lambda x: x[0]):
                print(f"   {c:22s} {t:5d}")
                try: sta = stats[c]
                except KeyError: stats[c] = {}; sta = stats[c]
                sta[k] = t

        except: continue

    print("\n\n")
    for k, sta in sorted(stats.items(), key=lambda x: x[0]):
        try: v = ops[k]
        except: v = 0
        print(f"\n{k:22s} {v:8d}\n- - - - - - - - - - - - - - - -")
        for c, t in sorted(sta.items(), key=lambda x: x[0]):
            if c == "_": continue
            print(f"   {c:22s} {t:5d}")

def print_timespents():
    if started_at is None:
        return
    total = time.time() - started_at
    fw = sum(fw_ts.values())
    bw = sum(bw_ts.values())
    ot = sum(other_ts.values())
    unknown = total - fw - bw - ot
    ops = {}

    print(f"\ncompute()     CALL  x   AVG  = TIME     %\n- - - - - - - - - - - - - - - - - - - - -")
    for k, v in sorted(fw_ts.items(), key=lambda x: -x[1]):
        ops["compute." + k] = fw_cn[k]
        print(f"{k:12s} {fw_cn[k]:5d}  {v/fw_cn[k]:.5f}  {v:3.4f}    {round(v*100/total):2d}")
    print(f"- - - - - - - - - - - - - - - - - - - - -\nFORWARD                      {fw:3.4f}  {round(fw*100/total):3d}%")

    print(f"\ngradient()    CALL  x   AVG  = TIME     %\n- - - - - - - - - - - - - - - - - - - - -")
    for k, v in sorted(bw_ts.items(), key=lambda x: -x[1]):
        ops["gradient." + k] = bw_cn[k]
        print(f"{k:12s} {bw_cn[k]:5d}  {v/bw_cn[k]:.5f}  {v:3.4f}    {round(v*100/total):2d}")
    print(f"- - - - - - - - - - - - - - - - - - - - -\nBACKWARD                     {bw:3.4f}  {round(bw*100/total):3d}%")

    print(f"\n                    CALL  x   AVG  = TIME     %\n- - - - - - - - - - - - - - - - - - - - - - - -")
    for k, v in sorted(other_ts.items(), key=lambda x: -x[1]):
        print(f"{k:18s} {other_cn[k]:5d}  {v/other_cn[k]:.5f}  {v:3.4f}    {round(v*100/total):2d}")
    print(f"- - - - - - - - - - - - - - - - - - - - - - - -")
    print(f"OTHERS                             {ot:3.4f}  {round(ot*100/total):3d}%")

    print(f"\nTotal    {total:.4f}s 100%\n- - - - - - - - - - -")
    print(f"Forward  {fw:.4f}s {round(fw*100/total):3d}%")
    print(f"Backward {bw:.4f}s {round(bw*100/total):3d}%")
    print(f"Others   {ot:.4f}s {round(ot*100/total):3d}%")
    print(f"Unknown  {unknown:.4f}s {round(unknown*100/total):3d}%")
    print_cuda_timespents(ops)
