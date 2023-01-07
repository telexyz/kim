import time

RECORD_TIMESPENT = False
started_at = None
fw_ts, fw_cn = {}, {}
bw_ts, bw_cn = {}, {}

def record_timespent(op, timespent, forward=True):
    global started_at
    if started_at is None:
        started_at = time.time()
    ts, cn = (fw_ts, fw_cn) if forward else (
        bw_ts, bw_cn)
    k = op.__class__.__name__
    try:
        ts[k] += timespent
        cn[k] += 1
    except KeyError:
        ts[k] = timespent
        cn[k] = 1


# CUDA timespents log
ts, call = {}, {}
def print_cuda_timespents():
    total = sum(ts.values())
    if total == 0:
        return
    print(f"\nCUDA            CALL  x   AVG  = TIME   %\n- - - - - - - - - - - - - - - - - - - - -")
    for k, v in sorted(ts.items(), key=lambda x: -x[1]):
        if call[k] == 0: continue
        print(f"{k:14s} {call[k]:5d}  {v/call[k]:.5f}  {v:3.4f}  {round(v*100/total):2d}")
    print(f"- - - - - - - - - - - - - - - - - - - - -\nTOTAL                        {total:3.4f}  100%")

def print_timespents():
    if started_at is None:
        return
    total = time.time() - started_at
    fw = sum(fw_ts.values())
    bw = sum(bw_ts.values())
    fwbw = fw + bw
    others = total - fwbw

    print(f"\nFORWARD       CALL  x   AVG  = TIME   %\n- - - - - - - - - - - - - - - - - - - -")
    for k, v in sorted(fw_ts.items(), key=lambda x: -x[1]):
        print(
            f"{k:12s} {fw_cn[k]:5d}  {v/fw_cn[k]:.5f}  {v:3.4f}  {round(v*100/total):2d}")

    print(f"\nBACKWARD      CALL  x   AVG  = TIME   %\n- - - - - - - - - - - - - - - - - - - -")
    for k, v in sorted(bw_ts.items(), key=lambda x: -x[1]):
        print(
            f"{k:12s} {bw_cn[k]:5d}  {v/bw_cn[k]:.5f}  {v:3.4f}  {round(v*100/total):2d}")
    
    print_cuda_timespents()

    print(f"\nTotal    {total:.4f}s 100%\n- - - - - - - - - - -")
    print(f"Forward  {fw:.4f}s {round(fw*100/total):3d}%")
    print(f"Backward {bw:.4f}s {round(bw*100/total):3d}%")
    print(f"Others   {others:.4f}s {round(others*100/total):3d}%")
