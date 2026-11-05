import csv, collections, sys
rows = list(csv.DictReader(open(sys.argv[1])))
k = sorted(((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]) for r in rows))
t_end = k[-1][1]
win = [e for e in k if e[0] > t_end - 4e9]
cyc = sum("_gdn_mtp_update_kernel" in e[2] for e in win) / 48
span = (win[-1][1] - win[0][0]) / 1e6; busy = sum(e[1] - e[0] for e in win) / 1e6
print(f"cycles {cyc:.0f}  per cycle: wall {span/cyc:.2f} ms busy {busy/cyc:.2f} ms kernels {len(win)/cyc:.0f}")
c = collections.Counter(); n = collections.Counter()
for e in win:
    c[e[2][:95]] += e[1] - e[0]; n[e[2][:95]] += 1
for name, d in c.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 14):
    print(f"{d / 1e6 / cyc:7.3f} ms x{n[name] / cyc:6.1f} {name}")
