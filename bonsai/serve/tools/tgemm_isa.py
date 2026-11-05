"""Static ISA report for the ternary WMMA kernels (gfx1201), no GPU needed.

Compiles ternary_kernels.hip to assembly with hipcc and prints, per kernel,
VGPR/SGPR/LDS usage, occupancy and the instruction mix of the hottest loop.
"""
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

SRC = (Path(__file__).resolve().parents[2]
       / "gguf_plugin/vllm_gguf_plugin/csrc/ternary/ternary_kernels.hip")
VGPR_PER_SIMD = 1536  # gfx12 wave32
LDS_PER_WGP = 131072
VGPR_GRANULE = 8
UNPACK = re.compile(r"v_(and|or|xor|lshr|lshl|ashr|perm|bfe|bfi|sub_nc|add_nc|"
                    r"and_or|or3|xor3|lshl_or|lshrrev|lshlrev|not)\w*")
CLASSES = [
    ("wmma", re.compile(r"v_wmma")),
    ("gload", re.compile(r"(global|buffer|flat)_load")),
    ("dsread", re.compile(r"ds_(read|load)")),
    ("dswrite", re.compile(r"ds_(write|store)")),
    ("waitcnt", re.compile(r"s_wait|s_waitcnt")),
    ("unpackALU", UNPACK),
    ("otherVALU", re.compile(r"v_")),
]


def compile_asm(out_dir):
    """Compile the .hip source to gfx1201 assembly and return its text."""
    import torch.utils.cpp_extension as ce
    csrc = SRC.parents[1]
    out = Path(out_dir) / "k.s"
    cmd = ["hipcc", "-x", "hip", "--offload-arch=gfx1201", "--cuda-device-only",
           "-S", "-O3", "-std=c++17", "-DUSE_ROCM", "-DUSE_CUDA", f"-I{csrc}"]
    cmd += [f"-I{p}" for p in ce.include_paths()] + [str(SRC), "-o", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out.read_text()


def parse_resources(asm):
    """Return {kernel: dict(vgpr, sgpr, lds, wg)} from .amdhsa_kernel blocks."""
    res = {}
    for m in re.finditer(r"\.amdhsa_kernel (\S+)(.*?)\.end_amdhsa_kernel",
                         asm, re.S):
        body = m.group(2)
        get = lambda k: int(re.search(rf"\.{k} (\d+)", body).group(1))
        res[m.group(1)] = dict(vgpr=get("amdhsa_next_free_vgpr"),
                               sgpr=get("amdhsa_next_free_sgpr"),
                               lds=get("amdhsa_group_segment_fixed_size"), wg=0)
    for wg, name in re.findall(
            r"\.max_flat_workgroup_size:\s*(\d+).*?\.name:\s*(\S+)", asm, re.S):
        if name in res:
            res[name]["wg"] = int(wg)
    return res


def occupancy(vgpr, lds, wg):
    """Return waves per SIMD limited by VGPRs and LDS (wave32, 4 SIMD/WGP)."""
    alloc = max(VGPR_GRANULE, -(-vgpr // VGPR_GRANULE) * VGPR_GRANULE)
    waves = min(16, VGPR_PER_SIMD // alloc)
    if lds and wg:
        wgs = LDS_PER_WGP // lds
        waves = min(waves, wgs * max(1, wg // 32) // 4)
    return waves


def kernel_lines(asm, name):
    """Return the instruction/label lines of one kernel body."""
    m = re.search(rf"^{re.escape(name)}:.*?^\.Lfunc_end\d+:", asm, re.S | re.M)
    lines = [l.split(";")[0].strip() for l in m.group(0).splitlines()]
    return [l for l in lines if l and not l.startswith(".")
            or re.match(r"\.LBB\S+:", l)]


def hottest_loop(lines):
    """Return the lines of the backward-branch loop holding most WMMAs."""
    labels = {l[:-1]: i for i, l in enumerate(lines) if l.endswith(":")}
    best = []
    for i, l in enumerate(lines):
        t = l.split()[-1] if l.startswith(("s_cbranch", "s_branch")) else None
        if t in labels and labels[t] < i:
            body = lines[labels[t]:i + 1]
            if sum("v_wmma" in b for b in body) > sum("v_wmma" in b for b in best):
                best = body
    return best


def classify(line):
    """Return the mix class of one instruction line."""
    for name, rx in CLASSES:
        if rx.match(line):
            return name
    return "other"


def mix(lines):
    """Return a Counter of instruction classes (labels skipped)."""
    return Counter(classify(l) for l in lines if not l.endswith(":"))


def report(asm):
    """Return the printable table for every kernel in the assembly."""
    out = []
    for name, r in parse_resources(asm).items():
        loop = hottest_loop(kernel_lines(asm, name))
        c = mix(loop)
        w = c["wmma"]
        ratio = f"{c['unpackALU'] / w:.1f}" if w else "-"
        out.append(f"{name[:60]}\n  VGPR {r['vgpr']} SGPR {r['sgpr']} LDS {r['lds']}B "
                   f"wg {r['wg']} occ {occupancy(r['vgpr'], r['lds'], r['wg'])}"
                   f" waves/SIMD | loop {len(loop)} instr, unpack:WMMA {ratio}\n  "
                   + " ".join(f"{k}={v}" for k, v in sorted(c.items())))
    return "\n".join(out)


def main():
    """Compile the kernels and print the report."""
    with tempfile.TemporaryDirectory() as d:
        print(report(compile_asm(d)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
