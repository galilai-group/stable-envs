"""Turn the measurement JSONs into the (ratio, batch) tables with crossovers.

Usage: python make_tables.py
"""
import json
import os

D = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(D, name)
    return json.load(open(p)) if os.path.exists(p) else None


def crossover(batches, ratios):
    """Smallest batch whose ratio vs the 1-core baseline is >= 1."""
    for b in batches:
        r = ratios.get(b)
        if isinstance(r, float) and r >= 1.0:
            return b
    return None


def cule_tables():
    r = load("cule_bench.json")
    if r is None:
        print("cule_bench.json: MISSING")
        return
    print("### CuLE throughput  (%s, %s, %d cpus)" % (r["node"], r["gpu"], r["cpus"]))
    for cname, c in r["configs"].items():
        cfg = c["cfg"]
        print("\n-- config %s: frameskip=%d color=%s rescale=%s" %
              (cname, cfg["frameskip"], cfg["color"], cfg["rescale"]))
        for g in r["games"]:
            base = c["ale_serial"].get(g)
            if base is None:
                continue
            print("  %-16s ale-py 1 core: %10.1f steps/s = %10.1f frames/s"
                  % (g, base, base * cfg["frameskip"]))
            for key in ("cule_gpu", "cule_cpu"):
                row = c.get(key, {}).get(g, {})
                ratios = {}
                cells = []
                for b in r["batches"]:
                    v = row.get(str(b), row.get(b))
                    if isinstance(v, dict) and "steps_per_s" in v:
                        ratios[b] = v["steps_per_s"] / base
                        cells.append("n=%-5d %10.0f f/s  x%7.2f" %
                                     (b, v["frames_per_s"], ratios[b]))
                    elif isinstance(v, dict):
                        cells.append("n=%-5d ERROR %s" % (b, str(v.get("error"))[:70]))
                if cells:
                    print("    %s  crossover=%s" % (key, crossover(r["batches"], ratios)))
                    for cell in cells:
                        print("      " + cell)


def envpool_tables():
    r = load("envpool_bench.json")
    if r is None:
        print("envpool_bench.json: MISSING")
        return
    print("\n### envpool / gym vector throughput (%s, %d cpus, frameskip %d)"
          % (r["node"], r["cpus"], r["frameskip"]))
    print("envpool=%s ale_py=%s gymnasium=%s" %
          (r.get("envpool_version"), r.get("ale_py_version"), r.get("gymnasium_version")))
    for g, base in r["ale_serial"].items():
        print("\n  %-16s ale-py 1 core: %10.1f steps/s = %10.1f frames/s"
              % (g, base, base * r["frameskip"]))
        for key in ("envpool", "gym_async", "gym_sync"):
            row = r.get(key, {}).get(g, {})
            if not row:
                continue
            batches = sorted(int(k) for k in row)
            ratios = {}
            print("    %s" % key)
            for b in batches:
                v = row.get(str(b), row.get(b))
                if isinstance(v, (int, float)):
                    ratios[b] = v / base
                    print("      n=%-5d %10.0f steps/s = %10.0f f/s  x%7.2f"
                          % (b, v, v * r["frameskip"], ratios[b]))
                else:
                    print("      n=%-5d %s" % (b, str(v)[:90]))
            print("      crossover(>=1x one core)=%s" % crossover(batches, ratios))


def parity_table():
    r = load("parity_cule.json")
    if r is None:
        print("\nparity_cule.json: MISSING")
        return
    print("\n### CuLE vs ale-py bit parity (%d steps/ROM, %s)" % (r["steps"], r["gpu"]))
    hdr = ("%-18s %8s %8s %9s %8s %8s %9s %7s %8s" %
           ("rom", "actsets", "frm=", "px=", "maxdiff", "ram=", "1stdiv", "shift", "noopshf"))
    print(hdr)
    for rom, v in r["roms"].items():
        if "error" in v:
            print("%-18s ERROR %s" % (rom, v["error"][:90]))
            continue
        a = v["rand"]["cule_gpu_vs_ale"]
        nz = v["noop"]["cule_gpu_vs_ale"]
        print("%-18s %8s %8.4f %9.5f %8d %8.4f %9s %7s %8.4f" %
              (rom, v["action_sets_match"], a["frames_exact_frac"],
               a["frac_pixels_exact"], a["max_abs_diff"], a["ram_exact_frac"],
               a["first_frame_divergence"], nz.get("best_shift"),
               nz.get("frames_exact_frac_at_best_shift", float("nan"))))
    print("\nsummary:", json.dumps(r.get("summary", {}), indent=1))


if __name__ == "__main__":
    cule_tables()
    envpool_tables()
    parity_table()
