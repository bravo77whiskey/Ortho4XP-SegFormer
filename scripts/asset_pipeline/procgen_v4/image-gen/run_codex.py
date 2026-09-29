"""Run one ``codex exec`` image session per group, N at a time.

    python image-gen/run_codex.py --pages europe_res_a europe_apt [--jobs 4]
                           [--skip europe_res_a/s3] [--work design_work]

A group is skipped when all its outputs already exist (codex also skips
existing files, so a rerun only fills gaps).  Logs: work/<page>/codex_<group>.log
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")


def run_group(work, page, g):
    wdir = os.path.join(work, page)
    names = sorted({c["name"] for c in g["chunks"]}) if "chunks" in g         else g["inputs"]
    outs = g.get("outputs", names)
    if all(os.path.isfile(os.path.join(wdir, "out", n + ".png"))
           for n in outs):
        return page, g["group"], "done", 0
    cmd = ["codex", "exec", "--skip-git-repo-check", "-s", "workspace-write",
           "-c", 'approval_policy="never"', "-C", wdir]
    for n in names:
        cmd += ["--image", os.path.join("in", n + ".png")]
    t0 = time.time()
    with open(os.path.join(wdir, f"prompt_{g['group']}.md"), "rb") as fin, \
            open(os.path.join(wdir, f"codex_{g['group']}.log"), "wb") as log:
        rc = subprocess.run(cmd, stdin=fin, stdout=log, stderr=subprocess.STDOUT,
                            cwd=wdir, shell=os.name == "nt").returncode
    missing = [n for n in outs
               if not os.path.isfile(os.path.join(wdir, "out", n + ".png"))]
    return page, g["group"], f"rc={rc} missing={missing}", time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None,
                    help="page/group names to run (default: all)")
    ap.add_argument("--work", default=WORK,
                    help="work root (design_work for level-2 designs)")
    args = ap.parse_args(argv)
    if not os.path.isabs(args.work):
        args.work = os.path.join(HERE, args.work)
    todo = []
    for page in args.pages:
        with open(os.path.join(args.work, page, "jobs.json"),
                  encoding="utf-8") as fh:
            for g in json.load(fh)["groups"]:
                name = f"{page}/{g['group']}"
                if name not in args.skip and (args.only is None
                                              or name in args.only):
                    todo.append((page, g))
    with ThreadPoolExecutor(args.jobs) as pool:
        for page, group, status, dt in pool.map(lambda a: run_group(args.work, *a), todo):
            print(f"{page}/{group}: {status} ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
