#!/usr/bin/env python3
"""Reproduce the paper end to end.

Step 1 runs the full experiment (run_experiment.py), which writes every
result CSV into ./results/. Step 2 builds the three figures, which read from
./results/ and write fig_mechanism.{pdf,png}, fig_wedge.{pdf,png}, and
fig_scatter.{pdf,png} into this folder.

Usage:   python run_all.py
"""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))

def run(rel):
    print(f"\n===== {rel} =====", flush=True)
    subprocess.run([sys.executable, rel], cwd=HERE, check=True)

if __name__ == "__main__":
    run("run_experiment.py")                 # writes ./results/*.csv
    run(os.path.join("figs", "fig_mechanism.py"))
    run(os.path.join("figs", "fig_wedge.py"))
    run(os.path.join("figs", "fig_scatter.py"))
    print("\nDone. CSVs in ./results/ ; figures fig_*.pdf in this folder.")
