#!/bin/bash
# runs the p7 queue; call repeatedly until it prints done
cd "$(dirname "$0")"; PY=~/mamba/envs/p2check/bin/python
BUDGET=70 PAUSE=30 timeout 100 $PY p7_run.py disc:0.005:0.95@1@2050 disc:0.005:0.95@1@2070 disc:0.005:0.95@1@2050@noceil disc:0.001:1.1@1@2050 disc:0.001:1.1@1@2070 disc:0.001:1.1@1@2050@noceil d2@1.5@2050 d2@2@2050 d2@3@2050 d2@1.5@2070 d2@2@2070 d2@3@2070 2>&1 | grep -v -i "warning\|out = "
