#!/bin/bash
# three parallel slots of the p7 queue (4 cores); call repeatedly until every item prints its result
cd "$(dirname "$0")"; PY=~/mamba/envs/p2check/bin/python; export BUDGET=65 PAUSE=25
SLOT=A timeout 100 $PY p7_run.py disc:0.001:1.1@1@2050@noceil 2>&1 | grep -E "premium|RESUME|WARN|rror" &
SLOT=B timeout 100 $PY p7_run.py d2@1.5@2050 d2@2@2050 d2@3@2050 d2@1.5@2070 2>&1 | grep -E "premium|RESUME|PAUSE|WARN|rror" &
SLOT=C timeout 100 $PY p7_run.py d2@2@2070 d2@3@2070 2>&1 | grep -E "premium|RESUME|PAUSE|WARN|rror" &
wait
