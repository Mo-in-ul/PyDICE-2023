#!/bin/bash
cd "$(dirname "$0")"; PY=~/mamba/envs/p2check/bin/python
BUDGET=70 timeout 100 $PY p7_tree.py "$@" 2>&1 | grep -v -i "warning\|out = " | tail -6
