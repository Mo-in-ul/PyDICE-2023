"""Build standalone GAMS files for the NEOS cross-check of the eta letter.

Starts from the published DICE2023-b-4-3-10.gms, inlines its two model include files
(Nonco2, FAIR), keeps only the optimal run (the three solves of def-opt-b-4-3-10.gms and
its SCC formula), and changes nothing in the model except elasmu. solprint is turned off
and the scenario put files are dropped, since NEOS returns only the listing file.
Usage: python make_neos.py PATH_TO/GAMS-DICE2023-b-4-3-10
(the published DICE-2023 GAMS folder from the DICE website; writes dice2023_eta<phi>.gms next to this script)
"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1]
PHIS = ('0.95', '1.30', '1.50', '1.75')

def read(*p):
    return open(os.path.join(SRC, *p), encoding='latin-1').read().replace('\r\n', '\n')

main = read('DICE2023-b-4-3-10.gms')
head = main.split('$include Include\\Putlong-4-3-10.gms')[0]
head = head.replace('$include Include\\Nonco2-b-4-3-1.gms', read('Include', 'Nonco2-b-4-3-1.gms'))
head = head.replace('$include Include\\FAIR-beta-4-3-1.gms', read('Include', 'FAIR-beta-4-3-1.gms'))
assert '$include' not in head
head = head.replace('option solprint = on', 'option solprint = off')
assert head.count('/ 0.95 /') == 1

tail = read('Include', 'def-opt-b-4-3-10.gms') + r'''
* ---- Output for the eta-letter cross-check (added; model unchanged) ----
scalar peakT, peakyear, scc2025, scc2050;
peakT = smax(t, tatm.l(t));
peakyear = smax(t$(tatm.l(t) = peakT), 2015 + 5*t.val);
scc2025 = scc('2'); scc2050 = scc('7');
option decimals = 8;
display elasmu, CO2.modelstat, CO2.solvestat, utility.l, scc2025, scc2050, peakT, peakyear;
display scc, tatm.l, miu.l, s.l, cpc.l;
'''
for phi in PHIS:
    txt = head.replace('/ 0.95 /', '/ %s /' % phi) + tail
    out = os.path.join(HERE, 'dice2023_eta%s.gms' % phi.replace('.', 'p'))
    open(out, 'w').write(txt)
    print(out, len(txt.splitlines()), 'lines')
