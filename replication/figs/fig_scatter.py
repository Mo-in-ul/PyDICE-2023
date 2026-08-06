#!/usr/bin/env python3
"""
fig_scatter (full) -- print-sized: small native canvas + large fonts so text stays
legible after \includegraphics[width=\textwidth] scales it down in the manuscript.
(a) damage specs: SCC vs control rate, binding stack on ceiling.
(b) discount sweep: rho on x, SCC and p^gross on shared log axis.
Grayscale. Flip USETEX for Computer Modern. Values from Tables main_results / reference_sweep.
"""
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
USETEX=False
mpl.rcParams.update({"text.usetex":USETEX,"font.family":"serif",
    "font.serif":["cmr10","CMU Serif","Computer Modern Roman","DejaVu Serif"],
    "mathtext.fontset":"cm","axes.unicode_minus":False,
    "font.size":11,"axes.linewidth":0.9,
    "xtick.labelsize":10,"ytick.labelsize":10})
BLACK="0.0"; MUBAR=0.24
TITLE=11.5; LAB=11; ANN=10; LEG=10

fig,(axA,axB)=plt.subplots(1,2,figsize=(6.3,3.1))

# ---------- panel (a) ----------
specs=[("Dietz-Stern",0.1960,45.20,"interior"),
       ("Nordhaus DICE-2016",0.2043,48.45,"interior"),
       ("Weitzman (2012)",0.2178,53.91,"interior"),
       ("Nordhaus DICE-2023",0.2342,60.94,"near"),
       ("Howard-Sterner Central",0.2400,86.40,"binding"),
       ("Kahn et al.",0.2400,91.57,"binding"),
       ("Howard-Sterner High",0.2400,157.52,"binding")]
fillc={"binding":BLACK,"near":"0.55","interior":"white"}
axA.axvline(MUBAR,color="0.45",lw=1.1,ls=(0,(1,2)),zorder=1)
for _,m,scc,st in specs:
    axA.plot(m,scc,marker="o",ms=7,mew=1.2,ls="none",zorder=3,mfc=fillc[st],mec=BLACK)
axA.text(MUBAR-0.004,178,r"$\bar{\mu}(2030)=0.24$",rotation=90,va="top",ha="right",fontsize=ANN,color="0.35")
axA.set_xlim(0.15,0.27); axA.set_ylim(0,190)
axA.set_xlabel(r"2030 control rate $\mu$",fontsize=LAB)
axA.set_ylabel(r"2030 SCC ($\$$/tCO$_2$)",fontsize=LAB)
axA.set_title(r"(a) damage specifications",fontsize=TITLE,loc="left")
for s in ("top","right"): axA.spines[s].set_visible(False)
legA=[Line2D([],[],marker="o",ls="none",ms=7,mfc=BLACK,mec=BLACK,label="binding"),
      Line2D([],[],marker="o",ls="none",ms=7,mfc="0.55",mec=BLACK,label="near-binding"),
      Line2D([],[],marker="o",ls="none",ms=7,mfc="white",mec=BLACK,label="interior")]
axA.legend(handles=legA,frameon=False,fontsize=LEG,loc="upper left",handletextpad=0.3,borderpad=0.2)

# ---------- panel (b) ----------
rho=np.array([1,2,3,4,5.]); scc=np.array([885.,213.,98.,57.,38.])
pgr=np.array([64.12,64.12,64.12,59.23,40.31]); binding=np.array([1,1,1,0,0],bool)
axB.plot(rho,pgr,color="0.45",lw=1.5,ls=(0,(5,2)),marker="s",ms=5.5,mfc="0.45",mec=BLACK,zorder=3,
         label=r"$p^{\mathrm{gross}}$")
axB.plot(rho,scc,color=BLACK,lw=1.7,zorder=3)
for x,y,b in zip(rho,scc,binding):
    axB.plot(x,y,marker="o",ms=7,mew=1.2,ls="none",zorder=4,mfc=(BLACK if b else "white"),mec=BLACK)
axB.plot([],[],marker="o",ms=7,ls="none",mfc=BLACK,mec=BLACK,label="SCC (binding)")
axB.plot([],[],marker="o",ms=7,ls="none",mfc="white",mec=BLACK,label="SCC (interior)")
axB.annotate(r"$p^{\mathrm{gross}}=\$64.12$, $\mu=0.24$",
             xy=(3.0,64.12),xytext=(2.45,48),fontsize=ANN-2,color="0.25",ha="center",va="center",
             arrowprops=dict(arrowstyle="->",color="0.5",lw=0.8))
axB.annotate("",xy=(0.86,885),xytext=(0.86,65),arrowprops=dict(arrowstyle="<->",color=BLACK,lw=1.1))
axB.text(1.55,160,r"SCC $\times 9$",fontsize=ANN,va="center",ha="center")
axB.set_yscale("log")
axB.set_xticks(rho); axB.set_xticklabels([f"{int(r)}%" for r in rho])
axB.set_xlabel(r"pure rate of time preference $\rho$",fontsize=LAB)
axB.set_ylabel(r"2030 value ($\$$/tCO$_2$, log)",fontsize=LAB)
axB.set_xlim(0.7,5.3); axB.set_ylim(30,1300); axB.invert_xaxis()
for s in ("top","right"): axB.spines[s].set_visible(False)
axB.legend(frameon=False,fontsize=LEG-0.5,loc="upper left",handlelength=1.6,borderpad=0.3,labelspacing=0.35)
axB.set_title(r"(b) discount sweep, damage fixed",fontsize=TITLE,loc="left")

fig.tight_layout(w_pad=1.6)
fig.savefig("fig_scatter.png",dpi=220,bbox_inches="tight")
fig.savefig("fig_scatter.pdf",bbox_inches="tight")
print("done; native size 6.3 x 3.1 in, base font 11pt")
