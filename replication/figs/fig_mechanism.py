#!/usr/bin/env python3
"""fig_mechanism -- schematic of the SCC decomposition tau^c = p^gross(mu_bar) + w_nu.
Print-sized: small native canvas + large fonts (matches fig_scatter). Flip USETEX for CM."""
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
USETEX = False
mpl.rcParams.update({"text.usetex":USETEX,"font.family":"serif",
    "font.serif":["cmr10","CMU Serif","Computer Modern Roman","DejaVu Serif"],
    "mathtext.fontset":"cm","axes.unicode_minus":False,
    "font.size":11,"axes.linewidth":0.9,"xtick.labelsize":11,"ytick.labelsize":10.5})
BLACK="0.0"; GRAY_BASE="0.80"; GRAY_WEDGE="0.55"
TITLE=11.5; LAB=11; ANN=10; EQ=9.5

fig,(axA,axB)=plt.subplots(1,2,figsize=(6.9,3.25))

# ================= PANEL (a) =================
SCC=100.0; mustar=0.42; mubar=0.24; beta=1.6
mu=np.linspace(0.0,0.52,400)
pgross=SCC*(mu/mustar)**beta
p_at_bar=SCC*(mubar/mustar)**beta
mid=0.5*(SCC+p_at_bar)

axA.plot(mu,pgross,color=BLACK,lw=1.8,zorder=3)
axA.axhline(SCC,color=BLACK,lw=1.3,ls=(0,(6,4)),zorder=2)
axA.plot([mubar,mubar],[0,SCC],color="0.45",lw=1.1,ls=(0,(1,2)),zorder=1)
axA.plot([mustar,mustar],[0,SCC],color="0.45",lw=1.1,ls=(0,(1,2)),zorder=1)
axA.plot([mubar],[p_at_bar],marker="o",ms=6,mfc="white",mec=BLACK,mew=1.3,zorder=4)
axA.plot([mustar],[SCC],marker="o",ms=6,mfc=BLACK,mec=BLACK,zorder=4)

# wedge double-arrow at the ceiling
axA.annotate("",xy=(mubar,SCC),xytext=(mubar,p_at_bar),
             arrowprops=dict(arrowstyle="<->",color=BLACK,lw=1.5),zorder=5)
# deferral-wedge label to the LEFT of the arrow (above the curve), with a short leader
axA.annotate(r"deferral wedge""\n"r"$w_\nu=\dfrac{\nu}{\sigma\,Q_{\mathrm{gross}}\,\Phi}$",
             xy=(mubar,mid), xytext=(0.205,mid), ha="right", va="center", fontsize=EQ,
             arrowprops=dict(arrowstyle="-",color="0.5",lw=0.8))

# p^gross(mu) label: moved left, above the curve
axA.text(0.435,131,r"$p^{\mathrm{gross}}(\mu)$",ha="right",va="bottom",fontsize=ANN)
# marginal-benefit label
axA.text(-0.03,SCC+3,r"$\lambda_E/\Phi=\mathrm{SCC}$ (marginal benefit)",
         ha="left",va="bottom",fontsize=ANN-0.5)

axA.set_xlim(-0.06,0.52); axA.set_ylim(0,142)
axA.set_xticks([mubar,mustar]); axA.set_xticklabels([r"$\bar{\mu}$",r"$\mu^{*}$"])
axA.set_yticks([p_at_bar,SCC]); axA.set_yticklabels([r"$p^{\mathrm{gross}}(\bar{\mu})$",r"$\tau^{c}=\mathrm{SCC}$"])
axA.set_xlabel(r"control rate $\mu$",fontsize=LAB)
axA.set_ylabel(r"$\$$/tCO$_2$",fontsize=LAB)
axA.set_title(r"(a) why the wedge arises",fontsize=TITLE,loc="left")
for s in ("top","right"): axA.spines[s].set_visible(False)

# ================= PANEL (b) =================
p_interior=62.0; p_bind=64.0; w_bind=89.0; x=[0,1.35]; w=0.46
axB.bar(x[0],p_interior,width=w,color=GRAY_BASE,edgecolor=BLACK,lw=1.0,zorder=3)
axB.bar(x[1],p_bind,width=w,color=GRAY_BASE,edgecolor=BLACK,lw=1.0,zorder=3,
        label=r"$p^{\mathrm{gross}}(\bar{\mu})$")
axB.bar(x[1],w_bind,width=w,bottom=p_bind,color=GRAY_WEDGE,edgecolor=BLACK,lw=1.0,
        hatch="////",zorder=3,label=r"deferral wedge $w_\nu$")
tot=p_bind+w_bind
axB.annotate("",xy=(x[1]+0.33,0),xytext=(x[1]+0.33,tot),arrowprops=dict(arrowstyle="<->",color="0.5",lw=1.0))
axB.text(x[1]+0.40,tot/2,r"$\tau^{c}$ (reported SCC)",rotation=90,va="center",ha="left",fontsize=ANN-1,color="0.3")
axB.text(x[0],p_interior+4,r"$\tau^{c}=p^{\mathrm{gross}}$",ha="center",va="bottom",fontsize=ANN-0.5)
axB.text(x[1],p_bind/2,r"$p^{\mathrm{gross}}(\bar{\mu})$",ha="center",va="center",fontsize=ANN-1,
         bbox=dict(boxstyle="round,pad=0.18",fc="white",ec="none"))
axB.text(x[1],p_bind+w_bind/2,r"$w_\nu$",ha="center",va="center",fontsize=ANN,
         bbox=dict(boxstyle="round,pad=0.22",fc="white",ec="none"))
axB.set_xticks(x); axB.set_xticklabels(["interior\nspecification","binding\nspecification"])
axB.set_xlim(-0.55,2.2); axB.set_ylim(0,190)
axB.set_ylabel(r"$\$$/tCO$_2$",fontsize=LAB)
axB.set_title(r"(b) what the reported SCC contains",fontsize=TITLE,loc="left")
axB.legend(frameon=False,fontsize=ANN-0.5,loc="upper left",handlelength=1.4,borderpad=0.2)
for s in ("top","right"): axB.spines[s].set_visible(False)

fig.tight_layout(w_pad=2.0)
fig.savefig("fig_mechanism.png",dpi=210,bbox_inches="tight")
fig.savefig("fig_mechanism.pdf",bbox_inches="tight")
print("done")
