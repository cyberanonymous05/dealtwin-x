"""Draw paper/figures/fig0_pipeline.png (pipeline overview with the corpus counts of the paper)."""
import sys, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"]="DejaVu Serif"
boxes=[("1.  Discovery and M&A gate","#2b6cb0","#e3ebf5","SEC EDGAR full-text search, 2013-2025: 3,841 primary DEFM14A/14C and\nSC 14D9 filings (3,336 targets)  ->  gate (SPACs, acquirer-side proxies,\nno merger agreement)  ->  2,416 deals: 1,947 merger proxies +\n469 cash tender offers, 65 two-digit SIC groups"),
("2.  Point-in-time enrichment","#2b6cb0","#e3ebf5","deal terms (context-filtered regex)  .  XBRL facts filed before the deal\n(IFRS / TTM / bank fallbacks)  .  FRED macro  .  acquirer type / listing /\ncross-border  .  completion status  .  legal-term flags  .\nclause passages (semantic selection)  .  10-K business description"),
("3.  DealDNA embedding (seven blocks)","#1e8a5a","#e0efe8","industry  |  financial (9)  |  structure (6)  |  acquirer (3)  |  macro (6)  |\nlegal clauses (FinBERT / LEGAL-BERT / MPNet)  |  business description (MPNet)"),
("4.  Causal-temporal evaluation","#c4572a","#f5e8e3","a query retrieves only deals filed on or before it  .  5 proxy relevance constructs  .\n7 retrievers (naive, SIC back-off, size bins, text peers, SIC2+text, linear, LambdaMART)  .\nfold-local preprocessing  .  cluster bootstrap + Holm over all 504 comparisons")]
fig,ax=plt.subplots(figsize=(10,10.3),dpi=300); ax.set_xlim(0,10); ax.set_ylim(0,10.2); ax.axis("off")
h=1.45; gap=0.62; y=8.1-0.2-h+0.05
hs=[2.2,2.2,1.55,1.9]; y=10.0
for (t,ec,fc,body),hh in zip(boxes,hs):
    y0=y-hh
    ax.add_patch(FancyBboxPatch((0.4,y0),9.2,hh,boxstyle="round,pad=0,rounding_size=0.12",lw=2.2,ec=ec,fc="white"))
    ax.add_patch(plt.Rectangle((0.45,y-0.6),9.1,0.55,fc=fc,ec="none"))
    ax.text(0.73,y-0.33,t,color=ec,fontsize=15,fontweight="bold",va="center")
    ax.text(0.73,y-0.78,body,fontsize=9.6,va="top",linespacing=1.55)
    if y0>0.5: ax.annotate("",xy=(5,y0-gap+0.12),xytext=(5,y0),arrowprops=dict(arrowstyle="-|>",lw=2,color="#555",mutation_scale=18))
    y=y0-gap
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "paper/figures/fig0_pipeline.png",bbox_inches="tight",facecolor="white")
