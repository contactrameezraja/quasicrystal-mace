import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':11})
d=np.load('ph_w265_band222_modes.npz',allow_pickle=True)
f=np.sort(d['frequencies'][0]); f=f[(f>0)&(f<20)]
gaps=[(f[i],f[i+1]) for i in range(len(f)-1) if f[i+1]-f[i]>0.3]
fig,ax=plt.subplots(figsize=(7.5,3.4))
for lo,hi in gaps: ax.axvspan(lo,hi,color='0.88',lw=0)
ax.vlines(f,0,1,color='k',lw=0.6)
for lo,hi in gaps:
    c=(lo+hi)/2; ax.plot(c,1.06,'v',color='k',ms=5,clip_on=False)
    ax.text(c,1.10,f'{c:.2f}',ha='center',va='bottom',fontsize=8)
ax.set_xlim(0,20); ax.set_ylim(0,1); ax.set_yticks([])
ax.set_xlabel(r'$E$ (meV)')
ax.text(0.3,0.9,r'W-phase, stacking zone boundary $\mathbf{q}=(-0.5,0.5,0)$',fontsize=9,va='top')
ax.text(0.3,0.78,f'{len(f)} modes below 20 meV, {len(gaps)} gaps wider than 0.3 meV',fontsize=9,va='top')
plt.tight_layout(); plt.savefig('fig_stacking_ladder.png',dpi=300)
print('centres:',[round((a+b)/2,3) for a,b in gaps])
