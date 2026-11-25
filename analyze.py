"""Aggregate generated evaluations and draw the experiment figures."""
import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from run import BASELINES, CP_VARIANTS, SWEEPS

METRICS = ['AppTimeout','TaskTimeout','D_all','E_avg']
DAGS = ['default','chain','wide']
DEFAULTS = dict(guide_ce=.1, timeout_credit=-7, energy_weight=.05, delay_weight=.1,
                shield_high=1.1, shield_normal=1.15, cp_coefficient=.45, overflow_multiplier=1.25)


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(order)-np.arange(len(order)))*pvalues[order]))
    return adjusted


def paired_statistics(differences):
    d = np.asarray(differences, dtype=float)
    if len(d) < 2 or not np.isfinite(d).all():
        raise ValueError('A paired comparison requires at least two finite matched blocks')
    mean = float(d.mean())
    se = float(d.std(ddof=1)/np.sqrt(len(d)))
    half = float(stats.t.ppf(.975,len(d)-1)*se)
    p = float(stats.ttest_1samp(d,0).pvalue) if se > 1e-12 else float(abs(mean)<1e-12)
    signs = np.array(list(itertools.product([-1,1],repeat=len(d))))
    exact = float(np.mean(np.abs((signs*d).mean(axis=1)) >= abs(mean)-1e-12))
    return dict(blocks=len(d),difference_pp=mean,ci_low=mean-half,ci_high=mean+half,
                paired_t_p=p,exact_signflip_p=exact,block_differences=';'.join(f'{v:.12g}' for v in d))


def load_traces(root, filename='heldout.csv'):
    frames=[]
    for path in sorted(root.rglob(filename)):
        df=pd.read_csv(path)
        df['run']=str(path.parent.relative_to(root))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f'No {filename} under {root}')
    df=pd.concat(frames,ignore_index=True)
    if not np.isfinite(df[METRICS].to_numpy()).all():
        raise ValueError('Non-finite evaluation metric')
    if ((df[['AppTimeout','TaskTimeout']]<0)|(df[['AppTimeout','TaskTimeout']]>100)).any().any():
        raise ValueError('Timeout rates must be percentages in [0,100]')
    keys=['run','dag','method','environment_seed','checkpoint','arrival_seed']
    if df.duplicated(keys).any():
        raise ValueError('Duplicate trace records')
    return df


def block_means(traces, partial=False):
    keys=['dag','method','checkpoint','environment_seed']
    if not partial:
        for _,g in traces.groupby(keys):
            if set(g.arrival_seed)!={21001,21002,21003} or len(g)!=3:
                raise ValueError('Each environment requires exactly three independent arrival traces')
        for _,g in traces.groupby(keys[:-1]):
            if set(g.environment_seed)!={1,3,5,7,42}:
                raise ValueError('Each comparison requires environment seeds 1,3,5,7,42')
    return traces.groupby(keys,as_index=False)[METRICS].mean()


def compare(args):
    traces=load_traces(args.runs)
    blocks=block_means(traces,args.allow_partial)
    blocks.to_csv(args.output/'environment_blocks.csv',index=False)
    summary=blocks.groupby(['dag','method','checkpoint'])[METRICS].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(args.output/'summary.csv',index=False)
    contrasts=[]
    for (dag,checkpoint),group in blocks.groupby(['dag','checkpoint']):
        full=group[group.method=='CPGAPPO'].set_index('environment_seed').AppTimeout
        for method in CP_VARIANTS[1:]+BASELINES:
            other=blocks[(blocks.dag==dag)&(blocks.method==method)&(blocks.checkpoint.isin([checkpoint,'none']))].set_index('environment_seed').AppTimeout
            common=full.index.intersection(other.index).sort_values()
            if len(common)<2:
                continue
            if not args.allow_partial and len(common)!=5:
                raise ValueError(f'Unmatched block: {dag}/{checkpoint}/{method}')
            if method in BASELINES and checkpoint!='selected':
                continue
            family=f'ablation:{dag}:{checkpoint}' if method in CP_VARIANTS else 'external:selected'
            contrasts.append(dict(dag=dag,checkpoint=checkpoint,method=method,family=family,
                                  **paired_statistics(other.loc[common]-full.loc[common])))
    if contrasts:
        paired=pd.DataFrame(contrasts)
        if not args.allow_partial:
            for family,g in paired.groupby('family'):
                expected=27 if family=='external:selected' else 6
                if len(g)!=expected:
                    raise ValueError(f'Incomplete Holm family {family}: {len(g)} of {expected}')
        paired['holm_p']=paired.groupby('family').paired_t_p.transform(lambda p:holm(p))
        paired.to_csv(args.output/'paired_tests.csv',index=False)
    pressure=[]
    for dag in DAGS:
        sub=blocks[(blocks.dag==dag)&blocks.checkpoint.isin(['selected','none'])]
        pivot=sub.pivot(index='environment_seed',columns='method',values='AppTimeout')
        if {'CPGAPPO','HybridPSOGA','Edge-only'}.issubset(pivot.columns):
            for seed,row in pivot[['CPGAPPO','HybridPSOGA','Edge-only']].dropna().iterrows():
                pressure.append(dict(dag=dag,environment_seed=seed,pressure=row['Edge-only'],
                                     group='high' if row['Edge-only']>60 else 'lower',
                                     difference_pp=row.HybridPSOGA-row.CPGAPPO))
    pressure=pd.DataFrame(pressure)
    if not pressure.empty:
        pressure.to_csv(args.output/'pressure_blocks.csv',index=False)
        pressure.groupby(['dag','group']).difference_pp.agg(['count','mean',lambda x:int((x>0).sum()),lambda x:int((x<0).sum())]).to_csv(args.output/'pressure_groups.csv')
    draw_comparison(blocks,pressure,args.output)
    timing=[]
    for path in sorted(args.runs.rglob('timing/selected_arrival*.csv')):
        dag,method,seed=path.parents[1].name.split('__')
        t=pd.read_csv(path);t['dag']=dag;t['method']=method
        timing.append(t)
    if timing:
        pd.concat(timing).groupby('method').decision_ms.agg(mean_ms='mean',p99_ms=lambda x:np.percentile(x,99),active_slots='count').to_csv(args.output/'decision_timing.csv')


def draw_comparison(blocks,pressure,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#3454D1','#F06428','#0099B5','#00A86B','#E5AB00','#CC287A','#863BB5','#00B8CC','#DC3645','#70A02D']
    methods=['CPGAPPO',*BASELINES]
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),sharey=True)
    for ax,dag in zip(axes,DAGS):
        group=blocks[(blocks.dag==dag)&blocks.checkpoint.isin(['selected','none'])]
        for i,method in enumerate(methods):
            values=group[group.method==method].AppTimeout.to_numpy()
            if not len(values):continue
            mean=values.mean();half=stats.t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)) if len(values)>1 else 0
            ax.bar(i,mean,color=colors[i],edgecolor='#333333',linewidth=.5)
            ax.errorbar(i,mean,yerr=[[min(mean,half)],[min(100-mean,half)]],color='#222222',capsize=3,linewidth=.9)
            ax.scatter(i+np.linspace(-.16,.16,len(values)),values,s=10,c='#333333',zorder=3,alpha=.8)
        ax.set(title=dag.title(),ylim=(0,104),xticks=range(len(methods)),xticklabels=methods)
        ax.tick_params(axis='x',rotation=48,labelsize=7)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('AppTimeout (%)')
    fig.tight_layout();fig.savefig(out/'fig5_apptimeout.pdf');plt.close(fig)
    if not pressure.empty:
        fig,ax=plt.subplots(figsize=(5.8,3.5))
        for dag,color,marker in zip(DAGS,colors,['o','s','^']):
            g=pressure[pressure.dag==dag]
            ax.scatter(g.pressure,g.difference_pp,c=color,marker=marker,label=dag.title())
        ax.axvline(60,color='#555555',ls='--',lw=1);ax.axhline(0,color='#555555',lw=.7)
        ax.set(xlabel='Edge-only AppTimeout (%)',ylabel='HybridPSOGA − CPGAPPO (pp)')
        ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/'fig6_pressure.pdf');plt.close(fig)


def sensitivity(args):
    df=load_traces(args.runs,'training_evaluation.csv')
    df['parameter']=df.run.map(lambda p:Path(p).parts[0])
    df['value']=df.run.map(lambda p:float(Path(p).parts[1]))
    summary=df.groupby(['parameter','value','dag'])[METRICS].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(args.output/'sensitivity.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    for dag in df.dag.unique():
        mean=df[df.dag==dag].groupby(['parameter','value']).AppTimeout.mean()
        parameters=[p for p in SWEEPS if p in mean.index.get_level_values(0)]
        cells=np.full((5,len(parameters)),np.nan)
        for j,p in enumerate(parameters):
            if DEFAULTS[p] not in mean[p].index:raise ValueError(f'Missing default {p}')
            for i,v in enumerate(SWEEPS[p]):
                if v in mean[p].index:cells[i,j]=mean[p][DEFAULTS[p]]-mean[p][v]
        fig,ax=plt.subplots(figsize=(10,3.4))
        bound=max(.1,np.nanmax(np.abs(cells)))
        im=ax.imshow(cells,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
        for j,p in enumerate(parameters):
            for i,v in enumerate(SWEEPS[p]):
                if np.isfinite(cells[i,j]):ax.text(j,i,f'{v:g}\n{cells[i,j]:+.2f}',ha='center',va='center',fontsize=8)
                if v==DEFAULTS[p]:ax.add_patch(Rectangle((j-.48,i-.48),.96,.96,fill=False,lw=1.6))
        ax.set(xticks=range(len(parameters)),xticklabels=parameters,yticks=[])
        ax.tick_params(axis='x',labelsize=8,rotation=25)
        fig.colorbar(im,ax=ax,label='AppTimeout reduction (pp)')
        fig.tight_layout();fig.savefig(args.output/f'fig7_sensitivity_{dag}.pdf');plt.close(fig)


def variance(args):
    df=load_traces(args.runs)
    instances=df.groupby(['run','dag','environment_seed','checkpoint'])[METRICS].mean().reset_index()
    instances.to_csv(args.output/'optimization_instances.csv',index=False)
    instances.groupby(['dag','environment_seed','checkpoint'])[METRICS].agg(['count','mean','std']).to_csv(args.output/'optimization_variance.csv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['compare','sensitivity','variance'],default='compare')
    p.add_argument('--allow-partial',action='store_true',help='Permit reduced smoke-test grids; not the paper protocol')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    {'compare':compare,'sensitivity':sensitivity,'variance':variance}[args.mode](args)


if __name__=='__main__':main()
