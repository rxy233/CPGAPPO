"""Launch the manuscript's training and evaluation grids."""
import argparse
from pathlib import Path
import subprocess
import sys
from run import BASELINES, CP_VARIANTS, SWEEPS, TRAINERS

ROOT=Path(__file__).resolve().parent


def invoke(script,*args):
    subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],check=True)


def run_methods(methods,out):
    for method in methods:
        if method in CP_VARIANTS or method in TRAINERS:
            invoke('run.py','--mode','train','--method',method,'--output',out)
        invoke('run.py','--mode','evaluate','--method',method,'--output',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=['main','ablation','sensitivity','variance','all'],default='main')
    p.add_argument('--output',type=Path,default=ROOT/'runs')
    args=p.parse_args();choice=args.experiment
    if choice in ('main','all'):
        run_methods(['CPGAPPO',*BASELINES],args.output/'comparison')
    if choice in ('ablation','all'):
        run_methods(CP_VARIANTS[1:] if choice=='all' else CP_VARIANTS,args.output/'comparison')
    if choice in ('main','ablation','all'):
        invoke('analyze.py','--runs',args.output/'comparison','--output',args.output/'analysis')
    if choice in ('sensitivity','all'):
        for parameter in SWEEPS:
            dags=['default','chain','wide'] if parameter in ('guide_ce','shield_high','shield_normal') else ['default']
            invoke('run.py','--mode','sweep','--parameter',parameter,'--output',args.output/'sensitivity','--dags',*dags)
        invoke('analyze.py','--mode','sensitivity','--runs',args.output/'sensitivity','--output',args.output/'sensitivity_analysis')
    if choice in ('variance','all'):
        for optimization_seed in (None,101,103):
            out=args.output/'variance'/('base' if optimization_seed is None else f'opt{optimization_seed}')
            extra=[] if optimization_seed is None else ['--optimization-seed',optimization_seed]
            for mode in ('train','evaluate'):
                invoke('run.py','--mode',mode,'--method','CPGAPPO','--seeds',1,3,'--output',out,*extra)
        invoke('analyze.py','--mode','variance','--runs',args.output/'variance','--output',args.output/'variance_analysis')


if __name__=='__main__':main()
