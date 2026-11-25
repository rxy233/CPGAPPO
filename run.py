"""Train and evaluate online DAG offloading on independently seeded arrivals."""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CP_VARIANTS = ("CPGAPPO", "noguidece", "noshield", "noappcredit", "nocp", "fwdonly", "alloff")
BASELINES = ("DGMA_adapt", "TransEdgeStyle", "GATDQN", "PPO",
             "HybridPSOGA", "Genetic", "Greedy", "Edge-only", "Local-only")
TRAINERS = {
    "DGMA_adapt": ("train_dgma_adapt_wrapper", "train_dgma_adapt_wrapper"),
    "TransEdgeStyle": ("train_transedge_style_wrapper", "train_transedge_style_wrapper"),
    "GATDQN": ("train_dynamic_gat_dqn", "train_dynamic_gat_dqn_wrapper"),
    "PPO": ("train_ppo", "train_ppo_wrapper"),
}
CP_NAMES = dict(zip(CP_VARIANTS, (
    "train_cpgappo_dual_cpgappo", "train_cpgappo_dual_wo_guidece",
    "train_cpgappo_dual_wo_shield", "train_cpgappo_dual_wo_appcredit",
    "train_cpgappo_dual_wo_cpseq", "train_cpgappo_dual_forward_only",
    "train_cpgappo_dual_all_off")))
SWEEPS = {
    "guide_ce": [0, .05, .1, .2, .4],
    "timeout_credit": [-14, -10, -7, -4, -2],
    "energy_weight": [0, .025, .05, .1, .2],
    "delay_weight": [0, .05, .1, .2, .4],
    "shield_high": [1, 1.05, 1.1, 1.15, 1.2],
    "shield_normal": [1.05, 1.1, 1.15, 1.2, 1.3],
    "cp_coefficient": [.2, .3, .45, .6, .8],
    "overflow_multiplier": [1, 1.1, 1.25, 1.5, 2],
}


def configure(args, dag, seed, run_dir):
    import torch
    from experiments.exp_utils import CONFIG
    from utils.constant import para
    import Algorithms.Train.train_cpgappo_unified as trainer
    import Algorithms.RealGATPPO.agent_cpgappo as agent
    import Algorithms.RealGATPPO.cpgappo_core as core
    matrix = ROOT / "matrix" / ("matrix_60.txt" if dag == "default" else f"matrix_60_{dag}.txt")
    os.environ["MATRIX_OVERRIDE_PATH"] = str(matrix)
    torch.set_num_threads(args.threads)
    CONFIG.update(SEED=0, RUN_DIR=str(run_dir), MAX_STEPS=args.max_steps,
                  STOP_ARRIVAL_STEP=min(2000, args.max_steps - 1),
                  BURST_PROB=.2, BURST_SIZE=55, AUTO_EVAL_AFTER_TRAIN=False,
                  OPTIMIZATION_SEED=args.optimization_seed)
    para.update(user_num=args.users, deadline_slot=55)
    kwargs = {}
    if args.parameter is not None:
        if args.value is None:
            raise ValueError("--value is required with --parameter")
        value = args.value
        bindings = {
            "timeout_credit": (trainer, "BONUS_TO"),
            "energy_weight": (core, "ENERGY_WEIGHT"),
            "delay_weight": (core, "DELAY_WEIGHT"),
            "shield_high": (trainer, "SHIELD_THR_HIGH"),
            "shield_normal": (trainer, "SHIELD_THR_NORM"),
            "cp_coefficient": (agent, "CP_REM_COEFF"),
            "overflow_multiplier": (agent, "OVERFLOW_MULT"),
        }
        if args.parameter == "guide_ce":
            kwargs["lambda_guide"] = value
        else:
            module, attr = bindings[args.parameter]
            setattr(module, attr, value)
    return matrix, kwargs


def evaluate_cp(args, dag, seed, run_dir, checkpoint, training_trace=False):
    import time
    import torch
    import numpy as np
    import baseline_evaluation as evaluation
    import Algorithms.Train.train_cpgappo_unified as trainer
    from Algorithms.RealGATPPO.cpgappo_core import extract_cpgappo_state
    from Algorithms.RealGATPPO.agent_cpgappo import GAT_PPO_Agent_CPGAPPO
    from experiments.exp_utils import generate_arrival_plan
    from utils.constant import para
    matrix, _ = configure(args, dag, seed, run_dir)
    evaluation.MAX_STEPS = args.max_steps
    evaluation.STOP_ARRIVAL_STEP = min(2000, args.max_steps - 1)
    tag = "CPGAPPO" if args.method == "CPGAPPO" else "CPGAPPO_" + args.method
    ckpt = run_dir / "checkpoints" / (tag + ("_last" if checkpoint == "last" else "") + ".pt")
    rows = []
    for arrival in ([seed] if training_trace else args.arrival_seeds):
        ctx = evaluation._build_env(str(matrix), str(run_dir), seed)
        ts, gs, bc = (ctx[k] for k in ("ts", "gs", "bc"))
        plan = generate_arrival_plan(arrival, args.max_steps, evaluation.STOP_ARRIVAL_STEP, .3, .2, 27, 55)
        data, global_feat, _ = extract_cpgappo_state(ts, (0, 0), 0, ctx["task_complex_index"])
        model = GAT_PPO_Agent_CPGAPPO(data.x.shape[1], len(global_feat), para["edge_num"]+2,
                                    ctx["device"], use_backward=args.method != "fwdonly")
        model.policy.load_state_dict(torch.load(ckpt, map_location=ctx["device"], weights_only=True), strict=True)
        begin = time.perf_counter()
        _, _, _, metrics, _ = trainer.eval_cpgappo_once(
            model, ts, gs, bc, plan, ctx["task_complex_index"], args.max_steps,
            use_shield=args.method not in ("noshield", "alloff"),
            use_cp=args.method not in ("nocp", "alloff"))
        evaluation.EVALUATION_ARRIVAL_SEED = arrival
        evaluation.write_timing(ts, run_dir, checkpoint)
        energy, latency = ts.get_avg_results(only_successful=False, timeout_charge="deadline")
        rows.append(dict(dag=dag, method=args.method, environment_seed=seed,
                         arrival_seed=arrival, checkpoint=checkpoint,
                         AppTimeout=100*metrics["app_timeout_rate"],
                         TaskTimeout=100*metrics["task_timeout_rate"],
                         D_all=latency, E_avg=energy, elapsed_seconds=time.perf_counter()-begin))
    return rows


def execute_cell(args):
    dag, seed = args.dags[0], args.seeds[0]
    run_dir = args.output.resolve() / f"{dag}__{args.method}__seed{seed}"
    if args.mode == "train":
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    _, kwargs = configure(args, dag, seed, run_dir)
    config_path = run_dir / 'configuration.json'
    configuration = dict(dag=dag, method=args.method, environment_seed=seed,
                         applications=args.users, max_steps=args.max_steps,
                         parameter=args.parameter, value=args.value,
                         optimization_seed=args.optimization_seed)
    if args.mode == 'train':
        config_path.write_text(json.dumps(dict(configuration, episodes=args.episodes), indent=2), encoding='utf-8')
    elif config_path.exists():
        saved = json.loads(config_path.read_text(encoding='utf-8'))
        for key, value in configuration.items():
            if saved.get(key) != value:
                raise ValueError(f'Evaluation setting {key} differs from the training configuration')
    if args.mode == "train":
        if args.method in CP_VARIANTS:
            import Algorithms.Train.train_cpgappo_unified as module
            getattr(module, CP_NAMES[args.method])(gpu_id=0, seed_offset=seed, episodes=args.episodes, **kwargs)
        elif args.method in TRAINERS:
            mod, name = TRAINERS[args.method]
            module = importlib.import_module("Algorithms.Train." + mod)
            getattr(module, name)(gpu_id=0, seed_offset=seed, episodes=args.episodes)
        return
    if args.method in CP_VARIANTS:
        rows = []
        for checkpoint in args.checkpoints:
            rows.extend(evaluate_cp(args, dag, seed, run_dir, checkpoint, args.training_trace))
    else:
        import baseline_evaluation as evaluation
        evaluation.OUT_ROOT = str(args.output.resolve())
        evaluation.MAX_STEPS = args.max_steps
        evaluation.STOP_ARRIVAL_STEP = min(2000, args.max_steps-1)
        rows = []
        for arrival in ([seed] if args.training_trace else args.arrival_seeds):
            evaluation.EVALUATION_ARRIVAL_SEED = arrival
            result = evaluation.reeval_one(args.method, dag, seed)
            if result.get("status") != "ok":
                raise RuntimeError(result)
            rows.append(dict(dag=dag, method=args.method, environment_seed=seed,
                             arrival_seed=arrival, checkpoint="selected" if args.method in TRAINERS else "none",
                             AppTimeout=result["AppTO"], TaskTimeout=result["TaskTO"],
                             D_all=result["D_all"], E_avg=result["E_all"]))
    destination = run_dir / ("training_evaluation.csv" if args.training_trace else "heldout.csv")
    with destination.open("w",newline="",encoding="utf-8") as fp:
        writer=csv.DictWriter(fp,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    print(destination,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("train","evaluate","sweep"),default="train")
    p.add_argument("--method", choices=CP_VARIANTS+BASELINES,default="CPGAPPO")
    p.add_argument("--dags",nargs="+",choices=("default","chain","wide"),default=["default","chain","wide"])
    p.add_argument("--seeds",nargs="+",type=int,default=[1,3,5,7,42])
    p.add_argument("--arrival-seeds",nargs="+",type=int,default=[21001,21002,21003])
    p.add_argument("--checkpoints",nargs="+",choices=("selected","last"),default=["selected","last"])
    p.add_argument("--episodes",type=int,default=100)
    p.add_argument("--max-steps",type=int,default=8000)
    p.add_argument("--users",type=int,default=150)
    p.add_argument("--threads",type=int,default=1)
    p.add_argument("--optimization-seed",type=int)
    p.add_argument("--output",type=Path,default=ROOT/"runs")
    p.add_argument("--parameter",choices=tuple(SWEEPS))
    p.add_argument("--value",type=float)
    p.add_argument("--training-trace",action="store_true")
    p.add_argument("--cell",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.episodes<1 or args.max_steps<2 or args.users<1 or args.threads<1:
        p.error("Episodes, users and threads must be positive; max-steps must be at least two")
    if not args.training_trace and set(args.seeds)&set(args.arrival_seeds):
        p.error("Held-out arrival seeds must differ from training arrival seeds")
    if args.mode=="sweep":
        if args.parameter is None:
            p.error("--parameter is required for a sweep")
        for value in SWEEPS[args.parameter]:
            for mode in ("train","evaluate"):
                command=[sys.executable,str(Path(__file__).resolve()),"--mode",mode,"--method","CPGAPPO",
                         "--parameter",args.parameter,"--value",str(value),"--output",str(args.output/args.parameter/str(value)),
                         "--dags",*args.dags,"--seeds",*map(str,args.seeds),"--episodes",str(args.episodes),
                         "--users",str(args.users),"--max-steps",str(args.max_steps),"--threads",str(args.threads)]
                if args.optimization_seed is not None:
                    command += ["--optimization-seed", str(args.optimization_seed)]
                if mode=="evaluate":
                    command+=["--training-trace","--checkpoints","selected"]
                subprocess.run(command,check=True)
        return
    if args.cell:
        if len(args.dags)!=1 or len(args.seeds)!=1:
            p.error("A cell requires one topology and one environment seed")
        execute_cell(args);return
    for dag in args.dags:
        for seed in args.seeds:
            command=[sys.executable,str(Path(__file__).resolve()),"--cell","--mode",args.mode,
                     "--method",args.method,"--dags",dag,"--seeds",str(seed),"--output",str(args.output),
                     "--episodes",str(args.episodes),"--max-steps",str(args.max_steps),
                     "--users",str(args.users),"--threads",str(args.threads),
                     "--arrival-seeds",*map(str,args.arrival_seeds),"--checkpoints",*args.checkpoints]
            if args.training_trace:
                command.append("--training-trace")
            if args.parameter is not None:
                command+=["--parameter",args.parameter,"--value",str(args.value)]
            if args.optimization_seed is not None:
                command+=["--optimization-seed",str(args.optimization_seed)]
            subprocess.run(command,check=True)


if __name__=="__main__":
    main()
