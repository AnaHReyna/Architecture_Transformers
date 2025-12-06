import os, sys, random, argparse, glob
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from copy import deepcopy
from tensorboard.backend.event_processing import event_accumulator as EA

from envs.carla.carla_env import InterSection
from train.init_configs import get_argument, set_configs
from algos.sac import SAC
from envs.runners.off_policy_trainer_carla import Trainer


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_env(args):
    env = InterSection()
    env.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(args.neighbors + 1, args.N_steps, args.dim,))
    env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

    try:
        env.seed(getattr(args, "seed", None))
    except Exception:
        pass
    
    return env


def train_once(base_args, seed: int, gpu_override: int):
    """
    Executa uma run de treino e retorna o diretório de logs do experimento (runner._output_dir).
    """
    set_global_seed(seed)

    args = deepcopy(base_args)
    args.dir_suffix = (args.dir_suffix + f"_seed{seed}") if args.dir_suffix else f"seed{seed}"

    args, algo_params, runner_params = set_configs(args, test=False)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    use_gpu = gpu_override if gpu_override is not None else getattr(args, "gpu", None)
    if use_gpu is not None and use_gpu >= 0 and len(gpus) > use_gpu:
        tf.config.set_visible_devices([gpus[use_gpu]], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[use_gpu], True)
        print(f"[seed {seed}] Using GPU:", gpus[use_gpu])
    else:
        tf.config.set_visible_devices([], 'GPU')
        print(f"[seed {seed}] Training on CPU.")

    env = make_env(args)
    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        max_action=env.action_space.high[0],
        **algo_params
    )

    runner = Trainer(policy=policy, env=env, args=args, test_env=env, **runner_params)
    run_dir = runner._output_dir  # onde tfevents/ckpts serão gravados

    print(f"[seed {seed}] output_dir: {run_dir}")
    runner()

    return run_dir


def load_scalar_series(run_dir: str, tag: str):
    """
    Lê a 'tag' escalar do TensorBoard em 'run_dir'.
    Retorna pandas.Series com index=step e values=valor.
    """

    ea = EA.EventAccumulator(run_dir, size_guidance={'scalars': 0})
    ev = ea.Scalars(tag)
    steps = np.array([e.step for e in ev], dtype=np.int64)
    vals  = np.array([e.value for e in ev], dtype=np.float64)

    s = pd.Series(vals, index=steps).sort_index()
    return s


def reindex_to_grid(s: pd.Series, grid_step: int, max_step: int = None):
    if max_step is None:
        max_step = int(s.index.max())
    grid = np.arange(0, max_step + 1, grid_step, dtype=np.int64)
    df = pd.DataFrame({'y': s})
    df = df.reindex(grid).sort_index().ffill().bfill()  # preenche do último valor e do primeiro
    return df['y']


def ema(series: pd.Series, gamma: float):
    m = None
    out = []
    for x in series.values:
        if m is None:
            m = x
        else:
            m = gamma * m + (1 - gamma) * x                                        

        out.append(m)
    return pd.Series(out, index=series.index)


def plot_aggregated(run_dirs, tag="Common/training_success", grid_step=200, ema_gamma=0.99, label="Transformer (scene_rep)",
                    out_path=None, max_step=None, title="Treinamento — média ± DP"):
    series = []
    max_each = []

    for rd in run_dirs:
        s = load_scalar_series(rd, tag)
        max_each.append(int(s.index.max()))
        series.append(s)
        max_each.append(int(s.index.max()))

    if max_step is None:
        max_step = min(max_each)  # interseção das runs

    processed = []
    for s in series:
        ss = reindex_to_grid(s, grid_step=grid_step, max_step=max_step)
        if ema_gamma is not None:
            ss = ema(ss, ema_gamma)
        processed.append(ss)

    M = pd.concat(processed, axis=1)  # cada coluna = uma run
    n = M.shape[1]
    mean = M.mean(axis=1)
    std  = M.std(axis=1, ddof=(1 if n > 1 else 0))

    x = mean.index.values
    plt.figure(figsize=(8,5))
    plt.plot(x, mean.values, label=f"{label} (média de {n} execuções)")
    if n >= 2:
        plt.fill_between(x, (mean-std).values, (mean+std).values, alpha=0.25, label="± desvio-padrão")
    plt.xlabel("Passos de ambiente")
    plt.ylabel("Taxa de sucesso no treinamento\n(janela 20 eps, EMA=0.99)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
        print("Figura salva em:", out_path)
    plt.show()


def main():
    sp = argparse.ArgumentParser(description="Treina N seeds e plota média±DP (Fig.9 style).",
                                 add_help=True)
    sp.add_argument("--seeds", default="0,1", help="Lista de seeds, ex.: 0,1")
    sp.add_argument("--gpu", type=int, default=None, help="GPU a usar (prioriza esta sobre --gpu do treino)")
    sp.add_argument("--grid-step", type=int, default=200, help="Step entre pontos (Fig.9 usa 200)")
    sp.add_argument("--ema", type=float, default=0.99, help="EMA para suavização (Fig.9 usa 0.99)")
    sp.add_argument("--tag", default="Common/training_success", help="Tag do TensorBoard")
    sp.add_argument("--label", default="Transformer (scene_rep)", help="Rótulo da curva")
    sp.add_argument("--out", default="fig9_scene_rep.png", help="Arquivo para salvar a figura")
    sp.add_argument("--skip-train", action="store_true", help="Pular treino e só plotar (usa runs existentes)")
    script_args, rest = sp.parse_known_args() # Captura flags do script e deixa o resto para o parser do projeto

    base_parser = get_argument()
    base_args   = base_parser.parse_args(rest)

    seeds = [int(s.strip()) for s in script_args.seeds.split(",") if s.strip() != ""]

    run_dirs = []
    if not script_args.skip_train:
        for sd in seeds:
            print(f"\n===== Treinando seed {sd} =====")
            rd = train_once(base_args, seed=sd, gpu_override=script_args.gpu)
            run_dirs.append(rd)
    else:
        # tenta achar runs recentes contendo seedX no nome
        candidates = []
        for root, dirs, _ in os.walk(base_args.logdir):
            for d in dirs:
                if any(f"seed{sd}" in d for sd in seeds):
                    candidates.append(os.path.join(root, d))
        candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
        picked, seen = [], set()
        for c in candidates:
            for sd in seeds:
                if f"seed{sd}" in c and sd not in seen:
                    picked.append(c); seen.add(sd); break
            if len(seen) == len(seeds):
                break
        if len(picked) != len(seeds):
            raise SystemExit("Não achei pastas para todas as seeds; rode sem --skip-train.")
        run_dirs = picked
        print("Usando runs existentes:", run_dirs)

    # Agrega e plota (estilo Fig. 9)
    print("\n===== Agregando e plotando (Fig. 9 style) =====")
    plot_aggregated(
        run_dirs=run_dirs,
        tag=script_args.tag,
        grid_step=script_args.grid_step,
        ema_gamma=script_args.ema,
        label=script_args.label,
        out_path=script_args.out,
        max_step=None,
        title="Treinamento — média ± DP (Fig. 9)"
    )

if __name__ == "__main__":
    main()