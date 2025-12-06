# # import os, glob, re
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from tensorboard.backend.event_processing import event_accumulator as EA
# import tensorflow as tf
# from init_configs import get_argument, set_configs
# # from tabulate import tabulate


# def ema(series, gamma):
#     m = None
#     out = []
#     for x in series.values:
#         if m is None:
#             m = x
#         else:
#             m = gamma * m + (1 - gamma) * x
#         out.append(m)
#     return pd.Series(out, index=series.index)


# def reindex_to_grid(s, grid):
#     df = pd.DataFrame({'y': s})
#     df = df.reindex(grid)
#     # df['y'] = df['y'].interpolate()
#     return df


# def load_series(run_dir, tag):
#     ea = EA.EventAccumulator(run_dir, size_guidance={EA.SCALARS: 0, EA.TENSORS: 0})
#     ea.Reload()
#     tags = ea.Tags()
#     if tag in tags.get('scalars', []):
#         ev = ea.Scalars(tag)
#         steps = np.array([e.step for e in ev], dtype=np.int64)
#         vals  = np.array([e.value for e in ev], dtype=np.float64)
#     else:
#         tag in tags.get('tensors', [])
#         ev = ea.Tensors(tag)
#         steps = np.array([e.step for e in ev], dtype=np.int64)
#         vals  = np.array([tf.make_ndarray(e.tensor_proto).reshape(-1)[0] for e in ev],
#                          dtype=np.float64)
#     s = pd.Series(vals, index=steps)
#     return s


# def interpolation(s, total_steps=100000):

#     grid_steps = np.arange(1, total_steps + 1, dtype=np.int64)
#     y = s.reindex(grid_steps)

#     if len(s.index) >= 4:
#         # print(s.notna().sum())
#         method = 'cubic'

#     # if s.notna().sum() >= 4:
#     #     method = 'cubic'
#     else:
#         method = 'linear'

#     y = y.interpolate(method=method).ffill().bfill()
#     return y

# def find_runs_auto(model_dir, base):
#     if model_dir is not None:
#         md = Path(model_dir)
#         algo_dir = md.parents[2]

#     candidates = []
#     for p in algo_dir.glob(f'{base}*'):
#         td = p / "tb" / "train"
#         if p.is_dir():
#             candidates.append(str(td))
#     return candidates


# def main(args, tag):
#     GRID_STEP = 200
#     EMA_GAMMA = 0.99
#     # TITLE     = f"Training — {args.algo} - {tag}"
#     TITLE     = f"Train"
#     YLABEL    = tag

#     args, algo_params, runner_params = set_configs(args, test=False)

#     if args.model_dir is None:
#         args.model_dir = str(Path("../train/results") / args.algo / "tb" / "train")

#     run_dirs = find_runs_auto(args.model_dir, base=args.algo)
#     print('=========', run_dirs)
    
#     series = []
#     names  = []
#     for rd in run_dirs:
#         s = load_series(rd, tag)
#         s = interpolation(s)
#         series.append(s)
#         exp_name = Path(rd).parents[1].name  # {algo}, {algo}_1, {algo}_2
#         names.append(exp_name)


#     mins = []
#     for s in series:
#         mini = s.index.min()
#         mins.append(mini)

#     maxs= []
#     for s in series:
#         maxi = s.index.max()
#         maxs.append(maxi)

    
#     start = max(mins)
#     end   = min(maxs)                
#     grid = np.arange(start, end + 1, GRID_STEP, dtype=np.int64)

#     proc = []
#     for i, s in enumerate(series):
#         y = reindex_to_grid(s, grid)
#         y = ema(y, EMA_GAMMA)
#         if i < len(names):
#             proc.append(y.rename(names[i]))

#     M = pd.concat(proc, axis=1)
#     n = M.shape[1]
#     mean = M.mean(axis=1)

#     if n > 1:
#         std = M.std(axis=1, ddof = 1)
#     else:
#         std = pd.Series(0.0, index=M, name='std')

#     # plot: média ± DP (se houver 2+ runs)
#     plt.figure(figsize=(8, 5))
#     plt.plot(mean.index.values, mean.values, '.', label=f"{args.algo} · {tag} (média de {n} runs)")
#     if n >= 2:
#         plt.fill_between(mean.index.values, (mean - std).values, (mean + std).values,
#                          alpha=0.25, label="± desvio-padrão")
#     plt.xlabel("Passos de ambiente")
#     plt.ylabel(YLABEL)
#     plt.title(TITLE)
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()



# if __name__ == "__main__":
#     parser = get_argument()
#     args = parser.parse_args()
#     tag = "success"  
#     main(args, tag)
##########################################################################################################################################################

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator as EA
import tensorflow as tf
from init_configs import get_argument, set_configs


def ema(series, gamma):
    m = None
    out = []
    for x in series.values:
        if m is None:
            m = x
        else:
            m = gamma * m + (1 - gamma) * x
        out.append(m)
    return pd.Series(out, index=series.index)


def reindex_to_grid(s, grid):
    df = pd.DataFrame({'y': s})
    df = df.reindex(grid)
    return df


def load_series(run_dir, tag):
    ea = EA.EventAccumulator(run_dir, size_guidance={EA.SCALARS: 0, EA.TENSORS: 0})
    ea.Reload()
    tags = ea.Tags()
    if tag in tags.get('scalars', []):
        ev = ea.Scalars(tag)
        steps = np.array([e.step for e in ev], dtype=np.int64)
        vals  = np.array([e.value for e in ev], dtype=np.float64)
    else:
        ev = ea.Tensors(tag)
        steps = np.array([e.step for e in ev], dtype=np.int64)
        vals  = np.array([tf.make_ndarray(e.tensor_proto).reshape(-1)[0] for e in ev],
                         dtype=np.float64)
    s = pd.Series(vals, index=steps)
    print('=========', s)
    return s


def interpolation(s, total_steps=100000):
    grid_steps = np.arange(1, total_steps + 1, dtype=np.int64)
    y = s.reindex(grid_steps)

    if len(s.index) >= 4:
        method = 'cubic'
    else:
        method = 'linear'

    y = y.interpolate(method=method).ffill().bfill()
    return y


def find_runs_auto(model_dir, base):
    if model_dir is None:
        return []

    md = Path(model_dir)
    if not md.exists():
        return []

    # manter o comportamento original
    if len(md.parents) < 3:
        return []

    algo_dir = md.parents[2]

    candidates = []
    for p in algo_dir.glob(f'{base}*'):
        td = p / "tb" / "train"
        if p.is_dir() and td.exists():
            candidates.append(str(td))
    return candidates


def main(args, tag):
    GRID_STEP = 200
    EMA_GAMMA = 0.99
    TITLE     = "Train"
    YLABEL    = tag

    args, algo_params, runner_params = set_configs(args, test=False)

    grouped_runs = []  # lista de dicts: {group: 'treino_1', run_dir: '...'}
    if args.model_dir is None:
        base_root = Path("../train/results")
        # subdirs = ["treino_1", "treino_2"]
        subdirs = ["treino_1"]

        for sd in subdirs:
            md = base_root / sd / args.algo / "tb" / "train"
            found = find_runs_auto(str(md), base=args.algo)
            for rd in found:
                grouped_runs.append({"group": sd, "run_dir": rd})
    else:
        # comportamento antigo
        found = find_runs_auto(args.model_dir, base=args.algo)
        for rd in found:
            grouped_runs.append({"group": "default", "run_dir": rd})

    # se não achou nada, não tem o que plotar
    if not grouped_runs:
        print("Nenhum run encontrado.")
        return

    # carregar todas as séries
    all_series = []
    for item in grouped_runs:
        rd = item["run_dir"]
        grp = item["group"]

        s = load_series(rd, tag)
        s = interpolation(s)

        all_series.append({
            "group": grp,
            "series": s
        })

    # alinhar min/max pra todo mundo
    mins = [x["series"].index.min() for x in all_series]
    maxs = [x["series"].index.max() for x in all_series]

    start = max(mins)
    end   = min(maxs)
    grid = np.arange(start, end + 1, GRID_STEP, dtype=np.int64)

    # agora vamos reindexar e aplicar EMA em cada série
    # e agrupar por nome de grupo
    grouped_processed = {}  # group -> list of series (pd.Series)
    for item in all_series:
        grp = item["group"]
        s   = item["series"]
        y = reindex_to_grid(s, grid)
        y = ema(y, EMA_GAMMA)  # fazer interpolação depois
        if grp not in grouped_processed:
            grouped_processed[grp] = []
        grouped_processed[grp].append(y)

    # plot
    plt.figure(figsize=(8, 5))


    # renomear legendas
    name_map = {
        "treino_1": "With future state",
        "treino_2": "Without future state"
    }

    for grp, series_list in grouped_processed.items():
        # empilha pra fazer média do grupo
        M = pd.concat(series_list, axis=1)
        mean = M.mean(axis=1)
        if M.shape[1] > 1:
            std = M.std(axis=1, ddof=1)
        else:
            std = pd.Series(0.0, index=M.index)

        # uma curva por grupo
        # plt.plot(mean.index.values, mean.values, label=f"{grp} (n={M.shape[1]})", marker='.')
        label_name = name_map.get(grp, grp)  # usa nome mapeado se existir
        plt.plot(mean.index.values, mean.values, label=f"{label_name} (n={M.shape[1]})", marker='.')


        # se quiser manter o sombreado por grupo:
        if M.shape[1] >= 2:
            plt.fill_between(mean.index.values,
                             (mean - std).values,
                             (mean + std).values,
                             alpha=0.15)

    plt.xlabel("Environment steps")
    # plt.ylabel(YLABEL)
    plt.ylabel("Success Rate")
    plt.title(TITLE)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = get_argument()
    args = parser.parse_args()
    tag = "success"
    main(args, tag)