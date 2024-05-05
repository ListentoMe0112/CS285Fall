from typing import Callable, Optional, Tuple
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os

log_dir = "F:\CS285\homework_fall2023\hw4\data"

def get_data(path:str, x:str) -> Tuple[list[float], list[float]]:
    acc = EventAccumulator(path)
    acc.Reload()
    step = []
    ret = []
    values = acc.Scalars(x)
    for val in values:
        step.append(val.step)
        ret.append(val.value)

    return step, ret

def get_all_key_name(path:str) -> None:
    print(path)
    acc = EventAccumulator(path)
    acc.Reload()
    
    keys = acc.Tags()['scalars']
    for key in keys:
        print(key)

def plot_problem_three() -> None:
    files:list[str] = os.listdir(log_dir)

    _, axes = plt.subplots(nrows=1, ncols=3)
    idx = 0
    for file in files:
        figure_name:str = ""
        if "multi" not in file:
            continue
        if "obstacles" in file:
            figure_name = "obstacles"
        elif "reacher" in file:
            figure_name = "reacher"
        elif "cheetah" in file:
            figure_name = "cheetah"
        
        log_files = os.listdir(os.path.join(log_dir, file))
        for log_file in log_files:
            if "events" not in log_file:
                continue
            # print(figure_name)
            # get_all_key_name(os.path.join(log_dir, file, log_file))
            step, eval_return = get_data(os.path.join(log_dir, file, log_file), "eval_return")
            axes[idx].plot(step, eval_return)
            axes[idx].set_title(figure_name)
            axes[idx].set_xlabel("Step")
            axes[idx].set_ylabel("eval_return")
            idx += 1
    plt.show()

def plot_problem_four() -> None:
    files:list[str] = os.listdir(log_dir)

    datas = { 
        "e1_h10_a1000" : {"step" : [], "eval_return": []},
        "e3_h10_a1000" : {"step" : [], "eval_return": []},
        "e6_h10_a1000" : {"step" : [], "eval_return": []},
        "e3_h5_a1000"  : {"step" : [], "eval_return": []},
        "e3_h20_a1000" : {"step" : [], "eval_return": []},
        "e3_h10_a500"  : {"step" : [], "eval_return": []},
        "e3_h10_a2000" : {"step" : [], "eval_return": []},
    }

    figures_info = {
        "ensemble_size"       : ["e1_h10_a1000", "e3_h10_a1000", "e6_h10_a1000"],
        "horizon_length"      : ["e3_h5_a1000", "e3_h10_a1000", "e3_h20_a1000"],
        "num_action_sequence" : ["e3_h10_a500", "e3_h10_a1000", "e3_h10_a2000"]
    }

    _, axes = plt.subplots(nrows=1, ncols=3)

    # Get Data
    for file in files:
        if "reacher_ablation" not in file:
            continue

        tag:str = ""
        for k,v in datas.items():
            if k in file:
                tag = k
                break
        if tag == "":
            print("error not find tag") 
            exit(1)


        log_files = os.listdir(os.path.join(log_dir, file))
        for log_file in log_files:
            if "events" not in log_file:
                continue
            datas[tag]["step"], datas[tag]["eval_return"] = get_data(os.path.join(log_dir, file, log_file), "eval_return")
    # Plot Figure
    _, axes = plt.subplots(nrows = 1, ncols = 3)
    idx = 0
    for name, figure_info in figures_info.items():
        for tag in figure_info:
            axes[idx].plot(datas[tag]["step"], datas[tag]["eval_return"], label = tag)
            axes[idx].legend()
            axes[idx].set_title(name)
            axes[idx].set_xlabel("Step")
            axes[idx].set_ylabel("eval_return")
 
            
        idx += 1
    plt.show()

def plot_problem_five() -> None:
    files:list[str] = os.listdir(log_dir)
    _, axes = plt.subplots(nrows=1, ncols=1)

    tag:str = ""
    for file in files:
        if "cheetah_multi" in file:
            tag = "random_shoot"
        elif "cheetah_cem" in file and "iters2" in file:
            tag = "cem_iter2"
        elif "cheetah_cem" in file and "iters4" in file:
            tag = "cem_iter4"
        else:
            continue
        log_files = os.listdir(os.path.join(log_dir, file))
        for log_file in log_files:
            if "events" not in log_file:
                continue
            step, eval_return = get_data(os.path.join(log_dir, file, log_file), "eval_return")
            axes.plot(step, eval_return, label = tag)
            axes.legend()
            axes.set_title("CEM && RandomShooting")
            axes.set_xlabel("Step")
            axes.set_ylabel("eval_return")
    plt.show()

def plot_problem_six() -> None:
    files:list[str] = os.listdir(log_dir)
    _, axes = plt.subplots(nrows=1, ncols=1)

    tag:str = ""
    for file in files:
        if "cheetah_mbpo" not in file:
            continue
        if "rollout1" in file and "rollout10" not in file:
            tag = "rollout1"
        elif "rollout10" in file:
            tag = "rollout10"
        else:
            tag = "rollout0"

        log_files = os.listdir(os.path.join(log_dir, file))
        for log_file in log_files:
            if "events" not in log_file:
                continue
            step, eval_return = get_data(os.path.join(log_dir, file, log_file), "eval_return")
            axes.plot(step, eval_return, label = tag)
            axes.legend()
            axes.set_title("MBPO")
            axes.set_xlabel("Step")
            axes.set_ylabel("eval_return")
    plt.show()


if __name__ == "__main__":
    # plot_problem_three()
    # plot_problem_four()
    # plot_problem_five()
    plot_problem_six()