from typing import Callable, Optional, Tuple
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os

log_dir = "F:\CS285\homework_fall2023\hw5\data"

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

def plot_cql_result() -> None:
    files:list[str] = os.listdir(log_dir)
    _, axes = plt.subplots(nrows=1, ncols=2)

    tag:str = ""
    for file in files:
        if "hw5_offline" not in file or "cql" not in file or "PointmassMedium" not in file:
            continue

        tag = "unknown"
        if "cql0.1" in file:
            tag = "cql0.1"
        elif "cql10" in file:
            tag = "cql10"
        elif "cql1" in file:
            tag = "cql1"
        elif "cql0" in file:
            tag = "dqn"
        elif "cql5" in file:
            tag = "cql5"
        
        print(tag)


        log_files = os.listdir(os.path.join(log_dir, file))
        for log_file in log_files:
            if "events" not in log_file:
                continue
            step, q_values = get_data(os.path.join(log_dir, file, log_file), "q_values")
            
            axes[0].plot(step, q_values, label = tag, alpha=0.5)
            axes[0].legend()

            step, eval_return = get_data(os.path.join(log_dir, file, log_file), "eval_return")
            axes[1].plot(step, eval_return, label = tag, alpha=0.5)
            axes[1].legend()

        axes[0].set_title("CQL QValues")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("QValues")
        axes[1].set_title("CQL Eval Return")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Eval Return")

    plt.show()

if __name__ == "__main__":
    plot_cql_result()