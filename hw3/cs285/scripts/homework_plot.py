from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os

log_dir = "F:\CS285\homework_fall2023\hw3\data"

def get_data(path:str, x:str) -> list:
    acc = EventAccumulator(path)
    acc.Reload()
    step = []
    ret = []
    values = acc.Scalars(x)
    for val in values:
        step.append(val.step)
        ret.append(val.value)

    return step, ret

# eval_return
# eval_ep_len
# eval/return_std
# eval/return_max
# eval/return_min
# eval/ep_len_std
# eval/ep_len_max
# eval/ep_len_min
# train_return
# train_ep_len
# critic_loss
# q_values
# target_values
# grad_norm
# epsilon
# lr
def get_all_key_name(path:str) -> None:
    print(path)
    acc = EventAccumulator(path)
    acc.Reload()
    
    keys = acc.Tags()['scalars']
    for key in keys:
        print(key)

def plot_experiment_one() -> None:
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    files = os.listdir(log_dir)
    
    for file in files:
        if "CartPole" not in file:
            continue
        log_files = os.listdir(os.path.join(log_dir, file))
        
        if "lr_0.05" in file:
            tag = "lr=0.05"
        else:
            tag = "lr=0.001"
        
        for log_file in log_files:
            # get_all_key_name(os.path.join(log_dir, file, log_file))
            step, q_values = get_data(os.path.join(log_dir, file, log_file), "q_values")
            axes[0].plot(step, q_values, label=tag + "q_values")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("q_values")
            
            step, critic_loss = get_data(os.path.join(log_dir, file, log_file), "critic_loss")
            axes[1].plot(step, critic_loss, label=tag + "critic_loss")
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("critic_loss")
    axes[0].legend()
    axes[1].legend()
    plt.show()
    
def plot_experiment_two() -> None:
    fig, axes = plt.subplots(nrows=1, ncols=1)
    
    files = os.listdir(log_dir)
    for file in files:
        if not ("LunarLander" in file and "doubleq" not in file):
            continue
        tag = file[-2:]
        
        log_files = os.listdir(os.path.join(log_dir, file))
        
        for log_file in log_files:
            step, eval_return = get_data(os.path.join(log_dir, file, log_file), "eval_return")
            axes.plot(step, eval_return, label = tag)
            axes.set_xlabel("Step")
            axes.set_ylabel("eval_return")
    
    axes.legend()
    plt.show()
    return

def plot_experiment_three() -> None:
    fig, axes = plt.subplots(nrows=1, ncols=1)
    files = os.listdir(log_dir)
    
    for file in files:
        if ("LunarLander" in file and "doubleq" in file):
            tag = "vanilla dqn"
            log_files = os.listdir(os.path.join(log_dir, file))
        
            for log_file in log_files:
                step, eval_return = get_data(os.path.join(log_dir, file, log_file), "eval_return")
                axes.plot(step, eval_return, label = tag, color = "red")
                axes.set_xlabel("Step")
                axes.set_ylabel("eval_return")
        elif ("LunarLander" in file and "doubleq" not in file):
            tag = "double dqn"
            log_files = os.listdir(os.path.join(log_dir, file))
        
            for log_file in log_files:
                step, eval_return = get_data(os.path.join(log_dir, file, log_file), "eval_return")
                axes.plot(step, eval_return, label = tag, color = "blue")
                axes.set_xlabel("Step")
                axes.set_ylabel("eval_return")
                
    axes.legend()
    plt.show()
    
def plot_experiment_four() -> None:
    fig, axes = plt.subplots(nrows=1, ncols=1)
    files = os.listdir(log_dir)
    
    for file in files:
        if ("MsPacman" not in file):
            continue
        log_files = os.listdir(os.path.join(log_dir, file))
    
        for log_file in log_files:
            step, eval_return = get_data(os.path.join(log_dir, file, log_file), "eval_return")
            axes.plot(step, eval_return, label = "eval", color = "blue")
            step, train_return = get_data(os.path.join(log_dir, file, log_file), "train_return")
            axes.plot(step, train_return, label = "train", color = "red", alpha = 0.4)
            axes.set_xlabel("Step")
            axes.set_ylabel("eval_return")
                
    axes.legend()
    plt.show()

def plot_experiment_five() -> None:
    fig, axes = plt.subplots(nrows=1, ncols=1)
    files = os.listdir(log_dir)
    
    for file in files:
        if ("HalfCheetah" not in file):
            continue
        
        log_files = os.listdir(os.path.join(log_dir, file))

        tag = "reinforce10"
        if ("reinforce1" in file and "reinforce10" not in file):
            tag = "reinforce1"
        elif "reparametrize" in file:
            tag = "reparametrize"
        
        for log_file in log_files:
            step, eval_return = get_data(os.path.join(log_dir, file, log_file), "eval_return")
            axes.plot(step, eval_return, label=tag)

    axes.set_xlabel("Step")
    axes.set_ylabel("eval_return")
    axes.legend()
    plt.show()

def plot_experiment_six() -> None:
    fig, axes = plt.subplots(nrows=1, ncols=1)
    files = os.listdir(log_dir)
    
    for file in files:
        if ("humanoid" not in file):
            continue
        
        log_files = os.listdir(os.path.join(log_dir, file))
        
        for log_file in log_files:
            step, eval_return = get_data(os.path.join(log_dir, file, log_file), "eval_return")
            axes.plot(step, eval_return)

    axes.set_xlabel("Step")
    axes.set_ylabel("eval_return")
    axes.legend()
    plt.show()

if __name__ == "__main__":
    # plot_experiment_one()
    # plot_experiment_two()
    # plot_experiment_three()
    # plot_experiment_four()
    # plot_experiment_five() 
    plot_experiment_six()