from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os

log_dir = "F:\CS285\homework_fall2023\hw2\data"

def get_data(path:str, key:str) :
    acc = EventAccumulator(path)
    acc.Reload()
    step = []
    ret = []
    for event in acc.Scalars("Train_EnvstepsSoFar"):
        step.append(event.value)
    
    for event in acc.Scalars(key):
        ret.append(event.value)

    return step, ret

def plot_experiment_one():
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    files = os.listdir(log_dir)
    
    info_dict_sb = {"ori" : {"xs" : [], "ys": []}, 
                 "rtg" :{"xs":[], "ys" : []}, 
                 "na" : {"xs" : [], "ys": []}, 
                 "rtg_na" : {"xs" : [], "ys": []}}
    info_dict_lb = {"ori" : {"xs" : [], "ys": []}, 
                "rtg" :{"xs":[], "ys" : []}, 
                "na" : {"xs" : [], "ys": []}, 
                "rtg_na" : {"xs" : [], "ys": []}}
    
    for file in files:
        if "CartPole" not in file:
            continue
        # small batch
        if "lb" not in file:
            log_files = os.listdir(log_dir + "/"+file)
            for log_file in log_files:
                tag = "ori"
                print(log_file)
                for k, v in info_dict_sb.items():
                    if k in file and "rtg_na" not in file:
                        tag = k
                    if "rtg_na" in file:
                        tag = "rtg_na"
                info_dict_sb[tag]["xs"], info_dict_sb[tag]["ys"] = get_data(log_dir + "/"+ file + "/" + log_file, "Eval_AverageReturn")
                axes[0].plot(info_dict_sb[tag]["xs"], info_dict_sb[tag]["ys"], label=tag)
        # large batch
        else:
            log_files = os.listdir(log_dir + "/"+file)
            for log_file in log_files:
                tag = "ori"
                print(log_file)
                for k, v in info_dict_lb.items():
                    if k in file and "rtg_na" not in file:
                        tag = k
                    if "rtg_na" in file:
                        tag = "rtg_na"
                info_dict_lb[tag]["xs"], info_dict_lb[tag]["ys"] = get_data(log_dir + "/"+ file + "/" + log_file, "Eval_AverageReturn")
                axes[1].plot(info_dict_lb[tag]["xs"], info_dict_lb[tag]["ys"], label=tag)
                
    axes[0].set_title("Small Batch EvalAvergeReturn vs TrainStepsSoFar")
    axes[1].set_title("Large Batch EvalAvergeReturn vs TrainStepsSoFar")
    axes[0].legend()
    axes[1].legend()
    
    plt.tight_layout()
    plt.ylabel("Average Return")
    plt.xlabel("Training Steps")
    plt.show()

def plot_experiment_two():
    files = os.listdir(log_dir)
    _, axes = plt.subplots(nrows=1, ncols=2)
    
    info_dict = {"ori" : {"xs" : [], "ys": []}, 
                 "baseline" :{"xs":[], "ys" : []}}
    
    for file in files:
        if "cheetah" not in file:
            continue
        # baseline version
        if "baseline" in file:
            tag = "baseline"
            log_files = os.listdir(log_dir + "/"+file)
            for log_file in log_files:
                info_dict[tag]["xs"], info_dict[tag]["ys"] = get_data(log_dir + "/"+ file + "/" + log_file, "Baseline_Loss")
                axes[0].plot(info_dict[tag]["xs"], info_dict[tag]["ys"], label=tag)
                info_dict[tag]["xs"], info_dict[tag]["ys"] = get_data(log_dir + "/"+ file + "/" + log_file, "Eval_AverageReturn")
                axes[1].plot(info_dict[tag]["xs"], info_dict[tag]["ys"], label=tag)
        else:
            tag = "ori"
            log_files = os.listdir(log_dir + "/"+file)
            for log_file in log_files:
                info_dict[tag]["xs"], info_dict[tag]["ys"] = get_data(log_dir + "/"+ file + "/" + log_file, "Eval_AverageReturn")
                axes[1].plot(info_dict[tag]["xs"], info_dict[tag]["ys"], label=tag)
                
                
    axes[0].set_title("BaseLine Experiment")
    axes[1].set_title("Eval Average Return")
    axes[1].legend()
    
    plt.tight_layout()
    axes[0].set_ylabel("Baseline_Loss")
    axes[1].set_ylabel("Eval_AverageReturn")
    plt.xlabel("Training Steps")
    plt.show()
    
def plot_experiment_three():
    files = os.listdir(log_dir)
    
    info_dict = {"xs" : [], "ys": []}
    tags = ["lambda0_", "lambda0.95_", "lambda0.98_", "lambda0.99_", "lambda1_"]
    
    for file in files:
        if "lunar_lander" not in file:
            continue
        log_files = os.listdir(log_dir + "/"+file)
        for tag in tags:
            if tag in file:
                for log_file in log_files:
                    info_dict["xs"], info_dict["ys"] = get_data(log_dir + "/"+ file + "/" + log_file, "Eval_AverageReturn")
                    plt.plot(info_dict["xs"], info_dict["ys"], label = tag[:-1])
                
    info_dict["ys"] = [180 for _ in info_dict["xs"]]
    plt.plot(info_dict["xs"], info_dict["ys"], label = "Over180")
                
    plt.tight_layout()
    plt.legend()
    plt.ylabel("Eval_AverageReturn")
    plt.xlabel("Training Steps")
    plt.show()

if __name__ == "__main__":
    plot_experiment_three()