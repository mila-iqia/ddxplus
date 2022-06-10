from env import *
from agent import *
from metrics import compute_metrics, write_json
import copy
import torch
import pickle
import time
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
import argparse

from orion.client import report_results

try:
  import wandb 
  wandb_flag = True 
except:
  wandb_flag = False
  try:
      import mlflow
      mlflow_flag = True
  except:
      mlflow_flag = False

def init_logging(time_stamp, args):
    if wandb_flag:
        wandb.init(name=time_stamp, group="aarlc", project="medical_evidence_collection")
        wandb.config.update(args)
    elif mlflow_flag:
        mlflow.set_experiment(experiment_name=args.exp_name)
        mlflow.start_run()
        args = vars(args) if hasattr(args, "__dict__") else args
        for k in args:
            mlflow.log_param(k, args.get(k))
    else:
        pass


def log_metric_data(result):
    if wandb_flag:
        wandb.log(result)
    elif mlflow_flag:
        batch = result.pop("batch", 0) if "batch" in result else result.pop("epoch", 0)
        mlflow.log_metrics(result, batch)
    else:
        pass

def main():
    print("Initializing Environment and generating Patients....")
    env = environment(args, args.train_data_path, train=True)
    print(f"Training environment size: {env.sample_size}")
    patience = args.patience
    best_val_accuracy = None
    if args.eval_on_train_epoch_end:
        eval_env = environment(args, args.val_data_path, train=False)
        print(f"Validation environment size: {eval_env.sample_size}")
    agent = Policy_Gradient_pair_model(state_size = env.state_size, disease_size = env.diag_size, symptom_size= env.symptom_size, LR = args.lr, Gamma = args.gamma)
    threshold_list = []
    best_a = 0
    if args.threshold_random_initial:
        threshold = np.random.rand(env.diag_size)
    else:
        threshold = args.threshold * np.ones(env.diag_size)

    step_idx = 0
    best_epoch_train_accuracy = 0
    for epoch in range(args.EPOCHS):
        env.reset()
        agent.train()
        num_batches = env.sample_size // args.batch_size
        steps_on_ave = 0
        pos_on_ave = 0
        accu_on_ave = 0
        for batch_idx in tqdm(range(num_batches), total=num_batches, desc=f"epoch {epoch}: "):
            step_idx += 1
            states = []
            action_m = []
            rewards_s = []
            action_s = []
            true_d = []
            true_diff_ind = []
            true_diff_prob = []

            s, true_disease, true_diff_indices, true_diff_probas, _ = env.initialize_state(args.batch_size)
            s_init = copy.deepcopy(s)
            s_final = copy.deepcopy(s)

            a_d, p_d = agent.choose_diagnosis(s)
            init_ent = entropy(p_d, axis = 1)
            
            done = (init_ent < threshold[a_d])
            right_diag = (a_d == env.disease) & done

            diag_ent = np.zeros(args.batch_size)
            finl_diag = np.zeros(args.batch_size).astype(int) - 1
            diag_ent[right_diag] = init_ent[right_diag]
            ent = init_ent

            for i in range(args.MAXSTEP):
                a_s = agent.choose_action_s(s)
                
                s_, r_s, done, right_diag, final_idx, ent_, a_d_ = env.step(s, a_s, done, right_diag, agent, init_ent, threshold, ent)
                s_final[final_idx] = s_[final_idx]
                diag_ent[right_diag] = ent_[right_diag]
                finl_diag[right_diag] = a_d_[right_diag]
                # print(max(finl_diag[right_diag]))
                # print(max(a_d_[right_diag]))
                # print(finl_diag[right_diag])
                # print(a_d_[right_diag])
                # input()
                if i == (args.MAXSTEP - 1):
                    r_s[~done] += 1

                states.append(s)
                rewards_s.append(r_s)
                action_s.append(a_s)
                true_d.append(true_disease)
                true_diff_ind.append(true_diff_indices)
                true_diff_prob.append(true_diff_probas)
                
                s = s_
                ent = ent_
                
                if all(done):
                    break
            
            diag = np.sum(done)
            s_final[~done] = s_[~done]

            _, all_step, ave_reward_s = agent.create_batch(states, rewards_s, action_s, true_d, true_diff_ind, true_diff_prob)
            a_d, p_d = agent.choose_diagnosis(s)
            
            t_d = (a_d == env.disease) & (~done)
            diag_ent[t_d] = entropy(p_d[t_d], axis = 1)
            finl_diag[t_d] = a_d[t_d]
            # print(max(finl_diag))
            for idx, item in enumerate(finl_diag):
                if item >= 0 and abs(threshold[item] - diag_ent[idx]) > 0.01:
                    threshold[item] = (args.lamb * threshold[item] + (1-args.lamb) * diag_ent[idx])   #update the threshold

            agent.update_param_rl()
            agent.update_param_c()
            
            accuracy = (sum(right_diag)+sum(t_d))/(args.batch_size)
            best_a = np.max([best_a, accuracy])

            ave_pos = np.sum(env.inquired_symptoms * env.all_state) / max(1, all_step)
            ave_step = all_step / args.batch_size

            threshold_list.append(threshold)

            print("==Epoch:", epoch+1, '\tAve. Accu:', accuracy, '\tBest Accu:', best_a, '\tAve. Pos:', ave_pos)
            print('Threshold:', threshold[:5], '\tAve. Step:', ave_step, '\tAve. Reward Sym.:', ave_reward_s, '\n')
            
            steps_on_ave = batch_idx / (batch_idx + 1) * steps_on_ave + 1 / (batch_idx + 1) * ave_step
            pos_on_ave = batch_idx / (batch_idx + 1) * pos_on_ave + 1 / (batch_idx + 1) * ave_pos
            accu_on_ave = batch_idx / (batch_idx + 1) * accu_on_ave + 1 / (batch_idx + 1) * accuracy
            
            # wandb logging
            results_dict = {
                "accuracy/train": accuracy,
                "best_accuracy/train": best_a,
                "average_pos/train": ave_pos,
                "average_step/train": ave_step,
                "average_symptom_reward/train": ave_reward_s,
                "epoch": epoch,
                "batch": step_idx - 1,
            }
            log_metric_data(results_dict)

        # wandb logging
        best_epoch_train_accuracy = max(accu_on_ave, best_epoch_train_accuracy)
        results_dict = {
            "epoch_accuracy/train": accu_on_ave,
            "epoch_best_accuracy/train": best_epoch_train_accuracy,
            "epoch_average_pos/train": pos_on_ave,
            "epoch_average_step/train": steps_on_ave,
            "epoch": epoch,
        }
        print("==Epoch:", epoch+1, '\tAve. EpochAccu:', accu_on_ave, '\tBest EpochAccu:', best_epoch_train_accuracy, '\tAve. EpochPos:', pos_on_ave)
        print('EpochThreshold:', threshold[:5], '\tAve. EpochStep:', steps_on_ave, '\n')
        log_metric_data(results_dict)

        agent.save_model(args)
        info = str(args.dataset) + '_' + str(args.threshold) + '_' + str(args.mu) + '_' + str(args.nu) + '_' + str(args.trail)
        with open(f'{args.save_dir}/threshold_changing_curve_'+info+'.pkl', 'wb') as f:
            pickle.dump(threshold_list, f)

        if args.eval_on_train_epoch_end:
            val_result = test(agent=agent, threshold=threshold, epoch=epoch, env=eval_env, log_flag=False)
            val_accuracy = val_result["epoch_accuracy/validation"]
            if best_val_accuracy is None or val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                agent.save_model(args, prefix="best_")
                patience = args.patience
            else:
                patience -= 1
            val_result["epoch_best_accuracy/validation"] = best_val_accuracy
            log_metric_data(val_result)
            if patience == 0:
                break
        else:
            if epoch == args.EPOCHS - 1:
                val_result = test(agent=agent, threshold=threshold, log_flag=True)
                best_val_accuracy = val_result["epoch_accuracy/validation"]

    report_results([dict(name="dev_metric", type="objective", value=-float(best_val_accuracy))])

@torch.no_grad()
def test(agent=None, threshold=None, epoch=0, env=None, log_flag=False):
    print("Initializing Environment and generating Patients....")
    if env is None:
        env = environment(args, args.val_data_path, train=False)
        print(f"Validation environment size: {env.sample_size}")
    if agent is None:
        agent = Policy_Gradient_pair_model(state_size = env.state_size, disease_size = env.diag_size, symptom_size= env.symptom_size, LR = args.lr, Gamma = args.gamma)
        agent.load_model(args, prefix=args.model_prefix)

    agent.eval()

    if threshold is None:
        info = str(args.dataset) + '_' + str(args.threshold) + '_' + str(args.mu) + '_' + str(args.nu) + '_' + str(args.trail)
        with open(f'{args.checkpoint_dir}/threshold_changing_curve_'+info+'.pkl', 'rb') as f:
            bf = pickle.load(f)
            threshold = bf[args.eval_epoch if args.eval_epoch is not None else -1]

    steps_on_ave = 0
    pos_on_ave = 0
    accu_on_ave = 0

    num_batches = env.sample_size // args.eval_batch_size
    epoch_metrics = {}
    env.reset()
    for batch_idx in tqdm(range(num_batches), total=num_batches):
        states = []
        action_m = []
        rewards_s = []
        action_s = []
        true_d = []
        true_diff_ind = []
        true_diff_prob = []


        s, true_disease, true_diff_indices, true_diff_probas, _  = env.initialize_state(args.eval_batch_size)
        s_init = copy.deepcopy(s)
        s_final = copy.deepcopy(s)

        a_d, p_d = agent.choose_diagnosis(s)
        init_ent = entropy(p_d, axis = 1)
        
        done = (init_ent < threshold[a_d])
        right_diag = (a_d == env.disease) & done

        diag_ent = np.zeros(args.eval_batch_size)
        diag_ent[right_diag] = init_ent[right_diag]
        ent = init_ent
        
        if args.compute_eval_metrics:
            preds = [p_d]

        for i in range(args.MAXSTEP):

            a_s = agent.choose_action_s(s, args.deterministic)
            s_, r_s, done, right_diag, final_idx, ent_, a_d_ = env.step(s, a_s, done, right_diag, agent, init_ent, threshold, ent)

            s_final[final_idx] = s_[final_idx]
            # diag_ent[right_diag] = ent_[right_diag]

            if i == args.MAXSTEP - 1:
                r_s[done==False] -= 1

            states.append(s)
            rewards_s.append(r_s)
            action_s.append(a_s)
            true_d.append(true_disease)
            true_diff_ind.append(true_diff_indices)
            true_diff_prob.append(true_diff_probas)
            
            if args.compute_eval_metrics:
                preds.append(agent.choose_diagnosis(s_)[1])
            
            s = s_
            ent = ent_
            
            if all(done):
                break

        diag = np.sum(done)
        s_final[~done] = s_[~done]
        
        if args.compute_eval_metrics:
            _, final_diagnosis = agent.choose_diagnosis(s_final)
            
        valid_timesteps, all_step, ave_reward_s = agent.create_batch(states, rewards_s, action_s, true_d, true_diff_ind, true_diff_prob)
        a_d, p_d = agent.choose_diagnosis(s)
        finl_ent = entropy(p_d, axis = 1)
        t_d = (a_d == env.disease) & (~done)
        # diag_ent[t_d] = finl_ent[t_d]
        
        ave_step = all_step / args.eval_batch_size
        ave_pos = np.sum(env.inquired_symptoms * env.all_state) / max(1, all_step)
        accurate = (sum(right_diag) + sum(t_d)) / args.eval_batch_size

        steps_on_ave = batch_idx / (batch_idx + 1) * steps_on_ave + 1 / (batch_idx + 1) * ave_step
        pos_on_ave = batch_idx / (batch_idx + 1) * pos_on_ave + 1 / (batch_idx + 1) * ave_pos
        accu_on_ave = batch_idx / (batch_idx + 1) * accu_on_ave + 1 / (batch_idx + 1) * accurate
        print('Ave. Step: ', steps_on_ave, '\tAve. Pos: ', pos_on_ave, '\tAve. Accu: ', accu_on_ave)
        # wandb logging
        results_dict = {
            "accuracy/validation": accu_on_ave,
            "average_pos/validation": pos_on_ave,
            "average_step/validation": steps_on_ave,
            "epoch": epoch,
            "batch": epoch * num_batches + batch_idx,
        }
        log_metric_data(results_dict)

        if args.compute_eval_metrics:
            preds = np.array(preds).swapaxes(0, 1)
            batch_metrics = compute_metrics(
                env.target_differential, true_disease, final_diagnosis, preds, valid_timesteps,
                env.all_state, env.inquired_symptoms, env.symptom_mask, env.atcd_mask, env.severity_mask, tres=0.01
            )
            epoch_metrics = {a: (batch_idx / (batch_idx + 1)) * epoch_metrics.get(a, 0) + (1 / (batch_idx + 1)) * batch_metrics[a] for a in batch_metrics.keys()}
        

    results_dict = {
        "epoch_accuracy/validation": accu_on_ave,
        "epoch_average_pos/validation": pos_on_ave,
        "epoch_average_step/validation": steps_on_ave,
        "epoch": epoch,
    }
    if args.compute_eval_metrics:
        epoch_metrics = {a: epoch_metrics[a].tolist() if hasattr(epoch_metrics[a], "tolist") else epoch_metrics[a] for a in epoch_metrics.keys()}
        results_dict.update(epoch_metrics)
    if log_flag:
        log_metric_data(results_dict)
    return results_dict

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Process Settings')
    parser.add_argument('--dataset', type=str, default = 'casande',
                        help='Name of the dataset')
    parser.add_argument('--seed', type=int, default = 42,
                        help='set a random seed')
    parser.add_argument('--threshold', type=float, default = 1,
                        help='set a initial threshold')
    parser.add_argument('--threshold_random_initial', action="store_true",
                        help='randomly initialize threshold')
    parser.add_argument('--batch_size', type=int, default = 200,
                        help='batch_size for each time onpolicy sample collection')
    parser.add_argument('--eval_batch_size', type=int, default = 0,
                        help='batch_size for each time onpolicy evaluation')
    parser.add_argument('--eval_on_train_epoch_end', action="store_true",
                        help='evaluate at the end of each epoch')
    parser.add_argument('--EPOCHS', type=int, default = 100,
                        help='training epochs')
    parser.add_argument('--MAXSTEP', type=int, default = 30,
                        help='max inquiring turns of each MAD round')
    parser.add_argument('--patience', type=int, default = 10,
                        help='patience')
    parser.add_argument('--nu', type=float, default = 2.5,
                        help='nu')
    parser.add_argument('--mu', type=float, default = 1,
                        help='mu')
    parser.add_argument('--lr', type=float, default = 1e-4,
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default = 0.99,
                        help='reward discount rate')
    parser.add_argument('--train', action="store_true",
                        help='whether test on the exsit result model or train a new model')
    parser.add_argument('--trail', type=int, default = 1)
    parser.add_argument('--eval_epoch', type=int, default = None, help='the epoch to use for evaluation')
    parser.add_argument('--lamb', type=float, default = 0.99,
                        help='polyak factor for threshold adjusting')
    parser.add_argument('--exp_name', type=str, default='EfficientRL', help='Experience Name')
    parser.add_argument('--save_dir', type=str, default='./output', help='directory to save the results')
    parser.add_argument('--checkpoint_dir', type=str, help='directory containing the checkpoints to restore')
    parser.add_argument('--train_data_path', type=str, required=True, help='path to the training data file')
    parser.add_argument('--val_data_path', type=str, required=True, help='path to the validation data file')
    parser.add_argument('--evi_meta_path', type=str, required=True, help='path to the evidences (symptoms) meta data',
                        default = './release_evidences.json'
    )
    parser.add_argument('--patho_meta_path', type=str, required=True, help='path to the pathologies (diseases) meta data',
                        default = './release_conditions.json'
    )
    parser.add_argument('--include_turns_in_state', action="store_true", help='whether to include turns on state')
    parser.add_argument('--date_time_suffix', action="store_true", help='whether to add time stamp suffix on the specified save_dir forlder')
    parser.add_argument('--no_differential', action="store_true", help='whether to not use differential')
    parser.add_argument('--no_initial_evidence', action="store_true", help='whether to not use the given initial evidence but randomly select one')
    parser.add_argument('--compute_eval_metrics', action="store_true", help='whether to compute custom evaluation metrics')
    parser.add_argument('--deterministic', action="store_true", help='deterministic evaluation')
    parser.add_argument('--prefix', type=str, default='', help='prefix to be added to the saved metric file.')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix to be added to the model to be loaded.')
    
    args = parser.parse_args()
    if args.eval_batch_size == 0:
        args.eval_batch_size = args.batch_size
    if args.eval_epoch is None:
        args.eval_epoch = -1
    
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # setup wandb
    time_stamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    if args.train:
        # add save_dir
        if args.date_time_suffix:
            args.save_dir = os.path.join(args.save_dir, time_stamp)
        os.makedirs(args.save_dir)

    init_logging(time_stamp, args)
    
    if args.train:        
        main()
    else:
        eval_metrics = test()
        write_json(eval_metrics, f"{args.checkpoint_dir}/EffRlMetrics_{args.dataset.lower()}{args.prefix}.json")
