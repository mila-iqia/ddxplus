import os

import mlflow
import numpy as np
import torch
import random
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from asd_model import ASDMLP
from dataset import ASDPatientInteractionSimulator, SimPaDatasetTrain as SimPaDataset

from orion.client import report_results


BEST_PRETRAIN_MODEL_NAME = "asd_best_model_params.pkl"


def initialize_seed(seed):
    """Method for seeding the random generators.

    Parameters
    ----------
    seed: int
        the seed to be used.

    Returns
    -------
    None

    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def softXEnt(input, target, dim=-1, reduction='mean', weight=None):
    """Adapted from this pytorch discussion link.
       https://discuss.pytorch.org/t/
       soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501/2

       Per definition, we have (https://en.wikipedia.org/wiki/Cross_entropy):
           CE(p,q) = -(p * log(q)).sum()
       With a provided weight per class, the computation becomes:
           CE(p,q,w) = -(p * log(q)).sum() * (p * w).sum()
    """
    tmp_weight = 1.0 if weight is None else weight.unsqueeze(0)
    logprobs = F.log_softmax(input, dim=dim)
    avg_logprobs = (target * logprobs).sum(dim=dim)
    avg_weight = 1.0 if weight is None else (target * tmp_weight).sum(dim=dim)
    result = -(avg_logprobs * avg_weight)
    flag = weight is None
    if reduction == 'mean':
        result = result.mean() if flag else result.sum() / avg_weight.sum()
    elif reduction == 'sum':
        result = result.sum()
    return result


def dist_accuracy(
    pred,
    target,
    target_indices,
    target_probas,
    k,
    restrict_to_k=False,
    ignore_index=-1,
    reduction='mean',
):
    """Computes the fraction of the predicted topk classes in the target distribution.

    Here, we can have dirac distribution (`target`) or soft labels defined through
    the parameters `target_indices` and `target_probas`. They respectively represent
    the class indices involved in the target distribution and their corresponding
    probability. The provided `ignore_index` can be used as padding element in the
    `target_indices` field.

    Parameters
    ----------
    pred: np.array
        an array of size `C` where C is the number of classes.
        This tensor represents the logit values.
    target: int, np.array
        the target indice. It is used only if the soft distribution is None. That is,
        `target_indices` or `target_probas` are None.
    target_indices: np.array
        an array of size `D` where D <= C is the number of classes
        present in the soft distribution.
    target_probas: np.array
        a tensor of same size as `target_indices` representing the probability
        associated to each class therein.
    k: int
        The number of top element to be considered in the predicted distribution.
    restrict_to_k: bool
        Flag indicating whether or not the differential should be restricted to the topk
        values for the metric computation. Default: False
    ignore_index: int
        Specifies a target value that is ignored and does not contribute
        to the computation. Default: -1

    Return
    ----------
    result: float
        the computed fraction.

    """

    def tmp_f(a, kmin, ignore_index):
        top, ind = a[:kmin], a[kmin:]
        # mask target indices
        msk = ind != ignore_index
        # find the intersection
        rslt = np.intersect1d(top, ind)
        # compute the fraction wrt to min(kmin, len(msk))
        return 0.0 if msk.sum() == 0 else len(rslt) / min(kmin, msk.sum())

    # if not differential based, create a differential with the right number
    if (target_indices is None) or (target_probas is None):
        target_indices = np.array([target], dtype=int).reshape((-1, 1))
        target_probas = np.ones(target_indices.shape, dtype=np.float32)

    assert ignore_index < 0
    target_indices = target_indices.reshape((-1, target_indices.shape[-1]))
    target_probas = target_probas.reshape(target_indices.shape)

    if restrict_to_k:
        # sort target indices
        s_ind = np.argsort(target_probas, axis=-1)
        s_target_probas = np.take_along_axis(target_probas, s_ind[:, ::-1], axis=-1)
        s_target_indices = np.take_along_axis(target_indices, s_ind[:, ::-1], axis=-1)
        min_l = min(s_target_probas.shape[-1], k)
        target_probas = s_target_probas[:, 0:min_l]
        target_indices = s_target_indices[:, 0:min_l]

    # get the topk indices from pred distribution
    topk = np.argsort(pred, axis=-1)[..., -k:]
    kmin = min(k, pred.shape[-1])
    topk = topk.reshape((-1, kmin))

    assert (
        topk.shape[0] == target_indices.shape[0]
    ), f"{topk.shape} - {target_indices.shape} - {k} - {pred.shape}"

    merge_arr = np.concatenate((topk, target_indices), axis=1)

    # apply along axis
    result = np.apply_along_axis(tmp_f, 1, merge_arr, kmin, ignore_index)
    return np.mean(result) if reduction == 'mean' else result


def create_environments(args):
    """Method for creating the environments for pretraining.

    Parameters
    ----------
    args: dict
        the arguments as provided in the command line.

    Return
    ------
    envs: list
        list of instantiated environments.

    """
    env_list = [ASDPatientInteractionSimulator(args, patient_filepath=args.data, train=True)]
    print(f"Training environment size: {env_list[-1].sample_size}")

    is_eval_off = args.eval_data is None or args.eval_data == args.data
    if not is_eval_off:
        env_list.append(ASDPatientInteractionSimulator(args, patient_filepath=args.eval_data, train=False))
        print(f"Eval environment size: {env_list[-1].sample_size}")

    if is_eval_off:
        env_list.append(None)

    return env_list


def create_datasets(env_train, env_valid, args):
    """Method for creating the training and valid datasets.

    Parameters
    ----------
    env_train: object
        the enviroment to be used for training.
    env_valid: object
        the enviroment to be used for validation.
    args: dict
        the arguments as provided in the command line.

    Return
    ------
    train_ds: dataset
        the training dataset.
    valid_ds: dataset
        the valid dataset.

    """
    ds_train = SimPaDataset(
        env_train, args.interaction_length, args.min_rec_ratio, args.only_diff_at_end
    )
    ds_valid = None
    if env_valid is not None:
        ds_valid = SimPaDataset(
            env_valid,
            args.interaction_length,
            args.min_rec_ratio,
            args.only_diff_at_end
        )
    elif not ((args.valid_percentage is None) or (args.valid_percentage == 0)):
        valid_size = int(len(ds_train) * args.valid_percentage)
        train_size = len(ds_train) - valid_size
        if valid_size > 0:
            ds_train, ds_valid = torch.utils.data.random_split(
                ds_train, [train_size, valid_size],
            )
    return ds_train, ds_valid


def get_predictions(logits):
    pred_info = torch.argmax(logits, dim=1).view(-1)
    pred_info = pred_info.cpu().numpy()
    return pred_info


def train_epoch(epoch, agent, optimizer, train_dl, args):
    """Train Epoch for the pretraining process.

    Parameters
    ----------
    epoch: int
        the epoch number.
    agent: agent
        the agent whose classifier should be pretrained.
    optimizer: optimizer
        the optimizer to be used.
    train_dl: dataloader
        the training dataloader.
    metric_factory: object
        the metric factory.
    args: dict
        the arguments as provided in the command line.

    Return
    ------
    avg_loss: float
        the obtained average loss value.
    avg_metric: float
        the obtained average performance metric value.

    """
    avg_loss_sym = 0.0
    avg_loss_ag = 0.0
    avg_loss_patho = 0
    avg_metric_ag = None
    avg_metric_sym = None
    avg_metric_patho = None
    num_elts_ag = 0
    num_elts_sym = 0
    agent.train(True)
    # assert train_dl.concatenate
    for _, (x_ag, ag_gt, x_sym, sym_gt, is_diff, diff) in tqdm(enumerate(train_dl), total=len(train_dl), desc=f"epoch {epoch}: "):
        num_elts_ag += x_ag.shape[0]
        num_elts_sym += x_sym.shape[0]
        num_diff = is_diff.sum().item()
        odiff = diff
        ois_diff = is_diff
        shape = [diff.shape[i] for i in range(len(diff.shape))]
        o_ind = np.ones(shape) * np.array(list(range(diff.shape[-1])))

        x_ag, ag_gt, x_sym, sym_gt, is_diff, diff = (
            x_ag.float().to(args.device),
            ag_gt.float().to(args.device),
            x_sym.float().to(args.device),
            sym_gt.long().to(args.device),
            is_diff.float().to(args.device),
            diff.float().to(args.device),
        )
        optimizer.zero_grad()
        # import pdb;pdb.set_trace()
        sym_pred, ag_pred, patho_pred = agent(x_sym, x_ag)

        # compute loss
        loss_ag = F.binary_cross_entropy_with_logits(
            ag_pred, ag_gt, weight=None, reduction="mean"
        )
        loss_sym = F.cross_entropy(
            sym_pred, sym_gt.view(-1), weight=None, reduction="mean"
        )
        loss_patho = (
            0.0
            if patho_pred is None
            else (
                softXEnt(patho_pred, diff, reduction='none') * is_diff.squeeze(1)
            ).sum() / max(1, num_diff)
        )
        loss = loss_ag + loss_sym + loss_patho
        loss.backward()

        # optimize
        optimizer.step()
        avg_loss_ag += loss_ag.item() * x_ag.shape[0]
        avg_loss_sym += loss_sym.item() * x_sym.shape[0]
        avg_loss_patho += (
            loss_patho.item() if hasattr(loss_patho, 'item') else loss_patho
        ) * x_ag.shape[0]
        if args.metric is not None:
            avg_metric_ag = 0.0 if avg_metric_ag is None else avg_metric_ag
            avg_metric_sym = 0.0 if avg_metric_sym is None else avg_metric_sym
            avg_metric_patho = 0.0 if avg_metric_patho is None else avg_metric_sym
            ag_pred = torch.sigmoid(ag_pred).view(-1).detach().cpu().numpy().flatten()
            # import pdb;pdb.set_trace()
            ag_pred[ag_pred < args.thres] = 0
            ag_pred[ag_pred >= args.thres] = 1

            sym_pred = get_predictions(sym_pred)
            # import pdb;pdb.set_trace()
            ag_acc = accuracy_score(ag_gt.view(-1).detach().cpu().numpy(), ag_pred)
            sym_acc = accuracy_score(sym_gt.view(-1).detach().cpu().numpy(), sym_pred)
            avg_metric_sym += sym_acc * x_sym.shape[0]
            avg_metric_ag += ag_acc * x_ag.shape[0]

            if patho_pred is None:
                avg_metric_patho += 0
            else:
                tmp_is_diff = ois_diff.view(-1).cpu().numpy()
                tmp_val = dist_accuracy(
                    patho_pred.detach().cpu().numpy(),
                    None,
                    o_ind,
                    odiff.cpu().numpy(),
                    5,
                    reduction='none'
                ) * tmp_is_diff
                avg_metric_patho += tmp_val.sum() / max(1, tmp_is_diff.sum())
    avg_loss_sym /= max(1, num_elts_sym)
    avg_loss_ag /= max(1, num_elts_ag)
    avg_loss_patho /= max(1, num_elts_ag)
    if avg_metric_ag is not None:
        avg_metric_ag /= max(1, num_elts_ag)
        avg_metric_sym /= max(1, num_elts_sym)
        avg_metric_patho /= max(1, num_elts_ag)
    else:
        avg_metric_ag = -avg_loss_ag
        avg_metric_sym = -avg_loss_sym
        avg_metric_patho = -avg_loss_patho
    ov = avg_metric_patho
    return avg_loss_ag, avg_metric_ag, avg_loss_sym, avg_metric_sym, avg_loss_patho, ov


def eval_epoch(epoch, agent, eval_dl, args):
    """Train Epoch for the pretraining process.

    Parameters
    ----------
    epoch: int
        the epoch number.
    agent: agent
        the agent whose classifier should be pretrained.
    eval_dl: dataloader
        the validate dataloader.
    metric_factory: object
        the metric factory.
    args: dict
        The arguments as provided in the command line.

    Return
    ------
    avg_loss: float
        the obtained average loss value.
    avg_metric: float
        the obtained average performance metric value.

    """

    with torch.no_grad():
        agent.train(False)
        avg_loss_sym = 0.0
        avg_loss_ag = 0.0
        avg_loss_patho = 0
        avg_metric_ag = None
        avg_metric_sym = None
        avg_metric_patho = None
        num_elts_ag = 0
        num_elts_sym = 0
        for _, (x_ag, ag_gt, x_sym, sym_gt, is_diff, diff) in tqdm(enumerate(eval_dl), total=len(eval_dl), desc=f"Eval epoch {epoch}: "):
            num_elts_ag += x_ag.shape[0]
            num_elts_sym += x_sym.shape[0]
            num_diff = is_diff.sum().item()
            odiff = diff
            ois_diff = is_diff
            shape = [diff.shape[i] for i in range(len(diff.shape))]
            o_ind = np.ones(shape) * np.array(list(range(diff.shape[-1])))

            x_ag, ag_gt, x_sym, sym_gt, is_diff, diff = (
                x_ag.float().to(args.device),
                ag_gt.float().to(args.device),
                x_sym.float().to(args.device),
                sym_gt.long().to(args.device),
                is_diff.float().to(args.device),
                diff.float().to(args.device),
            )

            sym_pred, ag_pred, patho_pred = agent(x_sym, x_ag)

            # compute loss
            loss_ag = F.binary_cross_entropy_with_logits(
                ag_pred, ag_gt, weight=None, reduction="mean"
            )
            loss_sym = F.cross_entropy(
                sym_pred, sym_gt.view(-1), weight=None, reduction="mean"
            )
            loss_patho = (
                0.0
                if patho_pred is None
                else (
                    softXEnt(patho_pred, diff, reduction='none') * is_diff.squeeze(1)
                ).sum() / max(1, num_diff)
            )
            loss = loss_ag + loss_sym + loss_patho
            _ = loss

            avg_loss_ag += loss_ag.item() * x_ag.shape[0]
            avg_loss_sym += loss_sym.item() * x_sym.shape[0]
            avg_loss_patho += (
                loss_patho.item() if hasattr(loss_patho, 'item') else loss_patho
            ) * x_ag.shape[0]
            if args.metric is not None:
                avg_metric_ag = 0.0 if avg_metric_ag is None else avg_metric_ag
                avg_metric_sym = 0.0 if avg_metric_sym is None else avg_metric_sym
                avg_metric_patho = 0.0 if avg_metric_patho is None else avg_metric_sym
                ag_pred = (
                    torch.sigmoid(ag_pred).view(-1).detach().cpu().numpy().flatten()
                )
                # import pdb;pdb.set_trace()
                ag_pred[ag_pred < args.thres] = 0
                ag_pred[ag_pred >= args.thres] = 1

                sym_pred = get_predictions(sym_pred)
                # import pdb;pdb.set_trace()
                ag_acc = accuracy_score(ag_gt.view(-1).detach().cpu().numpy(), ag_pred)
                sym_acc = accuracy_score(
                    sym_gt.view(-1).detach().cpu().numpy(), sym_pred
                )
                avg_metric_sym += sym_acc * x_sym.shape[0]
                avg_metric_ag += ag_acc * x_ag.shape[0]

                if patho_pred is None:
                    avg_metric_patho += 0
                else:
                    tmp_is_diff = ois_diff.view(-1).cpu().numpy()
                    tmp_val = dist_accuracy(
                        patho_pred.detach().cpu().numpy(),
                        None,
                        o_ind,
                        odiff.cpu().numpy(),
                        5,
                        reduction='none'
                    ) * tmp_is_diff
                    avg_metric_patho += tmp_val.sum() / max(1, tmp_is_diff.sum())
        avg_loss_sym /= max(1, num_elts_sym)
        avg_loss_ag /= max(1, num_elts_ag)
        avg_loss_patho /= max(1, num_elts_ag)
        if avg_metric_ag is not None:
            avg_metric_ag /= max(1, num_elts_ag)
            avg_metric_sym /= max(1, num_elts_sym)
            avg_metric_patho /= max(1, num_elts_ag)
        else:
            avg_metric_ag = -avg_loss_ag
            avg_metric_sym = -avg_loss_sym
            avg_metric_patho = -avg_loss_patho

    # import pdb;pdb.set_trace()
    ov = avg_metric_patho
    return avg_loss_ag, avg_metric_ag, avg_loss_sym, avg_metric_sym, avg_loss_patho, ov


def eval_epoch2(epoch, agent, eval_dl, args):
    """Train Epoch for the pretraining process.

    Parameters
    ----------
    epoch: int
        the epoch number.
    agent: agent
        the agent whose classifier should be pretrained.
    eval_dl: dataloader
        the validate dataloader.
    metric_factory: object
        the metric factory.
    args: dict
        The arguments as provided in the command line.

    Return
    ------
    avg_loss: float
        the obtained average loss value.
    avg_metric: float
        the obtained average performance metric value.

    """

    with torch.no_grad():
        agent.train(False)
        result = eval_dl.evaluate(
            agent,
            args.interaction_length,
            args.num_valid_trajs,
            None,
            args.seed,
            args.cuda_idx,
            args.thres,
            args.masked_inquired_actions,
            args.compute_metrics_flag,
            args.eval_batch_size,
        )
    # import pdb;pdb.set_trace()
    return result["average_evidence_recall"], result["average_step"]


def asd_classifier(agent, optimizer, train_ds, eval_ds, args):
    """Pretrain the agent classifier.

    Parameters
    ----------
    agent: agent
        the agent whose classifier should be pretrained.
    optimizer: optimizer
        the optimizer to be used.
    train_ds: dataset
        the training dataset.
    valid_ds: dataset
        the valid dataset.
    args: dict
        the arguments as provided in the command line.

    Return
    ------
    None

    """
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers
    )
    eval_dl = (
        None
        if eval_ds is None
        else DataLoader(eval_ds, batch_size=args.eval_batch_size, num_workers=args.n_workers)
    )

    best_performance = None
    remaining_patience = args.patience
    eval_epoch_interval = 1
    for i in range(args.num_epochs):
        eva_loss_ag, eva_metric_ag, eva_loss_sym, eva_metric_sym = 0, 0, 0, 0
        eva_loss_patho, eva_metric_patho, avg_val_perf = 0, 0, 0
        # for i in range(1):
        agent.train(True)
        [
            avg_loss_ag,
            avg_metric_ag,
            avg_loss_sym,
            avg_metric_sym,
            avg_loss_patho,
            avg_metric_patho,
        ] = train_epoch(
            i, agent, optimizer, train_dl, args
        )
        # import pdb;pdb.set_trace()
        mlflow.log_metric("asd-train-loss-sym", avg_loss_sym, i)
        mlflow.log_metric("asd-train-perf-sym", avg_metric_sym, i)
        mlflow.log_metric("asd-train-loss-ag", avg_loss_ag, i)
        mlflow.log_metric("asd-train-perf-ag", avg_metric_ag, i)
        mlflow.log_metric("asd-train-loss-patho", avg_loss_patho, i)
        mlflow.log_metric("asd-train-perf-patho", avg_metric_patho, i)
        if (eval_dl is not None) and (i % eval_epoch_interval == 0):
            agent.train(False)

            [
                eva_loss_ag,
                eva_metric_ag,
                eva_loss_sym,
                eva_metric_sym,
                eva_loss_patho,
                eva_metric_patho,
            ] = eval_epoch(
                i, agent, eval_dl, args
            )
            # avg_val_perf, il = eval_epoch(i, agent, eval_dl, args)
            mlflow.log_metric("asd-val-loss-sym", eva_loss_sym, i)
            mlflow.log_metric("asd-val-perf-sym", eva_metric_sym, i)
            mlflow.log_metric("asd-val-loss-ag", eva_loss_ag, i)
            mlflow.log_metric("asd-val-perf-ag", eva_metric_ag, i)
            mlflow.log_metric("asd-val-loss-patho", eva_loss_patho, i)
            mlflow.log_metric("asd-val-perf-patho", eva_metric_patho, i)

            avg_val_perf = (eva_metric_ag + eva_metric_sym) / 2.0
            # import pdb;pdb.set_trace()
            mlflow.log_metric("asd-val-avg-perf", avg_val_perf, i)
            if (best_performance is None) or (avg_val_perf > best_performance):
                best_performance = avg_val_perf
                params = dict(
                    itr=i,
                    agent_state_dict=agent.state_dict(),
                    pretrain_optimizer_state_dict=optimizer.state_dict(),
                )
                file_name = os.path.join(args.output, BEST_PRETRAIN_MODEL_NAME)
                torch.save(params, file_name)
                remaining_patience = args.patience
            else:
                remaining_patience = (
                    remaining_patience - 1
                    if args.patience is not None
                    else args.patience
                )
        print(
            f"ASD Epoch {i}: tr_loss_sym: {avg_loss_sym} tr_perf_sym: {avg_metric_sym} "
            f"tr_loss_ag: {avg_loss_ag} tr_perf_ag: {avg_metric_ag} "
            f"tr_loss_patho: {avg_loss_patho} tr_perf_patho: {avg_metric_patho} "
            f"va_loss_sym: {eva_loss_sym} va_perf_sym: {eva_metric_sym} "
            f"va_loss_ag: {eva_loss_ag} va_perf_ag: {eva_metric_ag} "
            f"va_loss_patho: {eva_loss_patho} va_perf_patho: {eva_metric_patho} "
            f"valid_perf: {avg_val_perf}."
        )
        if (remaining_patience is not None) and (remaining_patience < 0):
            break
    if best_performance is None:
        params = dict(
            itr=i,
            agent_state_dict=agent.state_dict(),
            pretrain_optimizer_state_dict=optimizer.state_dict(),
        )
        file_name = os.path.join(args.output, BEST_PRETRAIN_MODEL_NAME)
        torch.save(params, file_name)
    else:
        func_metric = "loss" if not args.metric else args.metric
        print(
            f"End Pretraining Epoch with best recall performance of ({func_metric}): "
            f"{best_performance}."
        )
        value = float(best_performance) if func_metric == "loss" else -float(best_performance)
        report_results([dict(name="dev_metric", type="objective", value=value)])
    # import pdb;pdb.set_trace()


def create_agent(params):
    assert params, "Params cannot be empty."
    model = ASDMLP(
        params.inp_size,
        params.hidden_sizes,
        params.action_state_size,
        params.symptom_prediction_size,
        params.patho_size,
    )
    return model


def asd_inference(agent, args):
    args.data = args.eval_data


def asd_train(args, params=None):
    """Method for pretraining an agent against a gym environment.

    Parameters
    ----------
    args: dict
        The arguments as provided in the command line.
    params: dict
        The parameters as provided in the configuration file.
        Default: None

    Return
    ------
    None

    """

    if params is None:
        params = {}

    assert (args.valid_percentage is None) or (
        (args.valid_percentage >= 0) and (args.valid_percentage < 1)
    )
    assert args.batch_size > 0
    assert args.patience >= 0

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.seed is None:
        args.seed = 1234

    args.metric = "accuracy"

    print(f"Asd training: Using seed [{args.seed}] for this process.")

    args.metric = None if args.metric is None else args.metric.lower()

    # initialize the random generator
    initialize_seed(args.seed)

    env_train, env_valid = create_environments(args)
    args.inp_size = env_train.state_size + 5
    args.symptom_prediction_size = env_train.symptom_size
    args.action_state_size = 1
    args.patho_size = env_train.diag_size if args.patho_size is None else args.patho_size
    

    # instantiate the agent
    args.device = torch.device(
        f"cuda:{args.cuda_idx}"
        if torch.cuda.is_available() and args.cuda_idx is not None
        else "cpu"
    )
    agent = create_agent(args)
    if args.cuda_idx is not None and torch.cuda.is_available():
        agent.to(args.device)

    # get the optimizer info
    optimCls, optim_params = optim.Adam, {}

    # create the datasets
    ds_train, ds_valid = create_datasets(env_train, env_valid, args)
    # print()
    optimizer = optimCls(agent.parameters(), lr=args.lr, **optim_params,)
    
    # train
    asd_classifier(agent, optimizer, ds_train, ds_valid, args)

