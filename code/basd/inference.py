import os
import json

import torch
import numpy as np
import random

from asd import BEST_PRETRAIN_MODEL_NAME, create_agent
from dataset import ASDPatientInteractionSimulator, SimPaDataset

EVAL_FILE = "Metrics.json"

def write_json(data, fp):
    with open(fp, "w") as outfile:
        json.dump(data, outfile, indent=4)

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


def load_model(model, path):
    data = torch.load(path)
    model.load_state_dict(data["agent_state_dict"])
    model.eval()
    return model


def asd_classifier_test(agent, eval_dl, args):
    with torch.no_grad():
        agent.train(False)
        results = eval_dl.evaluate(
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
    out_path = f"{args.output}/{args.eval_prefix}{EVAL_FILE}"
    write_json(results, out_path)
    return results


def asd_test(args, params=None):
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

    assert args.batch_size > 0

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.seed is None:
        args.seed = 1234

    print(f"Asd testing: Using seed [{args.seed}] for this process.")

    args.metric = "accuracy"

    # initialize the random generator
    initialize_seed(args.seed)
    
    # create environment
    env = ASDPatientInteractionSimulator(args, patient_filepath=args.data, train=False)
    print(f"Test environment size: {env.sample_size}")

    args.inp_size = env.state_size + 5
    args.symptom_prediction_size = env.symptom_size
    args.action_state_size = 1
    args.patho_size = env.diag_size if args.patho_size is None else args.patho_size

    # import pdb;pdb.set_trace()
    # instantiate the agent
    saved_model_path = f"{args.output}/{BEST_PRETRAIN_MODEL_NAME}"
    agent = create_agent(args)

    args.device = torch.device(
        f"cuda:{args.cuda_idx}"
        if torch.cuda.is_available() and args.cuda_idx is not None
        else "cpu"
    )

    agent = load_model(agent, saved_model_path)
    agent.to(args.device)
    
    # create the datasets
    ds_valid = SimPaDataset(env, args.interaction_length)
    
    # infer
    asd_classifier_test(agent, ds_valid, args)
