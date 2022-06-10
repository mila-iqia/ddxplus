#!/usr/bin/env python

import argparse
import os
import time

import mlflow
import torch

from asd import asd_train
from inference import asd_test


def main():
    """This is an utility function for pretraining MixedDQN like agents.

    Here, the objective is to pretrain the classifier branch of such an agent
    in a supervised way with the hope to significantly reduce the training time in
    the RL settings while boosting performances.

    This utility can also be used to train the classifier in a full observability
    setting where it is possible to obtain the upper bound classification
    performance of the agent assuming it has a perfect knowledge of the
    symptoms/antecedents experienced by the patients. This is done by enabling the
    `no_data_corrupt` flag.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", help="path to the patient data file", type=str, required=True
    )
    parser.add_argument(
        "--eval_data",
        help="path to the patient data file for evaluation purposes."
        " if not set, use the data paremeter as default value.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--n_workers", help="number of workers for dataloading.", type=int, default=0,
    )
    parser.add_argument(
        "--thres", help="threshold for decision making", type=float, default=0.5,
    )
    parser.add_argument(
        "--min_rec_ratio",
        help="minimum recovery ratio for evidences",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--only_diff_at_end",
        type=int,
        default=0,
        help="if 1, update the network with only diff prediction at the end",
    )
    parser.add_argument("--num_epochs", help="number of epochs", type=int, default=10)
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("--eval_batch_size", help="batch size for evaluation", type=int, default=0)
    parser.add_argument("--patience", help="patience", type=int, default=10)
    parser.add_argument(
        "--valid_percentage",
        help="the percentage of data to be used for validaton. Must be in [0, 1)."
        " Useful only if eval_data is not provided.",
        type=float,
        default=None,
    )
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument(
        "--output", help="path to outputs - will results here", type=str, default="./",
    )
    parser.add_argument(
        "--patho_size",
        help="size of the pathology output",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--interaction_length", help="length of the interaction", type=int, default=30,
    )
    parser.add_argument(
        "--num_valid_trajs",
        help="number of traj to be used for validation",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--hidden_size",
        help="size of the hidden layers",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--num_layers",
        help="number of  hidden layers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--run_mode",
        help="whether 2 run train or evaluation",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--eval_prefix",
        help="prefix to add to the eval file",
        type=str,
        default="",
    )
    parser.add_argument("--cuda_idx", help="gpu to use", type=int, default=None)
    parser.add_argument("--seed", help="seed to be used", type=int, default=None)
    parser.add_argument('--exp_name', type=str, default='ASD', help='Experience Name')
    parser.add_argument(
        "--datetime_suffix",
        help="add the following datetime suffix to the output dir: "
        "<output_dir>/<yyyymmdd>/<hhmmss>",
        action="store_true",
    )
    parser.add_argument(
        '--evi_meta_path', type=str, required=True, help='path to the evidences (symptoms) meta data',
        default = '/network/data2/amlrt_internships/automatic-medical-evidence-collection/data/release_evidences.json'
    )
    parser.add_argument(
        '--patho_meta_path', type=str, required=True, help='path to the pathologies (diseases) meta data',
        default = '/network/data2/amlrt_internships/automatic-medical-evidence-collection/data/release_conditions.json'
    )
    parser.add_argument('--include_turns_in_state', action="store_true", help='whether to include turns on state')
    parser.add_argument('--no_differential', action="store_true", help='whether to not use differential')
    parser.add_argument('--no_initial_evidence', action="store_true", help='whether to not use the given initial evidence but randomly select one')
    parser.add_argument('--compute_metrics_flag', action="store_true", help='whether to compute custom evaluation metrics')
    parser.add_argument('--masked_inquired_actions', action="store_true", help='whether to mask inquired actions')
    
    args = parser.parse_args()
    if args.eval_batch_size == 0:
        args.eval_batch_size = args.batch_size
    
    assert args.num_layers > 0
    assert args.hidden_size > 0
    args.hidden_sizes = [args.hidden_size] * args.num_layers
    
    assert args.only_diff_at_end in [0, 1], "only_diff_at_end must be either 0 or 1"
    args.only_diff_at_end = args.only_diff_at_end != 0

    # assert either validation data (eval_data) or non-zero valid_percentage
    args.train = args.run_mode == "train"
    if args.run_mode == "train":
        assert (args.eval_data is not None) or (
            args.valid_percentage is not None
            and (args.valid_percentage > 0 and args.valid_percentage < 1)
        )

    # if eval data is none, set it to data
    if args.eval_data is None:
        args.eval_data = args.data

    if not (args.cuda_idx is None):
        if not torch.cuda.is_available():
            print(
                f"No cuda found. Defaulting the cuda_idx param"
                f' from From "{args.cuda_idx}" to "None".'
            )
            args.cuda_idx = None

    # add datetime suffix if required
    if args.datetime_suffix:
        time_stamp = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        args.output = os.path.join(args.output, time_stamp)

    # to be done as soon as possible otherwise mlflow
    # will not log with the proper exp. name
    mlflow.set_experiment(experiment_name=args.exp_name)

    mlflow.start_run()
    run(args)
    mlflow.end_run()


def run(args):
    """Defines the setup needed for running the pretraining of MixedDQN like agents.

    This is an utility function that properly defines the setup
    needed for running the pre-training process as well as launching the process.

    Parameters
    ----------
    args : dict
        The arguments as provided in the command line.

    Returns
    -------
    None

    """

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.run_mode == "train":
        args.log_params = True
        tmp_args = vars(args) if hasattr(args, "__dict__") else args
        for k in tmp_args:
            mlflow.log_param(k, tmp_args.get(k))
        asd_train(args=args)
        # ll
    else:
        args.log_params = False
        asd_test(args=args)


if __name__ == "__main__":
    main()
