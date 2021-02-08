import argparse
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from data import TaskDataset, Parser
from reports import ReportManager

from models import LinearRegression, MLP

from maml import MAML

BASE_LEARNERS = {
    "linear": LinearRegression,
    "mlp": MLP
}

if __name__ == "__main__":

    # Take in bash command.
    cli_parser = argparse.ArgumentParser(description='MAML Experiment.')
    cli_parser.add_argument('--config', '-c', type=str, help='File containing configuration for experiment.', 
                                default='config/linear_centers.json')
    args = cli_parser.parse_args()

    # Load in the config file
    if args.config:
        with open(args.config) as json_file:
            config = json.load(json_file)
    else: 
        raise ValueError('')

    # Parse the arguments.
    parser = Parser(config)
    # Unpack the output arguments of the parser.
    exp_args, train_dataset_args, valid_dataset_args, model_args, maml_args, training_args = parser.parse()

    # Update the random seeds for reproducible results.
    if exp_args["seed"]:
        torch.random.manual_seed(exp_args["seed"])
        np.random.seed(exp_args["seed"])
        random.seed(exp_args["seed"])

    # Create a report.
    report = ReportManager(exp_args["path"])
    # Add configuration to report.
    report.add_section(section="config", config_file=config)
    report.save()

    # Create datasets.
    train_dataset = TaskDataset(train_dataset_args)
    valid_dataset = TaskDataset(valid_dataset_args)

    # Create dataloaders.
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2)

    # Create models and MAML objects.
    base_learner = BASE_LEARNERS[model_args["type"].lower()](**model_args["args"])
    # Add model section to report.
    report.add_section(section="model", model=base_learner, model_args=model_args)
    report.save()

    # Initialise the MAML.
    maml = MAML(base_learner, metatrain_dataloader=train_dataloader, metatest_dataloader=valid_dataloader, **maml_args)

    # Begin training of MAML
    maml.train(**training_args)
    # Output results of MAML to report
    report.add_section(section="training", model=maml)
    report.save()
