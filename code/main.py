import argparse
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data import TaskDataset, Parser
from reports import ReportManager

from models import LinearRegression

from maml import MAML

BASE_LEARNERS = {
    "linear": LinearRegression
}


if __name__ == "__main__":

    # Take in config file
    cli_parser = argparse.ArgumentParser(description='MAML Experiment.')
    cli_parser.add_argument('--config', '-c', type=str, help='File containing configuration for experiment.', default='config/linear.json')
    args = cli_parser.parse_args()

    if args.config:
        with open(args.config) as json_file:
            config = json.load(json_file)
    else: 
        raise ValueError('')

    # Parse the arguments

    parser = Parser(config)

    exp_args, train_dataset_args, valid_dataset_args, model_args, maml_args, training_args = parser.parse()

    # Create a report
    report = ReportManager(exp_args["path"])
    report.add_section(section="config", config=config)
    report.save()

    # Add data descriptions to report

    # Create dataset
    train_dataset = TaskDataset(train_dataset_args)
    valid_dataset = TaskDataset(valid_dataset_args)

    # fig = plt.figure()

    # for data in dataset:
    #     plt.plot(data["train"][0], data["train"][1])

    # plt.show()

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)

    # Create models and MAML objects
    base_learner = BASE_LEARNERS[model_args["type"].lower()](**model_args["args"])

    maml = MAML(base_learner, metatrain_dataloader=train_dataloader, metatest_dataloader=valid_dataloader, **maml_args)

    # Report details of model and MAML

    # Begin training of MAML
    maml.train(**training_args)

    # Output results of MAML to report
