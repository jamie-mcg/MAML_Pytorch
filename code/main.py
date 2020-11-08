import argparse
import json
import matplotlib.pyplot as plt

from data import TaskDataset, Parser

from models import LinearRegression

from maml import MAML

BASE_LEARNERS = {
    "linearregression": LinearRegression
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

    dataset_args, model_args, maml_args = parser.parse()

    # Create a report
    # report = ReportManager()

    # Add data descriptions to report

    # Create dataset
    dataset = TaskDataset(dataset_args)

    fig = plt.figure()

    for data in dataset:
        plt.plot(data["train"][0], data["train"][1])

    plt.show()

    # Create models and MAML objects
    # base_learner = BASE_LEARNERS[model_args["model_type"]]()

    # maml = MAML(base_learner, **maml_args)

    # Report details of model and MAML

    # Begin training of MAML
    # maml.train(**training_args)

    # Output results of MAML to report
