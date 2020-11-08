import argparse



if __name__ == "__main__":

    # Take in config file
    cli_parser = argparse.ArgumentParser(description='MAML Experiment.')
    cli_parser.add_argument('--config', '-c', type=str, help='File containing configuration for experiment.', default='config/linear.json')
    args = cli_parser.parse_args()

    if args.config:
        with open(args.config) as json_file:
            config = json.load(json_file)
    else: 
        raise ValueError('No config file provided as input. Please provide using the --config flag.')

    # Parse the arguments

    parser = Parser(config)

    config_json, experiment_args, dataset_args, model_args, loss_args, training_args = parser.parse()

    # Create a report

    # Add data descriptions to report

    # Create dataset

    # Create models and MAML objects

    # Report details of model and MAML

    # Begin training of MAML

    # Output results of MAML to report
