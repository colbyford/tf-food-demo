import click
from azureml.core import Experiment, Run
from azureml.core.authentication import AzureCliAuthentication
from azureml.train.dnn import TensorFlow
from azureml.train.hyperdrive import (BanditPolicy, HyperDriveConfig,
                                      PrimaryMetricGoal,
                                      RandomParameterSampling, choice,
                                      loguniform)


@click.command()
@click.option("--model-name", type=str, default='model')
@click.option('--run-id', type=str, required=True)
def register_model(model_name, run_id):
    """Register the model to the AML Workspace"""
    cli_auth = AzureCliAuthentication()

    experiment = Experiment.from_directory(".", auth=cli_auth)
    run = Run(experiment, run_id)

    run.register_model(model_name, model_path='outputs/final_model.hdf5',
                       model_framework='TfKeras', model_framework_version='1.13')


if __name__ == "__main__":
    register_model()
