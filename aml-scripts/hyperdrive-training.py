import click
from azureml.core import Experiment
from azureml.core.authentication import AzureCliAuthentication
from azureml.train.dnn import TensorFlow
from azureml.train.hyperdrive import (BanditPolicy, HyperDriveConfig,
                                      PrimaryMetricGoal,
                                      RandomParameterSampling, choice,
                                      loguniform)


@click.command()
@click.option("--epochs", type=int, default=10)
@click.option("--iterations", type=int, default=10)
@click.option("--ct", "--compute_target", type=str, default='gpu-cluster')
@click.option("--concurrent-runs", type=int, default=5)
def main(epochs, iterations, compute_target, concurrent_runs):
    cli_auth = AzureCliAuthentication()

    experiment = Experiment.from_directory(".", auth=cli_auth)
    ws = experiment.workspace

    cluster = ws.compute_targets[compute_target]
    food_data = ws.datastores['food_images']

    script_arguments = {
        "--data-dir": food_data.as_mount(),
        "--epochs": epochs
    }

    tf_est = TensorFlow(source_directory=".",
                        entry_script='code/train/train.py',
                        script_params=script_arguments,
                        compute_target=cluster,
                        conda_packages=['pillow', 'pandas'],
                        pip_packages=['click', 'seaborn'],
                        use_docker=True,
                        use_gpu=True,
                        framework_version='1.13'
                        )

    # Run on subset of food categories
    tf_est.run_config.arguments.extend(['apple_pie',
                                        'baby_back_ribs',
                                        'baklava',
                                        'beef_carpaccio'])

    param_sampler = RandomParameterSampling(
        {
            '--minibatch-size': choice(16, 32, 64),
            '--learning-rate': loguniform(-6, -4),
            '--optimizer': choice('rmsprop', 'adagrad', 'adam')
        }
    )

    # Create Early Termination Policy
    etpolicy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

    # Create HyperDrive Run Configuration
    hyper_drive_config = HyperDriveConfig(estimator=tf_est,
                                          hyperparameter_sampling=param_sampler,
                                          policy=etpolicy,
                                          primary_metric_name='acc',
                                          primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                          max_total_runs=iterations,
                                          max_concurrent_runs=concurrent_runs)

    # Submit the Hyperdrive Run
    hd_run = experiment.submit(hyper_drive_config)
    hd_run.wait_for_completion(raise_on_error=True)
    best_run = hd_run.get_best_run_by_primary_metric()
    print(f'##vso[task.setvariable variable=run_id]{best_run.id}')

if __name__ == "__main__":
    main()
