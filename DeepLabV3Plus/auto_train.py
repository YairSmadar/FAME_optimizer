import json
import argparse
import subprocess
import os


def update_config(config_file_path, train_config):
    """Update the configuration file with new training configuration."""
    with open(config_file_path, 'r') as file:
        config = json.load(file)

    # Update the main configuration with the training configuration details
    config.update(train_config)

    # Write the updated configuration back to the file
    with open(config_file_path, 'w') as file:
        json.dump(config, file, indent=4)


def run_trainings(configs_json_path):
    """Run training sessions based on configurations defined in a JSON file."""
    with open(configs_json_path, 'r') as file:
        main_config = json.load(file)

    gpu_number = main_config["gpu_number"]
    train_script = main_config["train_script"]
    config_file_path = main_config["config_file_path"]
    train_configs = main_config["train_configs"]

    if gpu_number != '-1':
        # Set the CUBLAS_WORKSPACE_CONFIG environment variable
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

        # Set the GPU number as an environment variable (if your training script uses this)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number

    for train_config in train_configs:
        # Update the configuration file with the current training configuration
        update_config(config_file_path, train_config)

        # Construct and run the training command
        command = f'python {train_script} --config {config_file_path}'
        subprocess.run(command, shell=True, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training sessions based on a configuration JSON file.')
    parser.add_argument('--configs_json_path', type=str, required=True,
                        help='Path to the JSON file containing the main and training configurations.')

    args = parser.parse_args()

    run_trainings(args.configs_json_path)
