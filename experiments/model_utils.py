import os
from pathlib import Path
from stable_baselines3.dqn.dqn import DQN


def save_model(model: DQN, output_dir: str, episode: int, model_type: str) -> None:
    """
    Save the model to a file with a name indicating the episode and model type.

    Args:
        model (DQN): The trained model to be saved.
        output_dir (str): Directory where the model will be saved.
        episode (int): Episode number to be appended to the output file name.
        model_type (str): Type of the model ('best' or 'final').
    """
    # Validate model_type
    if model_type not in ["best", "final"]:
        raise ValueError("Invalid model_type. Must be 'best' or 'final'.")
    import os
    from pathlib import Path
    from stable_baselines3.dqn.dqn import DQN

    def save_model(model: DQN, output_dir: str, episode: int, model_type: str) -> None:
        """
        Save the model to a file with a name indicating the episode and model type.

        Args:
            model (DQN): The trained model to be saved.
            output_dir (str): Directory where the model will be saved.
            episode (int): Episode number to be appended to the output file name.
            model_type (str): Type of the model ('best' or 'final').
        """
        # Validate model_type
        if model_type not in ["best", "final"]:
            raise ValueError("Invalid model_type. Must be 'best' or 'final'.")

        # Ensure the output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Define file name
        file_name = f"RL_Based_ep{episode}.zip"
        file_path = os.path.join(output_dir, file_name)

        # Save the model
        model.save(file_path)
        print(f"{model_type.capitalize()} model saved to {file_path}")
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Define file name
    file_name = f"RL_Based_ep{episode}.zip"
    file_path = os.path.join(output_dir, file_name)

    # Save the model
    model.save(file_path)
    print(f"{model_type.capitalize()} model saved to {file_path}")