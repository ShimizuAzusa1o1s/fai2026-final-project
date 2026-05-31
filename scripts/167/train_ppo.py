"""
RL Agent Training Script

Algorithm:
    - Trains a MaskablePPO agent via Curriculum Learning against progressively
      harder opponents (Minimizer -> Fast FlatMC -> Full FlatMC).

Characteristics:
    - **Optimization**: Uses SubprocVecEnv for vectorization and OMP limits for CPU scaling.
    - **Checkpoints**: Automatically saves progress after each phase and fixed intervals.

See Also:
    ``scripts/rl_env.py`` — The Gymnasium environment definition.
"""
import os
import sys
import time

# Set thread count to 1 to prevent massive CPU thread contention when running 16 envs
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

class SaveLatestCallback(BaseCallback):
    """
    A custom callback to periodically save the model as the latest version.
    
    This enables symmetric self-play by allowing the environment to load
    the most up-to-date policy during League Training.
    
    Attributes:
        save_freq (int): Number of steps between saves.
        save_path (str): Directory where the model will be saved.
        name_prefix (str): Filename prefix for the saved model.
    """
    def __init__(self, save_freq, save_path, name_prefix="rl_model_167_latest", verbose=1):
        """
        Initialize the SaveLatestCallback.
        
        Args:
            save_freq (int): The frequency (in steps) to save the model.
            save_path (str): The directory to save the model.
            name_prefix (str): The filename for the model.
            verbose (int): Verbosity level (0: no output, 1: info messages).
        """
        super(SaveLatestCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        """
        Action to take at each step.
        
        Returns:
            bool: True if the training should continue, False otherwise.
        """
        if self.n_calls % self.save_freq == 0:
            self.model.save(os.path.join(self.save_path, self.name_prefix))
        return True
import gymnasium as gym
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from rl_env import SixNimmtEnv

def train_agent(test_mode=False, start_stage=1, start_model_path=None):
    """
    Main training loop for the RL agent using Curriculum Learning.

    Args:
        test_mode (bool): If True, runs drastically fewer steps for debugging purposes.
        start_stage (int): The curriculum stage (1 to 5) to start from.
        start_model_path (str | None): Optional path to a specific model checkpoint to load.
    """
    model_dir = "src/players/b12705048/agents/models"
    os.makedirs(model_dir, exist_ok=True)
    
    n_envs = 16
    model = None
    
    # ---- Phase 1A: Minimizer (Trick 8 Spawning - High Certainty) ----
    if start_stage <= 1:
        env_kwargs_stage1a = {
            "opponent_type": "minimizer",
            "spawn_trick": 7,
            "reward_shaping_weight": 1.0
        }
        
        print("Initializing environment for Stage 1A (spawn_trick=7)...")
        env_stage1a = make_vec_env(
            SixNimmtEnv, 
            n_envs=n_envs, 
            env_kwargs=env_kwargs_stage1a,
            vec_env_cls=SubprocVecEnv
        )
        
        load_path = start_model_path if start_model_path else None
        if load_path and model is None:
            print(f"Loading model from {load_path}...")
            model = RecurrentPPO.load(load_path, env=env_stage1a)
        elif model is None:
            model = RecurrentPPO(
                "MlpLstmPolicy",
                env_stage1a,
                policy_kwargs={"net_arch": [512, 256], "lstm_hidden_size": 256},
                batch_size=4096,
                gamma=0.99,
                verbose=1,
            )
        else:
            model.set_env(env_stage1a)
            
        steps_1a = 5000 if test_mode else 200_000
        print(f"Starting Stage 1A: Training against Minimizer ({steps_1a} steps)...")
        start_time = time.time()
        
        model.learn(total_timesteps=steps_1a)
        
        print(f"Phase 1A Complete in {time.time() - start_time:.2f}s")
        model.save(f"{model_dir}/rl_model_167_stage1a")
        env_stage1a.close()

    # ---- Phase 1B: Minimizer (Trick 5 Spawning - Medium Certainty) ----
    if start_stage <= 1:
        env_kwargs_stage1b = {
            "opponent_type": "minimizer",
            "spawn_trick": 4,
            "reward_shaping_weight": 0.5
        }
        
        print("Initializing environment for Stage 1B (spawn_trick=4)...")
        env_stage1b = make_vec_env(
            SixNimmtEnv, 
            n_envs=n_envs, 
            env_kwargs=env_kwargs_stage1b,
            vec_env_cls=SubprocVecEnv
        )
        
        if model is None:
            model = RecurrentPPO.load(f"{model_dir}/rl_model_167_stage1a", env=env_stage1b)
        else:
            model.set_env(env_stage1b)
            
        steps_1b = 5000 if test_mode else 300_000
        print(f"Starting Stage 1B: Training against Minimizer ({steps_1b} steps)...")
        start_time = time.time()
        
        model.learn(total_timesteps=steps_1b)
        
        print(f"Phase 1B Complete in {time.time() - start_time:.2f}s")
        model.save(f"{model_dir}/rl_model_167_stage1b")
        env_stage1b.close()

    # ---- Phase 1C: Minimizer (Trick 1 Spawning - Full Uncertainty) ----
    if start_stage <= 1:
        env_kwargs_stage1c = {
            "opponent_type": "minimizer",
            "spawn_trick": 0,
            "reward_shaping_weight": 0.1
        }
        
        print("Initializing environment for Stage 1C (spawn_trick=0)...")
        env_stage1c = make_vec_env(
            SixNimmtEnv, 
            n_envs=n_envs, 
            env_kwargs=env_kwargs_stage1c,
            vec_env_cls=SubprocVecEnv
        )
        
        if model is None:
            model = RecurrentPPO.load(f"{model_dir}/rl_model_167_stage1b", env=env_stage1c)
        else:
            model.set_env(env_stage1c)
            
        steps_1c = 5000 if test_mode else 500_000
        print(f"Starting Stage 1C: Training against Minimizer ({steps_1c} steps)...")
        start_time = time.time()
        
        model.learn(total_timesteps=steps_1c)
        
        print(f"Stage 1C Complete in {time.time() - start_time:.2f}s")
        model.save(f"{model_dir}/rl_model_167_stage1c")
        env_stage1c.close()
        
        if test_mode:
            print("Test mode enabled, stopping after Stage 1.")
            return
            
    # ---- Phase 2: Truncated MC against FlatMC (0.01s limit) ----
    if start_stage <= 2:
        env_kwargs_stage2 = {
            "opponent_type": "flatmc",
            "opponent_time_limit": 0.01
        }
        
        print("Initializing environment for Stage 2...")
        env_stage2 = make_vec_env(
            SixNimmtEnv, 
            n_envs=n_envs, 
            env_kwargs=env_kwargs_stage2,
            vec_env_cls=SubprocVecEnv
        )
        
        if model is None:
            load_path = start_model_path if start_model_path else f"{model_dir}/rl_model_167_stage1c"
            print(f"Loading model from {load_path}...")
            model = RecurrentPPO.load(load_path, env=env_stage2)
        else:
            model.set_env(env_stage2)
        
        steps_2 = 2_000_000
        print(f"Starting Stage 2: Training against Truncated FlatMC ({steps_2} steps)...")
        start_time = time.time()
        
        checkpoint_callback = CheckpointCallback(save_freq=100_000 // n_envs, save_path=model_dir, name_prefix="rl_model_167_stage2")
        model.learn(total_timesteps=steps_2, callback=checkpoint_callback)
        
        print(f"Phase 2 Complete in {time.time() - start_time:.2f}s")
        model.save(f"{model_dir}/rl_model_167_stage2")
        env_stage2.close()

    # ---- Phase 3: Full Strength Finetuning against FlatMC (0.10s limit) ----
    if start_stage <= 3:
        env_kwargs_stage3 = {
            "opponent_type": "flatmc",
            "opponent_time_limit": 0.10
        }
        
        print("Initializing environment for Stage 3...")
        env_stage3 = make_vec_env(
            SixNimmtEnv, 
            n_envs=n_envs, 
            env_kwargs=env_kwargs_stage3,
            vec_env_cls=SubprocVecEnv
        )
        
        if model is None:
            load_path = start_model_path if start_model_path else f"{model_dir}/rl_model_167_stage2"
            print(f"Loading model from {load_path}...")
            model = RecurrentPPO.load(load_path, env=env_stage3)
        else:
            model.set_env(env_stage3)
        
        # Drop learning rate
        model.learning_rate = 1e-5
        
        steps_3 = 1_000_000
        print(f"Starting Stage 3: Training against Full FlatMC ({steps_3} steps)...")
        start_time = time.time()
        
        checkpoint_callback_3 = CheckpointCallback(save_freq=100_000 // n_envs, save_path=model_dir, name_prefix="rl_model_167_stage3")
        model.learn(total_timesteps=steps_3, callback=checkpoint_callback_3)
        
        print(f"Phase 3 Complete in {time.time() - start_time:.2f}s")
        model.save(f"{model_dir}/rl_model_167_stage3")
        env_stage3.close()

    # ---- Phase 4: League Training (Mixed Opponents + Symmetric Self-play) ----
    if start_stage <= 4:
        historical_models = [
            "src/players/b12705048/agents/models/rl_model_167_stage1c",
            "src/players/b12705048/agents/models/rl_model_167_stage2",
            "src/players/b12705048/agents/models/rl_model_167_stage3",
            "src/players/b12705048/agents/models/rl_model_167_latest"
        ]
        env_kwargs_stage4 = {
            "opponent_type": "mixed",
            "opponent_model_path": historical_models,
            "opponent_time_limit": 0.01  # Use fast flatmc to speed up training
        }
        
        # Increased envs to 8 since models are now cached efficiently
        n_envs_selfplay = 8
        print(f"Initializing environment for Stage 4 (League Training with {n_envs_selfplay} envs)...")
        env_stage4 = make_vec_env(
            SixNimmtEnv, 
            n_envs=n_envs_selfplay, 
            env_kwargs=env_kwargs_stage4,
            vec_env_cls=SubprocVecEnv
        )
        
        if model is None:
            load_path = start_model_path if start_model_path else f"{model_dir}/rl_model_167_stage3"
            print(f"Loading model from {load_path}...")
            model = RecurrentPPO.load(load_path, env=env_stage4)
        else:
            model.set_env(env_stage4)
            
        # Save initially so environments have a latest model to load
        model.save(f"{model_dir}/rl_model_167_latest")
            
        steps_4 = 5000 if test_mode else 2_000_000
        print(f"Starting Stage 4: League Training ({steps_4} steps)...")
        start_time = time.time()
        
        checkpoint_callback_4 = CheckpointCallback(save_freq=100_000 // n_envs_selfplay, save_path=model_dir, name_prefix="rl_model_167_stage4")
        save_latest_callback = SaveLatestCallback(save_freq=50_000 // n_envs_selfplay, save_path=model_dir)
        
        model.learn(total_timesteps=steps_4, callback=[checkpoint_callback_4, save_latest_callback])
        
        print(f"Phase 4 Complete in {time.time() - start_time:.2f}s")
        model.save(f"{model_dir}/rl_model_167_stage4")
        env_stage4.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode (fewer steps)")
    parser.add_argument("--start-stage", type=int, default=1, choices=[1, 2, 3, 4], help="Stage to start from")
    parser.add_argument("--start-model", type=str, default=None, help="Path to checkpoint model to load (without .zip)")
    args = parser.parse_args()
    
    train_agent(test_mode=args.test_mode, start_stage=args.start_stage, start_model_path=args.start_model)
