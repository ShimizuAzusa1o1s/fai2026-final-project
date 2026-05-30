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

# Automatically add the project root to sys.path so 'src' can be resolved
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
import numpy as np

from scripts.rl_env import SixNimmtEnv

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()

def train_agent(test_mode=False, start_stage=1, start_model_path=None):
    """
    Main training loop for the RL agent using Curriculum Learning.

    Args:
        test_mode (bool): If True, runs drastically fewer steps for debugging purposes.
        start_stage (int): The curriculum stage (1, 2, or 3) to start from.
        start_model_path (str | None): Optional path to a specific model checkpoint to load.
    """
    model_dir = "models/ppo_agent"
    os.makedirs(model_dir, exist_ok=True)
    
    n_envs = 16
    model = None
    
    # ---------------------------------------------------------
    # Stage 1: Sanity Check against Minimizer
    # ---------------------------------------------------------
    if start_stage <= 1:
        env_kwargs_stage1 = {
            "opponent_type": "minimizer",
        }
        
        print("Initializing environment for Stage 1...")
        env_stage1 = make_vec_env(
            SixNimmtEnv, 
            n_envs=n_envs, 
            env_kwargs=env_kwargs_stage1,
            vec_env_cls=SubprocVecEnv,
            wrapper_class=ActionMasker,
            wrapper_kwargs={"action_mask_fn": mask_fn}
        )
        
        model = MaskablePPO(
            "MlpPolicy",
            env_stage1,
            policy_kwargs={"net_arch": [512, 256]},
            batch_size=4096,
            gamma=0.99,
            verbose=1,
        )
        
        steps_1 = 5000 if test_mode else 500_000
        print(f"Starting Stage 1: Training against Minimizer ({steps_1} steps)...")
        start_time = time.time()
        
        model.learn(total_timesteps=steps_1)
        
        print(f"Stage 1 Complete in {time.time() - start_time:.2f}s")
        model.save(f"{model_dir}/stage1_model")
        env_stage1.close()
        
        if test_mode:
            print("Test mode enabled, stopping after Stage 1.")
            return
            
    # ---------------------------------------------------------
    # Stage 2: Truncated MC against FlatMC (0.01s limit)
    # ---------------------------------------------------------
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
            vec_env_cls=SubprocVecEnv,
            wrapper_class=ActionMasker,
            wrapper_kwargs={"action_mask_fn": mask_fn}
        )
        
        if model is None:
            load_path = start_model_path if start_model_path else f"{model_dir}/stage1_model"
            print(f"Loading model from {load_path}...")
            model = MaskablePPO.load(load_path, env=env_stage2)
        else:
            model.set_env(env_stage2)
        
        steps_2 = 2_000_000
        print(f"Starting Stage 2: Training against Truncated FlatMC ({steps_2} steps)...")
        start_time = time.time()
        
        checkpoint_callback = CheckpointCallback(save_freq=100_000 // n_envs, save_path=model_dir, name_prefix="stage2")
        model.learn(total_timesteps=steps_2, callback=checkpoint_callback)
        
        print(f"Stage 2 Complete in {time.time() - start_time:.2f}s")
        model.save(f"{model_dir}/stage2_model")
        env_stage2.close()

    # ---------------------------------------------------------
    # Stage 3: Full Strength Finetuning against FlatMC (0.10s limit)
    # ---------------------------------------------------------
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
            vec_env_cls=SubprocVecEnv,
            wrapper_class=ActionMasker,
            wrapper_kwargs={"action_mask_fn": mask_fn}
        )
        
        if model is None:
            load_path = start_model_path if start_model_path else f"{model_dir}/stage2_model"
            print(f"Loading model from {load_path}...")
            model = MaskablePPO.load(load_path, env=env_stage3)
        else:
            model.set_env(env_stage3)
        
        # Drop learning rate
        model.learning_rate = 1e-5
        
        steps_3 = 1_000_000
        print(f"Starting Stage 3: Training against Full FlatMC ({steps_3} steps)...")
        start_time = time.time()
        
        checkpoint_callback_3 = CheckpointCallback(save_freq=100_000 // n_envs, save_path=model_dir, name_prefix="stage3")
        model.learn(total_timesteps=steps_3, callback=checkpoint_callback_3)
        
        print(f"Stage 3 Complete in {time.time() - start_time:.2f}s")
        model.save(f"{model_dir}/stage3_model_final")
        env_stage3.close()

    # ---------------------------------------------------------
    # Stage 4: Self-play against Stage 1 Model
    # ---------------------------------------------------------
    if start_stage <= 4:
        env_kwargs_stage4 = {
            "opponent_type": "rl_agent",
            "opponent_model_path": "src/players/b12705048/agents/stage1_model"
        }
        
        n_envs_selfplay = 4
        print(f"Initializing environment for Stage 4 (reduced to {n_envs_selfplay} envs to save memory)...")
        env_stage4 = make_vec_env(
            SixNimmtEnv, 
            n_envs=n_envs_selfplay, 
            env_kwargs=env_kwargs_stage4,
            vec_env_cls=SubprocVecEnv,
            wrapper_class=ActionMasker,
            wrapper_kwargs={"action_mask_fn": mask_fn}
        )
        
        if model is None:
            load_path = start_model_path if start_model_path else "src/players/b12705048/agents/stage3_model_final"
            print(f"Loading model from {load_path}...")
            model = MaskablePPO.load(load_path, env=env_stage4)
        else:
            model.set_env(env_stage4)
            
        steps_4 = 5000 if test_mode else 1_000_000
        print(f"Starting Stage 4: Training against Stage 1 Model ({steps_4} steps)...")
        start_time = time.time()
        
        checkpoint_callback_4 = CheckpointCallback(save_freq=100_000 // n_envs_selfplay, save_path=model_dir, name_prefix="stage4")
        model.learn(total_timesteps=steps_4, callback=checkpoint_callback_4)
        
        print(f"Stage 4 Complete in {time.time() - start_time:.2f}s")
        model.save(f"{model_dir}/stage4_model")
        env_stage4.close()

    # ---------------------------------------------------------
    # Stage 5: Self-play against Stage 2 Model
    # ---------------------------------------------------------
    if start_stage <= 5:
        env_kwargs_stage5 = {
            "opponent_type": "rl_agent",
            "opponent_model_path": "src/players/b12705048/agents/stage2_model"
        }
        
        n_envs_selfplay = 4
        print(f"Initializing environment for Stage 5 (reduced to {n_envs_selfplay} envs to save memory)...")
        env_stage5 = make_vec_env(
            SixNimmtEnv, 
            n_envs=n_envs_selfplay, 
            env_kwargs=env_kwargs_stage5,
            vec_env_cls=SubprocVecEnv,
            wrapper_class=ActionMasker,
            wrapper_kwargs={"action_mask_fn": mask_fn}
        )
        
        if model is None:
            load_path = start_model_path if start_model_path else f"{model_dir}/stage4_model"
            print(f"Loading model from {load_path}...")
            model = MaskablePPO.load(load_path, env=env_stage5)
        else:
            model.set_env(env_stage5)
            
        steps_5 = 5000 if test_mode else 1_000_000
        print(f"Starting Stage 5: Training against Stage 2 Model ({steps_5} steps)...")
        start_time = time.time()
        
        checkpoint_callback_5 = CheckpointCallback(save_freq=100_000 // n_envs_selfplay, save_path=model_dir, name_prefix="stage5")
        model.learn(total_timesteps=steps_5, callback=checkpoint_callback_5)
        
        print(f"Stage 5 Complete in {time.time() - start_time:.2f}s")
        model.save(f"{model_dir}/stage5_model_final")
        env_stage5.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode (fewer steps)")
    parser.add_argument("--start-stage", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Stage to start from")
    parser.add_argument("--start-model", type=str, default=None, help="Path to checkpoint model to load (without .zip)")
    args = parser.parse_args()
    
    train_agent(test_mode=args.test_mode, start_stage=args.start_stage, start_model_path=args.start_model)
