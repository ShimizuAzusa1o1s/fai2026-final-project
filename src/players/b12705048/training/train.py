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
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
import numpy as np

from src.players.b12705048.training.rl_env import SixNimmtEnv

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()

def train_agent(test_mode=True):
    model_dir = "models/ppo_agent"
    os.makedirs(model_dir, exist_ok=True)
    
    n_envs = 16
    
    # ---------------------------------------------------------
    # Stage 1: Sanity Check against Minimizer
    # ---------------------------------------------------------
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

if __name__ == "__main__":
    train_agent(test_mode=False)
