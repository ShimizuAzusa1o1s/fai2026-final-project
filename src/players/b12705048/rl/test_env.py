#!/usr/bin/env python3
"""Diagnose info dict structure on termination."""
import sys, os
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, os.getcwd())

import numpy as np
import gymnasium as gym
from src.players.b12705048.rl.env import SixNimmtEnv

def make_env(seed):
    def _init():
        env = SixNimmtEnv(opponent_type='minimizer')
        env.reset(seed=seed)
        return env
    return _init

envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(2)])
obs, infos = envs.reset()

for step in range(10):
    masks = np.array(infos['action_mask'])
    actions = np.array([m.nonzero()[0][0] for m in masks])
    obs, rewards, terms, truncs, infos = envs.step(actions)
    
    if any(terms):
        print(f"Step {step}: TERMINATED")
        print(f"Info keys: {list(infos.keys())}")
        for key in infos:
            val = infos[key]
            if isinstance(val, np.ndarray):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            elif isinstance(val, list):
                print(f"  {key}: list len={len(val)}, types={[type(v).__name__ for v in val]}")
            else:
                print(f"  {key}: type={type(val).__name__}, val={val}")
        
        # Check for episode info
        if 'final_info' in infos:
            for i, fi in enumerate(infos['final_info']):
                if fi is not None:
                    print(f"  final_info[{i}]: keys={list(fi.keys()) if isinstance(fi, dict) else 'N/A'}")
                    if isinstance(fi, dict) and 'episode' in fi:
                        print(f"    episode: {fi['episode']}")

envs.close()
