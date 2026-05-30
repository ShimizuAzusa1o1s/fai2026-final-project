"""
Distills PyTorch MaskablePPO weights into a NumPy .npz file.
"""
import numpy as np
from sb3_contrib import MaskablePPO

def distill_weights(model_path, out_path):
    print(f"Loading model from {model_path}...")
    model = MaskablePPO.load(model_path)
    state_dict = model.policy.state_dict()
    
    weights = {
        'fc1_w': state_dict['mlp_extractor.policy_net.0.weight'].cpu().numpy(),
        'fc1_b': state_dict['mlp_extractor.policy_net.0.bias'].cpu().numpy(),
        'fc2_w': state_dict['mlp_extractor.policy_net.2.weight'].cpu().numpy(),
        'fc2_b': state_dict['mlp_extractor.policy_net.2.bias'].cpu().numpy(),
        'action_w': state_dict['action_net.weight'].cpu().numpy(),
        'action_b': state_dict['action_net.bias'].cpu().numpy(),
    }
    
    np.savez(out_path, **weights)
    print(f"Weights successfully saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="src/players/b12705048/agents/stage3_model_final")
    parser.add_argument("--out-path", type=str, default="src/players/b12705048/agents/numpy_ppo_weights.npz")
    args = parser.parse_args()
    distill_weights(args.model_path, args.out_path)
