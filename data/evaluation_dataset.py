import os
import yaml
import torch
import numpy as np
from dataset import CustomDataset
import pickle
from typing import Dict, List


class EvaluationDataset:
    def __init__(self, config_path: str, samples_per_factor: int = 100):
        """Initialize evaluation dataset generator.
        
        Args:
            config_path: Path to config.yaml containing controlled_factors
            samples_per_factor: Number of samples to generate per factor
        """
        self.config_path = config_path
        self.samples_per_factor = samples_per_factor
        
        # Load config and get controlled factors
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.all_factors = config['state_attributes']
    
    def generate_factor_samples(self) -> Dict[str, List[Dict]]:
        """Generate samples for each controlled factor.
        
        Returns:
            Dictionary mapping factor names to lists of samples
        """
        factor_samples = {}
        
        # For each controlled factor
        for factor in self.all_factors:
            print(f"Generating samples for factor: {factor}")
            
            # Create dataset with single factor subset
            dataset = CustomDataset(
                data_env_config=self.config_path,
                limit=self.samples_per_factor,
                mode='sample',
                factor_subset=[factor]
            )
            
            # Collect samples
            samples = []
            for i in range(self.samples_per_factor):
                sample = dataset[i]
                samples.append(sample)
            
            factor_samples[factor] = samples
            
        return factor_samples
    
    def save_samples(self, output_dir: str):
        """Generate and save factor samples to disk.
        
        Args:
            output_dir: Directory to save samples
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate samples
        factor_samples = self.generate_factor_samples()
        
        # Save samples
        output_path = os.path.join(output_dir, 'factor_samples.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(factor_samples, f)
        
        print(f"Saved factor samples to {output_path}")

def main():
    # Paths
    config_path = os.path.relpath('../configs/data_generator/config.yaml')
    output_dir = os.path.relpath('eval_data/evaluation_samples')
    
    # Generate and save samples
    # eval_dataset = EvaluationDataset(config_path)
    # eval_dataset.save_samples(output_dir)

    # Load and inspect saved samples
    output_path = os.path.join(output_dir, 'factor_samples.pkl')
    with open(output_path, 'rb') as f:
        loaded_samples = pickle.load(f)
    
    # Print summary of loaded samples
    print("\nInspecting loaded samples:")
    for factor, samples in loaded_samples.items():


        print(f"\nFactor: {factor}")
        print(f"Number of samples: {len(samples)}")
        print(f"Samples: {samples}")
        
        if len(samples) > 0:
            print("First sample keys:", samples[0].keys())
            print("First sample shapes:")
            for key, value in samples[0].items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {value.shape}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {len(value)}")
                else:
                    print(f"  {key}: {type(value)}")
        else:
            print('ERROR!!!')

if __name__ == '__main__':
    main()