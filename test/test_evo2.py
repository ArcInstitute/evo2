import argparse
import csv
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

from evo2 import Evo2

def read_prompts(input_file: Path) -> Union[List[List[str]]]:
    """Read prompts from input file."""
    promptseqs: List[str] = []
    
    with open(input_file, encoding='utf-8-sig', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            promptseqs.append(row[0])

    return promptseqs

def test_forward_pass(*, model, sequences):
    """Test model forward pass accuracy on sequences."""
    losses = []
    accuracies = []
    
    for seq in sequences:
        # Convert sequence to model input format
        input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=int).to('cuda:0')
        
        with torch.inference_mode():
            # Forward pass
            logits, _ = model.model.forward(input_ids.unsqueeze(0))
            
            # Calculate loss and accuracy
            target_ids = input_ids[1:]  # Shift right for next token prediction
            pred_logits = logits[0, :-1, :]
            
            # Cross entropy loss
            loss = F.cross_entropy(
                pred_logits, 
                target_ids.long()
            )
            
            # Get predictions
            pred_tokens = torch.argmax(pred_logits, dim=-1)
            
            # Calculate accuracy
            accuracy = (target_ids == pred_tokens).float().mean().item()
            
            losses.append(loss.item())
            accuracies.append(accuracy)
    
    # Print sequence results
    print("\nSequence Results:")
    for i, (loss, acc) in enumerate(zip(losses, accuracies)):
        print(f"Sequence {i+1}: Loss = {loss:.3f}, Accuracy = {acc:.2%}")
        if acc < 0.5:
            print("WARNING: Forward pass accuracy is below 50% on test sequence. Model may be broken, trained models should have >80% accuracy.")
    
    return accuracies, losses

def main():
    """
    Test sequence prediction accuracy using Evo2 models.
    Expected results for forward pass:
    - Evo 2 40B 1m: Loss ~0.216, Accuracy ~91.67%
    - Evo 2 7B 1m: Loss ~0.348, Accuracy ~86.35%
    - Evo 2 1B base: Loss ~0.502, Accuracy ~79.56%
    """
    parser = argparse.ArgumentParser(description="Test Evo2 Model Forward Pass")
    parser.add_argument("--model_name", choices=['evo2_7b', 'evo2_40b', 'evo2_7b_base', 'evo2_40b_base', 'evo2_1b_base'], 
                       default='evo2_7b',
                       help="Model to test")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    
    # Initialize model
    model = Evo2(args.model_name)
    
    # Read sequences
    sequences = read_prompts('vortex/test/data/prompts.csv')
    
    # Test forward pass
    accuracies, losses = test_forward_pass(
        model=model,
        sequences=sequences
    )
    
    # Calculate and validate results
    mean_loss = np.mean(losses)
    mean_accuracy = np.mean(accuracies) * 100
    print(f"\nMean Loss: {mean_loss:.3f}")
    print(f"Mean Accuracy: {mean_accuracy:.3f}%")
    
    # Validate against expected scores
    eps = 1e-3  # epsilon for float comparison
    expected_metrics = {
        'evo2_40b': {'loss': 0.2159424, 'acc': 91.673},
        'evo2_7b': {'loss': 0.3476563, 'acc': 86.346},
        'evo2_40b_base': {'loss': 0.2149658, 'acc': 91.741},
        'evo2_7b_base': {'loss': 0.3520508, 'acc': 85.921},
        'evo2_1b_base': {'loss': 0.501953125, 'acc': 79.556}
    }
    
    expected = expected_metrics[args.model_name]
    if abs(mean_loss - expected['loss']) < eps:
        print(f"\nTest Passed! Loss matches expected {expected['loss']:.3f}")
    else:
        print(f"\nTest Failed: Expected loss {expected['loss']:.3f}, got {mean_loss:.3f}")

if __name__ == "__main__":
    main()