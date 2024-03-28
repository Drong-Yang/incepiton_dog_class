"""Quick test script for model inference."""
import torch
from model import create_model

def test_inference():
    model = create_model('inception', num_classes=120)
    model.eval()
    dummy_input = torch.randn(1, 3, 299, 299)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 120), f'Unexpected shape: {output.shape}'
    print('Inference test passed!')

if __name__ == '__main__':
    test_inference()
