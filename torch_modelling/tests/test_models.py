# test_models_pytest.py
import pytest
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
import tempfile
import os


class BaseModelTest(ABC):
    """Base test class for PyTorch models using pytest."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures automatically for each test."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.create_model()
        self.model.to(self.device)
        self.input_shape = self.get_input_shape()
        self.output_shape = self.get_expected_output_shape()
        
        # Yield to run the test, then cleanup
        yield
        
        # Optional cleanup code here
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create and return the model to be tested.
        Why do we need to create a new model for testing a model?
        By creating a separate model for testing, we ensure that each test starts with a fresh instance???"""
        pass
    
    @abstractmethod
    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the input shape (batch_size, *dims)."""
        pass
    
    @abstractmethod
    def get_expected_output_shape(self) -> Tuple[int, ...]:
        """Return the expected output shape (batch_size, *dims)."""
        pass
    
    def create_sample_input(self, batch_size: int = 2) -> torch.Tensor:
        """Create sample input tensor for testing."""
        shape = (batch_size,) + self.input_shape[1:]
        return torch.randn(shape, device=self.device)
    
    def test_forward_pass(self):
        """Test that forward pass works and produces correct output shape."""
        x = self.create_sample_input()
        
        with torch.no_grad():
            output = self.model(x)
        
        expected_shape = (x.size(0),) + self.output_shape[1:]
        assert output.shape == expected_shape, \
            f"Expected output shape {expected_shape}, got {output.shape}"
    
    def test_backward_pass(self):
        """Test that backward pass works (gradients are computed)."""
        x = self.create_sample_input()
        x.requires_grad_(True)
        
        output = self.model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for model parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, \
                    f"Gradient is None for parameter {name}"
    
    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        assert trainable_params > 0, "Model has no trainable parameters"
    
    def test_model_mode_switching(self):
        """Test switching between train and eval modes."""
        # Test train mode
        self.model.train()
        assert self.model.training, "Model should be in training mode"
        
        # Test eval mode
        self.model.eval()
        assert not self.model.training, "Model should be in evaluation mode"
    
    def test_device_placement(self):
        """Test that model and its parameters are on the correct device."""
        for param in self.model.parameters():
            assert param.device == self.device, \
                f"Parameter not on expected device {self.device}"
    
    def test_output_dtype(self):
        """Test that output has the expected data type."""
        x = self.create_sample_input()
        output = self.model(x)
        assert output.dtype == torch.float32, \
            f"Expected output dtype float32, got {output.dtype}"
    
    def test_batch_independence(self):
        """Test that different batch sizes produce consistent per-sample outputs."""
        # Test with batch size 1
        x1 = self.create_sample_input(batch_size=1)
        output1 = self.model(x1)
        
        # Test with batch size 2 using the same input repeated
        x2 = torch.cat([x1, x1], dim=0)
        output2 = self.model(x2)
        
        # First sample should be the same
        torch.testing.assert_close(output1[0], output2[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(output1[0], output2[1], rtol=1e-5, atol=1e-5)
    
    def test_deterministic_output(self):
        """Test that model produces same output for same input (when in eval mode)."""
        self.model.eval()
        x = self.create_sample_input()
        
        with torch.no_grad():
            output1 = self.model(x)
            output2 = self.model(x)
        
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the entire model."""
        x = self.create_sample_input()
        output = self.model(x)
        loss = output.sum()
        
        # Compute gradients
        loss.backward()
        
        # Check that at least some parameters have non-zero gradients
        has_nonzero_grad = any(
            param.grad is not None and param.grad.abs().sum() > 1e-8
            for param in self.model.parameters() if param.requires_grad
        )
        assert has_nonzero_grad, "No parameters have non-zero gradients"
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_various_batch_sizes(self, batch_size):
        """Test model with various batch sizes using parametrization."""
        x = self.create_sample_input(batch_size=batch_size)
        output = self.model(x)
        expected_shape = (batch_size,) + self.output_shape[1:]
        assert output.shape == expected_shape
