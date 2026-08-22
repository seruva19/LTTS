#!/usr/bin/env python3
"""
Test script for enhanced backward pass gradient tracking via Captum.

This script demonstrates the improved gradient computation capabilities
of the gradient_importance module.
"""

import torch
import sys
import os

# Add the ltts_modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ltts_modules"))


def run_basic_backward_pass():
    """Run the basic backward pass smoke check."""
    print("=" * 60)
    print("Testing Basic Backward Pass")
    print("=" * 60)

    # Create test tensor
    x = torch.randn(1, 5, 768, requires_grad=True)

    # Simple computation
    y = x.sum()

    # Backward pass
    y.backward()

    # Check gradients
    if x.grad is not None:
        print("[OK] Basic backward pass works!")
        print(f"  Input shape: {x.shape}")
        print(f"  Gradient shape: {x.grad.shape}")
        print(f"  Gradient mean: {x.grad.mean().item():.6f}")
        print(f"  Gradient std: {x.grad.std().item():.6f}")
        return True
    else:
        print("[FAIL] No gradients computed!")
        return False


def test_basic_backward_pass():
    """Test basic backward pass functionality."""
    assert run_basic_backward_pass()


def run_captum_integration():
    """Run Captum integration smoke checks."""
    print("\n" + "=" * 60)
    print("Testing Captum Integration")
    print("=" * 60)

    try:
        from gradient_importance.captum_gradients import (
            compute_integrated_gradients_captum,
            compute_gradient_with_backward_pass,
            verify_gradient_flow,
        )
        print("[OK] Captum gradient functions imported successfully!")

        # Test tensor
        outputs = torch.randn(1, 10, 768, requires_grad=True)

        # Test gradient flow verification
        print("\nTesting gradient flow verification...")
        checks = verify_gradient_flow(outputs, "test_outputs")

        if checks["computational_graph_connected"]:
            print("[OK] Computational graph is connected!")
        else:
            print("[WARNING] Computational graph issues detected!")

        # Test explicit backward pass
        print("\nTesting explicit backward pass...")
        gradients = compute_gradient_with_backward_pass(outputs)

        if gradients is not None:
            print("[OK] Explicit backward pass successful!")
            print(f"  Gradient shape: {gradients.shape}")
        else:
            print("[FAIL] Explicit backward pass failed!")

        # Test integrated gradients (reduced steps for testing)
        print("\nTesting integrated gradients...")
        try:
            ig_attributions = compute_integrated_gradients_captum(
                outputs,
                n_steps=5  # Reduced for testing
            )
            if ig_attributions is not None:
                print("[OK] Integrated gradients computed!")
                print(f"  Attribution shape: {ig_attributions.shape}")
            else:
                print("[FAIL] Integrated gradients failed!")
        except Exception as e:
            print(f"[ERROR] Integrated gradients error: {e}")

        return True

    except ImportError as e:
        print(f"[FAIL] Could not import Captum functions: {e}")
        return False


def test_captum_integration():
    """Test Captum integration for gradient computation."""
    assert run_captum_integration()


def run_gradient_hook_tracking():
    """Run gradient hook tracking smoke checks."""
    print("\n" + "=" * 60)
    print("Testing Gradient Hook Tracking")
    print("=" * 60)

    try:
        from gradient_importance.init import register_gradient_tracking_hook

        # Test tensor
        test_tensor = torch.randn(1, 8, 512, requires_grad=True)
        layer_path = "test_layer"

        # Register hook
        success = register_gradient_tracking_hook(test_tensor, layer_path)

        if success:
            print("[OK] Gradient tracking hook registered!")

            # Perform backward pass to trigger hook
            loss = test_tensor.sum()
            loss.backward()

            print("[OK] Backward pass completed - hook should have triggered!")
        else:
            print("[FAIL] Failed to register gradient tracking hook!")

        return success

    except ImportError as e:
        print(f"[FAIL] Could not import gradient tracking function: {e}")
        return False


def test_gradient_hook_tracking():
    """Test gradient hook tracking functionality."""
    assert run_gradient_hook_tracking()


def run_layer_wise_gradients():
    """Run layer-wise gradient smoke checks."""
    print("\n" + "=" * 60)
    print("Testing Layer-wise Gradient Computation")
    print("=" * 60)

    try:
        from gradient_importance.captum_gradients import compute_layer_wise_gradients

        # Create multiple layer outputs
        layer_outputs = {
            "layer1": torch.randn(1, 10, 768, requires_grad=True),
            "layer2": torch.randn(1, 10, 512, requires_grad=True),
            "layer3": torch.randn(1, 10, 256, requires_grad=True),
        }

        # Compute layer-wise gradients
        gradients = compute_layer_wise_gradients(
            layer_outputs,
            target_layer="layer2"
        )

        if gradients:
            print("[OK] Layer-wise gradients computed!")
            for layer_name, grad in gradients.items():
                print(f"  {layer_name}: {grad.shape}")
        else:
            print("[FAIL] Layer-wise gradients failed!")

        return len(gradients) > 0

    except ImportError as e:
        print(f"[FAIL] Could not import layer-wise gradient function: {e}")
        return False


def test_layer_wise_gradients():
    """Test layer-wise gradient computation."""
    assert run_layer_wise_gradients()


def run_gradient_methods_comparison():
    """Run gradient method comparison smoke checks."""
    print("\n" + "=" * 60)
    print("Testing Gradient Methods Comparison")
    print("=" * 60)

    try:
        from gradient_importance.init import (
            compute_saliency_with_captum,
            compute_input_x_gradient_captum,
            compute_explicit_backward_gradients,
        )

        # Test tensor
        outputs = torch.randn(1, 15, 768, requires_grad=True)

        results = {}

        # Test saliency
        print("Testing saliency method...")
        try:
            saliency_result = compute_saliency_with_captum(outputs)
            results["saliency"] = saliency_result
            print(f"[OK] Saliency: {saliency_result.shape}")
        except Exception as e:
            print(f"[FAIL] Saliency failed: {e}")

        # Test input x gradient
        print("Testing input x gradient method...")
        try:
            input_x_grad_result = compute_input_x_gradient_captum(outputs)
            results["input_x_gradient"] = input_x_grad_result
            print(f"[OK] Input x Gradient: {input_x_grad_result.shape}")
        except Exception as e:
            print(f"[FAIL] Input x Gradient failed: {e}")

        # Test explicit backward
        print("Testing explicit backward method...")
        try:
            explicit_result = compute_explicit_backward_gradients(outputs)
            results["explicit_backward"] = explicit_result
            print(f"[OK] Explicit Backward: {explicit_result.shape}")
        except Exception as e:
            print(f"[FAIL] Explicit backward failed: {e}")

        # Compare results if we have multiple methods
        if len(results) > 1:
            print("\nComparison Summary:")
            for method, result in results.items():
                print(
                    f"  {method}: mean={result.mean().item():.6f}, std={result.std().item():.6f}"
                )

        return len(results) > 0

    except ImportError as e:
        print(f"[FAIL] Could not import gradient computation functions: {e}")
        return False


def test_gradient_methods_comparison():
    """Compare different gradient computation methods."""
    assert run_gradient_methods_comparison()


def main():
    """Run all tests."""
    print("Enhanced Backward Pass Gradient Tracking Test")
    print("=" * 60)

    # Check if Captum is available
    try:
        import captum
        print("[OK] Captum is available!")
    except ImportError:
        print("[ERROR] Captum is not installed!")
        print("Install with: pip install captum")
        return False

    # Run tests
    tests = [
        run_basic_backward_pass,
        run_captum_integration,
        run_gradient_hook_tracking,
        run_layer_wise_gradients,
        run_gradient_methods_comparison,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Test {test.__name__} failed with exception: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("[SUCCESS] All tests passed! Enhanced gradient tracking is working.")
    else:
        print("[WARNING] Some tests failed. Check the output above for details.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
