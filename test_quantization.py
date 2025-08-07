#!/usr/bin/env python3
"""
Test script for power-of-2 symmetric quantization.

This script runs quick tests to verify both PTQ and QAT work correctly.

Author: Yin Cao
"""

import subprocess
import sys
import os


def run_test(name, command, timeout=120):
    """Run a test command and return success status."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Command: {command}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            timeout=timeout,
            capture_output=True,
            text=True
        )
        print("✅ SUCCESS")
        print("Output:", result.stdout[-200:])  # Show last 200 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED with return code {e.returncode}")
        print(f"Error: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT after {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    print("Power-of-2 Symmetric Quantization Test Suite")
    print("=" * 60)

    tests = [
        ("Power-of-2 Quantizer Demo", "python utils/power_of_2_quantizer.py", 60),
        ("Pure PyTorch PTQ", "python ptq_quantize.py --data_path ./data --max_eval_batches 3", 120),
        ("Pure PyTorch QAT", "python qat_train.py --data_path ./data --epochs 1 --batch_size 64", 180),
        ("AIMET + Power-of-2 PTQ", "python aimet_power_of_2_ptq.py --data_path ./data --max_eval_batches 3", 120),
        ("AIMET + Power-of-2 QAT", "python aimet_power_of_2_qat.py --data_path ./data --epochs 1 --batch_size 64", 180),
    ]

    results = []
    for name, command, timeout in tests:
        success = run_test(name, command, timeout)
        results.append((name, success))

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:30} {status}")
        if success:
            passed += 1

    print(f"\nResults: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! Power-of-2 quantization is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
