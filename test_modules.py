#!/usr/bin/env python3
"""
Smoke tests for project modules.

Confirms imports succeed and minimal functions run with toy data. These tests
are intentionally light-weight to be runnable on dev machines without GPUs.

Author: Ahmed Alghaith
Date: August 2025
"""
from utils import *
def test_imports():
    """Assert core modules import successfully (smoke test)."""
    print("🧪 Testing module imports...")

    try:
        
        print("✅ utils.py imported successfully")

        # Test dependency checking
        deps = check_dependencies()

    except Exception as e:
        print(f"❌ utils.py import failed: {e}")
        return False

    try:
        from MusicStreamingEventProcessor import MusicStreamingEventProcessor
        print("✅ MusicStreamingEventProcessor imported successfully")
    except Exception as e:
        print(f"❌ MusicStreamingEventProcessor import failed: {e}")
        return False

    print("\n🎉 All core modules imported successfully!")
    return True


def test_basic_functionality():
    """Run a mini pipeline with synthetic data to catch regressions early."""
    print("\n🔧 Testing basic functionality...")

    try:
        from utils import temporal_split, clean_split, setup_plotting_style
        from MusicStreamingEventProcessor import MusicStreamingEventProcessor

        # Test setup function
        setup_plotting_style()
        print("✅ Plotting style setup works")

        # Create minimal test data
        test_data = pd.DataFrame({
            'userId': ['1', '2', '3'],
            'churn': [0, 1, 0]
        })

        # Test temporal split
        train_df, val_df, test_df = temporal_split(test_data)
        print("✅ Temporal split works")

        # Test clean split  
        X_clean, y_clean = clean_split(test_data[['userId']], test_data['churn'])
        print("✅ Clean split works")

        print("\n🎉 Basic functionality tests passed!")
        return True

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting module tests...")
    print("=" * 50)

    imports_ok = test_imports()
    if imports_ok:
        functionality_ok = test_basic_functionality()

        if functionality_ok:
            print("\n🏆 ALL TESTS PASSED!")
            print("📋 Modules are ready for use")
        else:
            print("\n⚠️ Some functionality tests failed")
    else:
        print("\n❌ Import tests failed")

    print("\n💡 If tests fail, check:")
    print("   1. All required dependencies are installed")
    print("   2. File paths are correct") 
    print("   3. No syntax errors in modules")
