#!/usr/bin/env python3
"""
Quick Installation and Training Setup
Installs dependencies and starts training with one command
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run command with progress tracking."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ SUCCESS!")
            return True
        else:
            print(f"❌ FAILED: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("🚀 Installing Dependencies for Enhanced Multi-Domain LLM")
    print("=" * 60)
    
    # Check if pip is available
    pip_command = "pip" if os.system("pip --version > nul 2>&1") == 0 else "pip3"
    
    dependencies = [
        "torch",
        "numpy>=1.21.0", 
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0"
    ]
    
    print(f"📦 Installing packages: {', '.join(dependencies)}")
    
    for package in dependencies:
        success = run_command(
            f"{pip_command} install {package}",
            f"Installing {package}"
        )
        
        if not success:
            print(f"⚠️ Failed to install {package}")
            print("You can try installing manually:")
            print(f"   {pip_command} install {package}")
            return False
    
    return True

def verify_installation():
    """Verify all components are working."""
    print("\n🧪 Verifying Installation...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        
        import numpy
        print(f"✅ NumPy {numpy.__version__} installed")
        
        import tqdm
        print("✅ TQDM installed")
        
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__} installed")
        
        # Test enhanced components
        from enhanced_data_generator import EnhancedDataGenerator
        from enhanced_tokenizer import EnhancedManimTokenizer
        from enhanced_model import MultiDomainLLM
        
        print("✅ All enhanced components ready")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main installation and setup function."""
    
    print("🎯 Enhanced Multi-Domain LLM Setup")
    print("=" * 50)
    print("This will install dependencies and prepare for training")
    print("Estimated time: 5-10 minutes")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed!")
        print("Please install manually:")
        print("   pip install torch numpy tqdm matplotlib")
        return False
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Installation verification failed!")
        return False
    
    print("\n🎉 INSTALLATION COMPLETE!")
    print("=" * 50)
    print("✅ All dependencies installed")
    print("✅ Enhanced components verified")
    
    print("\n🚀 READY TO TRAIN!")
    print("Choose your next step:")
    print()
    print("1️⃣ Quick Demo (Templates - Instant):")
    print("   python3 fixed_api_server.py")
    print()
    print("2️⃣ Full Training (Professional Quality - 2-6 hours):")
    print("   python3 full_training_pipeline.py")
    print()
    print("3️⃣ Step-by-step Training:")
    print("   python3 start_training.py              # Generate data")
    print("   python3 enhanced_tokenizer.py          # Build tokenizer") 
    print("   python3 enhanced_trainer.py            # Train model")
    print("   python3 enhanced_evaluator.py          # Evaluate model")
    
    # Ask if user wants to start training
    response = input("\n🤔 Start full training now? (y/N): ")
    if response.lower() in ['y', 'yes']:
        print("\n🚀 Starting training pipeline...")
        # Use Windows-compatible training pipeline
        os.system("python windows_training_pipeline.py --auto")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)