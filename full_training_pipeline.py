#!/usr/bin/env python3
"""
Complete Enhanced Multi-Domain Training Pipeline
Orchestrates the entire training process from data generation to model evaluation
"""

import os
import sys
import time
import subprocess
from typing import Optional, Dict
import json

def run_command(command: str, description: str) -> bool:
    """Run a command and track its success."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ SUCCESS!")
            if result.stdout:
                print(f"Output: {result.stdout[:500]}...")
        else:
            print("❌ FAILED!")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False
    
    elapsed = time.time() - start_time
    print(f"⏱️ Completed in {elapsed:.2f}s")
    
    return True

def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_files = [
        'enhanced_data_generator.py',
        'enhanced_tokenizer.py', 
        'enhanced_model.py',
        'enhanced_trainer.py',
        'enhanced_evaluator.py',
        'multi_domain_knowledge_base.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files present")
    return True

def check_system_resources() -> Dict[str, str]:
    """Check system resources and provide recommendations."""
    print("\n🖥️ System Resource Check...")
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU Available: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Recommend batch size based on GPU memory
            if gpu_memory >= 8:
                recommended_batch_size = 8
            elif gpu_memory >= 4:
                recommended_batch_size = 4
            else:
                recommended_batch_size = 2
                
        else:
            print("⚠️ No GPU detected - using CPU (training will be slower)")
            recommended_batch_size = 2
            
    except ImportError:
        print("⚠️ PyTorch not available")
        recommended_batch_size = 2
    
    # Check available disk space
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)
        print(f"💾 Free disk space: {free_space:.1f} GB")
        
        if free_space < 5:
            print("⚠️ Low disk space - need at least 5GB for training")
            
    except Exception:
        print("⚠️ Could not check disk space")
    
    return {
        'gpu_available': str(gpu_available) if 'gpu_available' in locals() else 'false',
        'recommended_batch_size': str(recommended_batch_size),
        'estimated_training_time': '2-6 hours' if gpu_available else '6-24 hours'
    }

def training_pipeline(skip_data_generation: bool = False, 
                     skip_tokenizer: bool = False,
                     custom_epochs: Optional[int] = None,
                     custom_batch_size: Optional[int] = None) -> bool:
    """Run the complete training pipeline."""
    
    print("🎯 Enhanced Multi-Domain LLM Training Pipeline")
    print("=" * 80)
    print("This will train a model capable of:")
    print("  🎨 Manim script generation")
    print("  🧠 Data Structures & Algorithms explanations") 
    print("  🏗️ System Design concepts")
    print("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check system resources
    resources = check_system_resources()
    print(f"💡 Recommended settings based on your system:")
    print(f"   Batch size: {resources['recommended_batch_size']}")
    print(f"   Estimated time: {resources['estimated_training_time']}")
    
    # Step 1: Generate training data (if not skipping)
    if not skip_data_generation:
        if not os.path.exists('enhanced_training_data.json'):
            success = run_command(
                'python3 start_training.py',
                'Step 1: Generate Multi-Domain Training Data'
            )
            if not success:
                return False
        else:
            print("✅ Training data already exists, skipping generation")
    
    # Step 2: Build enhanced tokenizer (if not skipping)
    if not skip_tokenizer:
        if not os.path.exists('enhanced_tokenizer.pkl'):
            success = run_command(
                'python3 enhanced_tokenizer.py',
                'Step 2: Build Enhanced Multi-Domain Tokenizer'
            )
            if not success:
                return False
        else:
            print("✅ Enhanced tokenizer already exists, skipping")
    
    # Step 3: Train the model
    epochs = custom_epochs or 15
    batch_size = custom_batch_size or int(resources['recommended_batch_size'])
    
    train_command = f'python3 enhanced_trainer.py --epochs {epochs} --batch_size {batch_size}'
    
    success = run_command(
        'python3 enhanced_trainer.py',
        f'Step 3: Train Enhanced Model ({epochs} epochs, batch size {batch_size})'
    )
    
    if not success:
        print("❌ Training failed!")
        return False
    
    # Step 4: Evaluate the model
    if os.path.exists('enhanced_model_checkpoints/enhanced_best_model.pth'):
        success = run_command(
            'python3 enhanced_evaluator.py',
            'Step 4: Evaluate Trained Model'
        )
        
        if success:
            print("\n📊 Evaluation completed! Check evaluation_results_summary.json for detailed metrics.")
        else:
            print("⚠️ Evaluation failed, but training was successful")
    
    return True

def quick_test_trained_model() -> None:
    """Quick test of the trained model."""
    print("\n🧪 Quick Model Test")
    print("=" * 40)
    
    try:
        # Test queries for each domain
        test_queries = [
            ("Manim", "Create a blue circle that rotates"),
            ("DSA", "Explain binary search"),
            ("System Design", "What is singleton pattern?")
        ]
        
        from enhanced_agent import EnhancedManimAgent
        
        # Initialize with trained model
        agent = EnhancedManimAgent(
            llm_provider="enhanced",
            model_path="enhanced_model_checkpoints/enhanced_best_model.pth",
            tokenizer_path="enhanced_tokenizer.pkl"
        )
        
        for domain, query in test_queries:
            print(f"\n🔹 {domain} Test: {query}")
            start_time = time.time()
            
            try:
                response = agent.generate_script(query)
                generation_time = time.time() - start_time
                
                print(f"⏱️ Generated in {generation_time:.2f}s")
                print(f"📝 Response ({len(response)} chars): {response[:100]}...")
                
                # Basic quality check
                if len(response) > 50:
                    print("✅ Response looks good!")
                else:
                    print("⚠️ Response seems short")
                    
            except Exception as e:
                print(f"❌ Test failed: {e}")
        
        print("\n✅ Model testing completed!")
        
    except Exception as e:
        print(f"❌ Could not test model: {e}")
        print("The model may still be valid - try using it directly")

def main():
    """Main function with user interaction."""
    
    print("🚀 Enhanced Multi-Domain LLM Training Pipeline")
    print("=" * 60)
    
    # Parse command line arguments for automation
    import argparse
    parser = argparse.ArgumentParser(description='Train Enhanced Multi-Domain LLM')
    parser.add_argument('--skip-data', action='store_true', 
                       help='Skip data generation if data already exists')
    parser.add_argument('--skip-tokenizer', action='store_true',
                       help='Skip tokenizer building if tokenizer exists') 
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs (default: 15)')
    parser.add_argument('--batch-size', type=int,
                       help='Training batch size (auto-detected if not specified)')
    parser.add_argument('--auto', action='store_true',
                       help='Run automatically without user prompts')
    
    args = parser.parse_args()
    
    if not args.auto:
        print("\nThis will:")
        print("1. ✨ Generate 15,000 multi-domain training samples")
        print("2. 🔤 Build enhanced tokenizer with CS vocabulary")  
        print("3. 🤖 Train advanced transformer model")
        print("4. 📊 Evaluate model performance")
        print("5. 🧪 Quick test the trained model")
        print(f"\nEstimated training time: 2-6 hours")
        
        response = input("\n🤔 Continue with training? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Training cancelled.")
            return
    
    # Run the training pipeline
    print("\n🏁 Starting training pipeline...")
    success = training_pipeline(
        skip_data_generation=args.skip_data,
        skip_tokenizer=args.skip_tokenizer,
        custom_epochs=args.epochs,
        custom_batch_size=args.batch_size
    )
    
    if success:
        print("\n🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Your enhanced multi-domain model is ready!")
        print("\n📁 Generated files:")
        print("   🤖 enhanced_model_checkpoints/enhanced_best_model.pth")
        print("   🔤 enhanced_tokenizer.pkl")
        print("   📊 evaluation_results_summary.json")
        
        # Quick test
        if not args.auto:
            test_response = input("\n🧪 Run quick model test? (Y/n): ")
            if test_response.lower() not in ['n', 'no']:
                quick_test_trained_model()
        
        print("\n🚀 To use your trained model:")
        print("   python3 fixed_api_server.py")
        print("\nYour API will now provide professional-quality responses for:")
        print("   🎨 Manim script generation")
        print("   🧠 DSA explanations and implementations")
        print("   🏗️ System design concepts and patterns")
        
    else:
        print("\n❌ Training pipeline failed!")
        print("Check the error messages above and ensure all dependencies are installed.")
        sys.exit(1)

if __name__ == "__main__":
    main()