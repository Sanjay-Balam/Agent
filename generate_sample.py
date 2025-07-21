#!/usr/bin/env python3
"""
Simple script to generate sample Manim scripts using your trained model.
"""

from agent import ManimAgent
from validator import ManimScriptValidator
import os

def check_model_files():
    """Check if required model files exist."""
    required_files = [
        'best_model_epoch_1.pth',
        'best_model_epoch_5.pth', 
        'best_model_epoch_10.pth',
        'best_model_epoch_15.pth',
        'best_model_epoch_20.pth',
        'final_manim_model.pth'
    ]
    
    model_file = None
    for file in required_files:
        if os.path.exists(file):
            model_file = file
            break
    
    tokenizer_file = 'tokenizer.pkl'
    
    if not model_file:
        print("❌ No model file found. Please train the model first:")
        print("   python train_model.py")
        return None, None
    
    if not os.path.exists(tokenizer_file):
        print("❌ No tokenizer file found. Please train the model first:")
        print("   python train_model.py")
        return None, None
    
    print(f"✅ Found model: {model_file}")
    print(f"✅ Found tokenizer: {tokenizer_file}")
    return model_file, tokenizer_file

def generate_samples():
    """Generate sample Manim scripts."""
    
    # Check if model files exist
    model_file, tokenizer_file = check_model_files()
    if not model_file or not tokenizer_file:
        return
    
    print("\n🤖 Initializing your custom Manim LLM...")
    
    try:
        # Initialize the agent with your custom model
        agent = ManimAgent(
            llm_provider="custom",
            model_path=model_file,
            tokenizer_path=tokenizer_file
        )
        
        # Initialize validator
        validator = ManimScriptValidator()
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model info: {agent.get_model_info()}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Try training the model first: python train_model.py")
        return
    
    # Sample requests to test
    sample_requests = [
        "Create a blue circle",
        "Make a red square that moves to the right",
        "Show the text 'Hello Manim'",
        "Display the mathematical formula E=mc²",
        "Create a circle that transforms into a square",
        "Make a bouncing ball animation",
        "Show a sine wave that draws itself",
        "Create a coordinate system with axes"
    ]
    
    print("\n🎬 Generating sample Manim scripts...")
    print("=" * 60)
    
    for i, request in enumerate(sample_requests, 1):
        print(f"\n📝 Request {i}: {request}")
        print("-" * 40)
        
        try:
            # Generate script
            script = agent.generate_script(request)
            
            # Validate script
            is_valid, fixed_script, report = validator.validate_and_fix(script)
            
            # Show result
            final_script = fixed_script if is_valid else script
            
            print("🎯 Generated Script:")
            print(final_script)
            
            # Show validation status
            if is_valid:
                print("✅ Script is valid!")
            else:
                print("⚠️  Script has issues (but still usable)")
            
            # Save to file
            filename = f"sample_{i}_{request.replace(' ', '_').replace(',', '').lower()}.py"
            with open(filename, 'w') as f:
                f.write(final_script)
            print(f"💾 Saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Error generating script: {e}")
        
        print("=" * 60)
    
    print("\n🎉 Sample generation complete!")
    print("📁 Check the generated .py files in your current directory")

def interactive_mode():
    """Interactive script generation."""
    
    # Check if model files exist
    model_file, tokenizer_file = check_model_files()
    if not model_file or not tokenizer_file:
        return
    
    print("\n🤖 Initializing your custom Manim LLM...")
    
    try:
        # Initialize the agent
        agent = ManimAgent(
            llm_provider="custom",
            model_path=model_file,
            tokenizer_path=tokenizer_file
        )
        
        validator = ManimScriptValidator()
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("\n🎬 Interactive Manim Script Generator")
    print("Enter your animation requests (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        try:
            request = input("\n📝 Your request: ").strip()
            
            if request.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not request:
                continue
            
            print("🤔 Generating script...")
            script = agent.generate_script(request)
            
            print("🔍 Validating script...")
            is_valid, fixed_script, report = validator.validate_and_fix(script)
            
            final_script = fixed_script if is_valid else script
            
            print("\n" + "="*50)
            print("🎯 GENERATED SCRIPT:")
            print("="*50)
            print(final_script)
            
            print("\n" + "="*50)
            print("📊 VALIDATION:")
            print("="*50)
            if is_valid:
                print("✅ Script is valid and ready to use!")
            else:
                print("⚠️  Script has minor issues:")
                print(report)
            
            # Ask to save
            save = input("\n💾 Save this script? (y/n): ").strip().lower()
            if save in ['y', 'yes']:
                filename = input("📁 Enter filename (without .py): ").strip()
                if filename:
                    with open(f"{filename}.py", 'w') as f:
                        f.write(final_script)
                    print(f"✅ Saved to {filename}.py")
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function with menu."""
    
    print("🎬 MANIM SCRIPT GENERATOR")
    print("Using your custom trained model")
    print("=" * 40)
    
    while True:
        print("\nChoose an option:")
        print("1. Generate sample scripts")
        print("2. Interactive mode")
        print("3. Quick test")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            generate_samples()
        elif choice == '2':
            interactive_mode()
        elif choice == '3':
            quick_test()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

def quick_test():
    """Quick test to verify the model works."""
    
    model_file, tokenizer_file = check_model_files()
    if not model_file or not tokenizer_file:
        return
    
    print("\n🧪 Quick Test - Verifying your model works...")
    
    try:
        agent = ManimAgent(
            llm_provider="custom",
            model_path=model_file,
            tokenizer_path=tokenizer_file
        )
        
        test_request = "Create a blue circle"
        print(f"📝 Test request: {test_request}")
        
        script = agent.generate_script(test_request)
        
        print("\n✅ SUCCESS! Your model is working!")
        print("🎯 Generated script:")
        print("-" * 30)
        print(script)
        print("-" * 30)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Try retraining the model: python train_model.py")

if __name__ == "__main__":
    main()