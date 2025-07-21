#!/usr/bin/env python3
"""
Script to generate Manim scripts using the custom trained LLM.
Usage: python generate_manim.py
"""

import sys
import os

# Add the parent directory to Python path to import the agent
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from agent import ManimAgent

def generate_script(request: str, save_to_file: bool = True) -> str:
    """
    Generate a Manim script from natural language request.
    
    Args:
        request: Natural language description of the animation
        save_to_file: Whether to save the generated script to a file
        
    Returns:
        Generated Manim script as string
    """
    try:
        # Define paths to model files in parent directory
        model_path = os.path.join(parent_dir, "best_model_epoch_10.pth")
        tokenizer_path = os.path.join(parent_dir, "tokenizer.pkl")
        
        # Initialize the agent with custom LLM and explicit paths
        agent = ManimAgent(
            llm_provider="custom",
            model_path=model_path,
            tokenizer_path=tokenizer_path
        )
        
        print(f"Generating script for: {request}")
        print("-" * 50)
        
        # Generate the script
        script = agent.generate_script(request)
        
        print("Generated Script:")
        print("=" * 50)
        print(script)
        print("=" * 50)
        
        # Save to file if requested
        if save_to_file:
            # Create a safe filename from the request
            filename = "".join(c for c in request.lower() if c.isalnum() or c in " -").strip()
            filename = filename.replace(" ", "_")[:50] + ".py"
            
            with open(filename, 'w') as f:
                f.write(script)
            print(f"Script saved to: {filename}")
        
        return script
        
    except Exception as e:
        error_msg = f"Error generating script: {str(e)}"
        print(error_msg)
        return error_msg

def batch_generate(requests: list) -> list:
    """Generate scripts for multiple requests."""
    try:
        # Define paths to model files in parent directory
        model_path = os.path.join(parent_dir, "best_model_epoch_10.pth")
        tokenizer_path = os.path.join(parent_dir, "tokenizer.pkl")
        
        agent = ManimAgent(
            llm_provider="custom",
            model_path=model_path,
            tokenizer_path=tokenizer_path
        )
        results = []
        
        for i, request in enumerate(requests, 1):
            print(f"\n[{i}/{len(requests)}] Processing: {request}")
            script = agent.generate_script(request)
            results.append({
                'request': request,
                'script': script
            })
            
            # Save each script
            filename = f"script_{i:02d}.py"
            with open(filename, 'w') as f:
                f.write(script)
            print(f"Saved to: {filename}")
        
        return results
        
    except Exception as e:
        print(f"Error in batch generation: {str(e)}")
        return []

def interactive_mode():
    """Interactive script generation."""
    try:
        # Define paths to model files in parent directory
        model_path = os.path.join(parent_dir, "best_model_epoch_10.pth")
        tokenizer_path = os.path.join(parent_dir, "tokenizer.pkl")
        
        agent = ManimAgent(
            llm_provider="custom",
            model_path=model_path,
            tokenizer_path=tokenizer_path
        )
        
        print("Manim Script Generator")
        print("Enter your animation requests (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            request = input("\nEnter request: ").strip()
            
            if request.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not request:
                continue
            
            script = agent.generate_script(request)
            print("\nGenerated Script:")
            print("=" * 50)
            print(script)
            print("=" * 50)
            
            # Ask to save
            save = input("\nSave to file? (y/n): ").strip().lower()
            if save in ['y', 'yes']:
                filename = input("Enter filename (without .py): ").strip() or "generated_script"
                with open(f"{filename}.py", 'w') as f:
                    f.write(script)
                print(f"Saved to: {filename}.py")
    
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    """Main function with example usage."""
    # Example 1: Single script generation
    print("=== Single Script Generation ===")
    generate_script("Create a blue circle that moves in a figure-8 pattern")
    
    print("\n=== Batch Generation ===")
    # Example 2: Batch generation
    requests = [
        "Create a red square that rotates 360 degrees",
        "Show the equation F = ma with animation",
        "Make a triangle that transforms into a circle"
    ]
    batch_generate(requests)
    
    # Example 3: Interactive mode (uncomment to use)
    # print("\n=== Interactive Mode ===")
    # interactive_mode()

if __name__ == "__main__":
    main()