#!/usr/bin/env python3
"""
Batch generate multiple Manim scripts from a list of requests.
"""

from agent import ManimAgent
from validator import ManimScriptValidator
import json
import os
from datetime import datetime

def batch_generate():
    """Generate scripts for multiple requests."""
    
    print("🚀 Batch Manim Script Generator")
    print("=" * 40)
    
    # Check if model exists
    model_files = ['best_model_epoch_1.pth', 'best_model_epoch_5.pth', 'best_model_epoch_10.pth', 'final_manim_model.pth']
    model_file = None
    for file in model_files:
        if os.path.exists(file):
            model_file = file
            break
    
    if not model_file:
        print("❌ No model file found. Please train the model first:")
        print("   python train_model.py")
        return
    
    # Initialize agent
    try:
        agent = ManimAgent(llm_provider="custom")
        validator = ManimScriptValidator()
        print(f"✅ Model loaded: {model_file}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Animation requests
    requests = [
        # Basic shapes
        "Create a red circle",
        "Make a blue square", 
        "Draw a green triangle",
        "Show a yellow rectangle",
        
        # Animations
        "Create a circle that appears with animation",
        "Make a square that fades in",
        "Show a line that draws itself",
        "Create a dot that moves in a circle",
        
        # Text and math
        "Show the text 'Welcome to Manim'",
        "Display the equation x² + y² = r²",
        "Create the formula E = mc²",
        "Show the quadratic formula",
        
        # Transformations
        "Create a circle that transforms into a square",
        "Make a small dot grow into a large circle",
        "Show a line that becomes an arrow",
        "Create a square that rotates 90 degrees",
        
        # Complex animations
        "Create a bouncing ball",
        "Make a pendulum that swings",
        "Show a sine wave animation",
        "Create a coordinate system with axes",
        
        # Advanced
        "Show matrix multiplication",
        "Create a 3D cube rotation",
        "Make a fractal pattern",
        "Show the Pythagorean theorem visually"
    ]
    
    print(f"📝 Generating {len(requests)} scripts...")
    
    results = []
    successful = 0
    failed = 0
    
    for i, request in enumerate(requests, 1):
        print(f"\n[{i}/{len(requests)}] Processing: {request}")
        
        try:
            # Generate script
            script = agent.generate_script(request)
            
            # Validate script
            is_valid, fixed_script, report = validator.validate_and_fix(script)
            
            final_script = fixed_script if is_valid else script
            
            # Save individual script
            filename = f"script_{i:02d}_{request.replace(' ', '_').replace(',', '').replace('²', '2').lower()}.py"
            with open(filename, 'w') as f:
                f.write(final_script)
            
            # Store result
            results.append({
                'id': i,
                'request': request,
                'script': final_script,
                'is_valid': is_valid,
                'filename': filename,
                'length': len(final_script),
                'timestamp': datetime.now().isoformat()
            })
            
            status = "✅ Valid" if is_valid else "⚠️  Issues"
            print(f"   {status} - Saved to {filename}")
            successful += 1
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed += 1
    
    # Save batch results
    batch_results = {
        'total_requests': len(requests),
        'successful': successful,
        'failed': failed,
        'model_file': model_file,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    with open('batch_results.json', 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    print(f"\n🎉 Batch generation complete!")
    print(f"📊 Results: {successful} successful, {failed} failed")
    print(f"📁 Individual scripts saved as script_XX_*.py")
    print(f"📄 Batch summary saved to batch_results.json")
    
    # Show sample results
    if results:
        print(f"\n📋 Sample Results:")
        for result in results[:3]:  # Show first 3
            print(f"   • {result['request']}")
            print(f"     Status: {'✅ Valid' if result['is_valid'] else '⚠️  Issues'}")
            print(f"     Length: {result['length']} characters")
            print(f"     File: {result['filename']}")

if __name__ == "__main__":
    batch_generate()