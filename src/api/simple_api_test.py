#!/usr/bin/env python3
"""
Simple API Test - Test the enhanced agent functionality without Flask dependencies
"""

import json
import time
from enhanced_agent import EnhancedManimAgent
from validator import ManimScriptValidator

def test_api_functionality():
    """Test the enhanced agent functionality that powers the API"""
    
    print("🚀 Testing Enhanced Manim Agent Functionality")
    print("=" * 60)
    
    # Initialize agent
    try:
        agent = EnhancedManimAgent(llm_provider="enhanced")
        print("✅ Enhanced agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return False
    
    # Initialize validator
    try:
        validator = ManimScriptValidator()
        print("✅ Validator initialized successfully")
    except Exception as e:
        print(f"⚠️ Validator failed to initialize: {e}")
        validator = None
    
    # Test the exact query from your Postman request
    test_query = "Create a blue circle that rotates 360 degrees"
    
    print(f"\n🎯 Testing query: {test_query}")
    
    # Generate script
    start_time = time.time()
    script = agent.generate_script(test_query)
    generation_time = time.time() - start_time
    
    print(f"⏱️ Generation time: {generation_time:.2f}s")
    
    # Prepare response data (mimicking API response)
    response_data = {
        'script': script,
        'prompt': test_query,
        'generation_time': round(generation_time, 2),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'model_info': agent.get_model_info().get('provider', 'Enhanced LLM')
    }
    
    # Validate script if validator is available
    if validator:
        try:
            is_valid, fixed_script, validation_report = validator.validate_and_fix(script)
            response_data.update({
                'is_valid': is_valid,
                'fixed_script': fixed_script if fixed_script != script else None,
                'validation_report': validation_report
            })
            
            if is_valid and fixed_script:
                response_data['script'] = fixed_script
                
        except Exception as e:
            print(f"⚠️ Validation failed: {e}")
            response_data.update({
                'is_valid': None,
                'validation_error': str(e)
            })
    else:
        response_data.update({
            'is_valid': None,
            'validation_report': "Validator not available - assuming valid"
        })
    
    # Add explanation
    try:
        explanation = agent.explain_script(script)
        response_data['explanation'] = explanation
    except Exception as e:
        print(f"⚠️ Explanation failed: {e}")
        response_data['explanation_error'] = str(e)
    
    # Print results
    print("\n📋 API Response (simulated):")
    print(json.dumps(response_data, indent=2))
    
    print("\n🔍 Generated Script:")
    print("-" * 40)
    print(script)
    print("-" * 40)
    
    return response_data

def test_multiple_domains():
    """Test multiple domain queries"""
    
    print("\n\n🌐 Testing Multiple Domain Queries")
    print("=" * 60)
    
    agent = EnhancedManimAgent(llm_provider="enhanced")
    
    test_queries = [
        ("Manim", "Create a red square that transforms into a blue circle"),
        ("DSA", "Explain merge sort algorithm"),
        ("System Design", "What is singleton design pattern?"),
        ("Mixed", "Visualize binary tree traversal")
    ]
    
    for domain, query in test_queries:
        print(f"\n🔹 {domain} Query: {query}")
        start_time = time.time()
        result = agent.generate_script(query)
        generation_time = time.time() - start_time
        
        print(f"⏱️ Time: {generation_time:.2f}s")
        print(f"📝 Result (first 150 chars): {result[:150]}...")
        
        if "```python" in result or "from manim import" in result:
            print("✅ Contains code")
        if "Time Complexity" in result or "O(" in result:
            print("✅ Contains complexity analysis")
        if "Definition:" in result or "Benefits:" in result:
            print("✅ Contains structured explanation")

if __name__ == "__main__":
    # Test main functionality
    response = test_api_functionality()
    
    # Test multiple domains
    test_multiple_domains()
    
    print(f"\n🎉 Testing completed!")
    
    # Summary
    if response:
        is_valid = response.get('is_valid')
        if is_valid is True:
            print("✅ Generated script is valid!")
        elif is_valid is False:
            print("⚠️ Generated script has issues but was fixed")
        else:
            print("❓ Script validation unavailable")
        
        print(f"🤖 Model: {response.get('model_info', 'Unknown')}")
        print(f"⏱️ Generation: {response.get('generation_time', 0)}s")