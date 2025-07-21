#!/usr/bin/env python3
"""
API Server for Manim Script Generator
Modern Flask API to serve the Next.js frontend
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from agent import ManimAgent
from validator import ManimScriptValidator
import os
import sys
import json
import time
from functools import wraps

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])  # Allow Next.js dev server

# Initialize global variables
agent = None
validator = None
model_info = {}

def require_agent(f):
    """Decorator to ensure agent is initialized before processing requests."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        global agent
        if not agent:
            return jsonify({
                'error': 'Model not initialized. Please check server logs.',
                'status': 'model_not_loaded'
            }), 503
        return f(*args, **kwargs)
    return decorated_function

def initialize_model():
    """Initialize the model and validator."""
    global agent, validator, model_info
    
    try:
        print("🤖 Initializing Manim Agent...")
        
        # Define paths to model files
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(parent_dir, "best_model_epoch_10.pth")
        tokenizer_path = os.path.join(parent_dir, "tokenizer.pkl")
        
        # Check if model files exist
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
            
        if not os.path.exists(tokenizer_path):
            print(f"❌ Tokenizer file not found: {tokenizer_path}")
            return False
        
        # Initialize agent with explicit paths
        agent = ManimAgent(
            llm_provider="custom",
            model_path=model_path,
            tokenizer_path=tokenizer_path
        )
        
        # Initialize validator
        validator = ManimScriptValidator()
        
        # Store model info
        model_info = agent.get_model_info()
        model_info['model_path'] = model_path
        model_info['tokenizer_path'] = tokenizer_path
        model_info['initialized_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        print("✅ Model initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return False

# Example prompts for the frontend
EXAMPLE_PROMPTS = [
    {
        "id": 1,
        "title": "Rotating Circle",
        "prompt": "Create a blue circle that rotates 360 degrees",
        "category": "basic"
    },
    {
        "id": 2,
        "title": "Bouncing Ball",
        "prompt": "Make a red ball that bounces up and down",
        "category": "physics"
    },
    {
        "id": 3,
        "title": "Mathematical Formula",
        "prompt": "Show the equation E=mc² with animation effects",
        "category": "math"
    },
    {
        "id": 4,
        "title": "Shape Transformation",
        "prompt": "Transform a circle into a square smoothly",
        "category": "transformation"
    },
    {
        "id": 5,
        "title": "Sine Wave",
        "prompt": "Create a sine wave that draws itself from left to right",
        "category": "graphs"
    },
    {
        "id": 6,
        "title": "Text Animation",
        "prompt": "Display 'Hello Manim!' with typewriter effect",
        "category": "text"
    }
]

@app.route('/')
def index():
    """API info endpoint."""
    return jsonify({
        'name': 'Manim Script Generator API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': agent is not None,
        'endpoints': {
            '/api/generate': 'POST - Generate Manim script from prompt',
            '/api/examples': 'GET - Get example prompts',
            '/api/health': 'GET - Health check',
            '/api/model-info': 'GET - Get model information'
        }
    })

@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': agent is not None,
        'validator_loaded': validator is not None,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/api/model-info')
def get_model_info():
    """Get model information."""
    global model_info
    return jsonify(model_info if model_info else {'error': 'Model not initialized'})

@app.route('/api/examples')
def get_examples():
    """Get example prompts for the frontend."""
    return jsonify({
        'examples': EXAMPLE_PROMPTS,
        'total': len(EXAMPLE_PROMPTS)
    })

@app.route('/api/generate', methods=['POST'])
@require_agent
def generate_script():
    """Generate Manim script from user prompt."""
    global agent, validator
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        user_prompt = data.get('prompt', '').strip()
        if not user_prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Optional parameters
        validate_script = data.get('validate', True)
        include_explanation = data.get('explain', False)
        
        print(f"🎯 Generating script for: {user_prompt}")
        
        # Generate script
        start_time = time.time()
        script = agent.generate_script(user_prompt)
        generation_time = time.time() - start_time
        
        if not script or script.startswith("Error"):
            return jsonify({
                'error': 'Failed to generate script',
                'details': script
            }), 500
        
        # Prepare response
        response_data = {
            'script': script,
            'prompt': user_prompt,
            'generation_time': round(generation_time, 2),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model_info': model_info.get('provider', 'Custom LLM')
        }
        
        # Validate script if requested
        if validate_script and validator:
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
        
        # Add explanation if requested
        if include_explanation:
            try:
                explanation = agent.explain_script(script)
                response_data['explanation'] = explanation
            except Exception as e:
                print(f"⚠️ Explanation failed: {e}")
                response_data['explanation_error'] = str(e)
        
        print(f"✅ Script generated successfully in {generation_time:.2f}s")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in generate_script: {e}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/api/improve', methods=['POST'])
@require_agent
def improve_script():
    """Improve an existing script based on feedback."""
    global agent
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        current_script = data.get('script', '').strip()
        improvement_request = data.get('improvement', '').strip()
        
        if not current_script or not improvement_request:
            return jsonify({'error': 'Both script and improvement request are required'}), 400
        
        print(f"🔧 Improving script: {improvement_request}")
        
        # Generate improved script
        start_time = time.time()
        improved_script = agent.improve_script(current_script, improvement_request)
        generation_time = time.time() - start_time
        
        return jsonify({
            'improved_script': improved_script,
            'original_script': current_script,
            'improvement_request': improvement_request,
            'generation_time': round(generation_time, 2),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        print(f"❌ Error in improve_script: {e}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Please check the API documentation'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong on the server'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Manim Script Generator API Server")
    print("=" * 60)
    
    # Check if model files exist
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(parent_dir, "best_model_epoch_10.pth")
    tokenizer_path = os.path.join(parent_dir, "tokenizer.pkl")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the model is trained and available.")
        sys.exit(1)
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer file not found: {tokenizer_path}")
        print("Please ensure the tokenizer is available.")
        sys.exit(1)
    
    # Initialize model
    if initialize_model():
        print("✅ Model loaded successfully!")
        print("🌐 API server starting...")
        print("📍 API available at: http://localhost:5001")
        print("🔗 Frontend should connect to: http://localhost:5001/api/")
        print("📋 Health check: http://localhost:5001/api/health")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 60)
        
        # Run the server
        app.run(
            debug=True,
            port=5001,
            host='0.0.0.0',
            threaded=True
        )
    else:
        print("❌ Failed to initialize model")
        print("Please make sure all model files are available and trained")
        sys.exit(1)