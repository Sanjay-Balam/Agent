#!/usr/bin/env python3
"""
Fixed API Server for Enhanced Manim Script Generator
Automatically uses simple HTTP server when Flask is not available
"""

import os
import sys
import json
import time

# Check for Flask availability first
FLASK_AVAILABLE = True
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    FLASK_AVAILABLE = False

# Always import our components
from ..models.enhanced_agent import EnhancedManimAgent
from ..utils.validator import ManimScriptValidator

def start_simple_server():
    """Start the simple HTTP server"""
    print("🚀 Starting Enhanced Manim API Server (Simple HTTP)")
    print("📍 API will be available at: http://localhost:5001")
    from .simple_http_server import run_server
    run_server(5001)

def start_flask_server():
    """Start the Flask server"""
    print("🚀 Starting Enhanced Manim API Server (Flask)")
    print("=" * 60)
    
    # Initialize Flask app
    app = Flask(__name__)
    CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])
    
    # Initialize global variables
    agent = None
    validator = None
    model_info = {}
    
    def require_agent(f):
        """Decorator to ensure agent is initialized before processing requests."""
        def decorated_function(*args, **kwargs):
            nonlocal agent
            if not agent:
                return jsonify({
                    'error': 'Model not initialized. Please check server logs.',
                    'status': 'model_not_loaded'
                }), 503
            return f(*args, **kwargs)
        return decorated_function
    
    def initialize_model():
        """Initialize the model and validator."""
        nonlocal agent, validator, model_info
        
        try:
            print("🤖 Initializing Enhanced Manim Agent...")
            
            # Define paths to model files
            parent_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(parent_dir, "best_model_epoch_10.pth")
            tokenizer_path = os.path.join(parent_dir, "tokenizer.pkl")
            
            # Initialize enhanced agent (handles missing files gracefully)
            agent = EnhancedManimAgent(
                llm_provider="enhanced",
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                use_enhanced_inference=True
            )
            
            # Initialize validator
            validator = ManimScriptValidator()
            
            # Store model info
            model_info = agent.get_model_info()
            model_info['initialized_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            print("✅ Enhanced Agent initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing agent: {e}")
            return False
    
    # Define routes
    @app.route('/')
    def index():
        """API info endpoint."""
        return jsonify({
            'name': 'Enhanced Manim Script Generator API',
            'version': '2.0.0',
            'status': 'running',
            'model_loaded': agent is not None,
            'domains': ['Manim', 'DSA', 'System Design'],
            'endpoints': {
                '/api/generate': 'POST - Generate script/content from prompt',
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
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'domains': ['Manim', 'DSA', 'System Design']
        })
    
    @app.route('/api/model-info')
    def get_model_info():
        """Get model information."""
        return jsonify(model_info if model_info else {'error': 'Model not initialized'})
    
    @app.route('/api/examples')
    def get_examples():
        """Get example prompts for the frontend."""
        examples = [
            {
                "id": 1,
                "title": "Rotating Circle (Manim)",
                "prompt": "Create a blue circle that rotates 360 degrees",
                "category": "manim"
            },
            {
                "id": 2,
                "title": "Binary Search (DSA)",
                "prompt": "Explain binary search algorithm with implementation",
                "category": "dsa"
            },
            {
                "id": 3,
                "title": "Singleton Pattern (System Design)",
                "prompt": "What is singleton design pattern with example?",
                "category": "system_design"
            },
            {
                "id": 4,
                "title": "Shape Transformation (Manim)",
                "prompt": "Transform a circle into a square smoothly",
                "category": "manim"
            },
            {
                "id": 5,
                "title": "Merge Sort (DSA)",
                "prompt": "Implement merge sort with complexity analysis",
                "category": "dsa"
            },
            {
                "id": 6,
                "title": "Load Balancer (System Design)",
                "prompt": "How does a load balancer work in distributed systems?",
                "category": "system_design"
            }
        ]
        
        return jsonify({
            'examples': examples,
            'total': len(examples)
        })
    
    @app.route('/api/generate', methods=['POST'])
    @require_agent
    def generate_script():
        """Generate script/content from user prompt."""
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
            
            print(f"🎯 Generating for: {user_prompt}")
            
            # Generate content
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
                'model_info': model_info.get('provider', 'Enhanced LLM')
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
            
            print(f"✅ Generated successfully in {generation_time:.2f}s")
            return jsonify(response_data)
            
        except Exception as e:
            print(f"❌ Error in generate: {e}")
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
    
    # Initialize model
    initialize_model()
    
    print("✅ Enhanced Agent loaded!")
    print("🌐 Flask API server starting...")
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

if __name__ == '__main__':
    if FLASK_AVAILABLE:
        start_flask_server()
    else:
        print("⚠️ Flask not available, using simple HTTP server...")
        start_simple_server()