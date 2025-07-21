#!/usr/bin/env python3
"""
Simple HTTP Server for Testing Enhanced Manim Agent
Uses Python's built-in http.server to avoid Flask dependencies
"""

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

from enhanced_agent import EnhancedManimAgent
from validator import ManimScriptValidator

class ManimAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Manim API endpoints"""
    
    def __init__(self, *args, **kwargs):
        # Initialize agent and validator (shared across requests)
        if not hasattr(ManimAPIHandler, '_agent'):
            print("🤖 Initializing Enhanced Manim Agent...")
            ManimAPIHandler._agent = EnhancedManimAgent(llm_provider="enhanced")
            
        if not hasattr(ManimAPIHandler, '_validator'):
            try:
                print("🔍 Initializing Script Validator...")
                ManimAPIHandler._validator = ManimScriptValidator()
            except Exception as e:
                print(f"⚠️ Validator initialization failed: {e}")
                ManimAPIHandler._validator = None
                
        super().__init__(*args, **kwargs)
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self._handle_root()
        elif parsed_path.path == '/api/health':
            self._handle_health()
        elif parsed_path.path == '/api/model-info':
            self._handle_model_info()
        elif parsed_path.path == '/api/examples':
            self._handle_examples()
        else:
            self._handle_404()
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/api/generate':
            self._handle_generate()
        elif parsed_path.path == '/api/improve':
            self._handle_improve()
        else:
            self._handle_404()
    
    def _send_cors_headers(self):
        """Send CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    
    def _send_json_response(self, data, status_code=200):
        """Send JSON response with proper headers"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self._send_cors_headers()
        self.end_headers()
        
        response = json.dumps(data, indent=2)
        self.wfile.write(response.encode('utf-8'))
    
    def _handle_root(self):
        """Handle root endpoint"""
        data = {
            'name': 'Enhanced Manim Script Generator API',
            'version': '2.0.0',
            'status': 'running',
            'model_loaded': True,
            'domains': ['Manim', 'DSA', 'System Design'],
            'endpoints': {
                '/api/generate': 'POST - Generate script/explanation from prompt',
                '/api/examples': 'GET - Get example prompts',
                '/api/health': 'GET - Health check',
                '/api/model-info': 'GET - Get model information'
            }
        }
        self._send_json_response(data)
    
    def _handle_health(self):
        """Handle health check"""
        data = {
            'status': 'healthy',
            'model_loaded': True,
            'validator_loaded': self._validator is not None,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'domains': ['Manim', 'DSA', 'System Design']
        }
        self._send_json_response(data)
    
    def _handle_model_info(self):
        """Handle model info request"""
        data = self._agent.get_model_info()
        data['initialized_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._send_json_response(data)
    
    def _handle_examples(self):
        """Handle examples request"""
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
        
        data = {
            'examples': examples,
            'total': len(examples)
        }
        self._send_json_response(data)
    
    def _handle_generate(self):
        """Handle script generation request"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            # Parse JSON
            try:
                request_data = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError:
                self._send_json_response({'error': 'Invalid JSON data'}, 400)
                return
            
            user_prompt = request_data.get('prompt', '').strip()
            if not user_prompt:
                self._send_json_response({'error': 'No prompt provided'}, 400)
                return
            
            # Optional parameters
            validate_script = request_data.get('validate', True)
            include_explanation = request_data.get('explain', False)
            
            print(f"🎯 Generating for: {user_prompt}")
            
            # Generate content
            start_time = time.time()
            script = self._agent.generate_script(user_prompt)
            generation_time = time.time() - start_time
            
            if not script or script.startswith("Error"):
                self._send_json_response({
                    'error': 'Failed to generate script',
                    'details': script
                }, 500)
                return
            
            # Prepare response
            response_data = {
                'script': script,
                'prompt': user_prompt,
                'generation_time': round(generation_time, 2),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'model_info': self._agent.get_model_info().get('provider', 'Enhanced LLM')
            }
            
            # Validate if requested
            if validate_script and self._validator:
                try:
                    is_valid, fixed_script, validation_report = self._validator.validate_and_fix(script)
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
                    explanation = self._agent.explain_script(script)
                    response_data['explanation'] = explanation
                except Exception as e:
                    print(f"⚠️ Explanation failed: {e}")
                    response_data['explanation_error'] = str(e)
            
            print(f"✅ Generated successfully in {generation_time:.2f}s")
            self._send_json_response(response_data)
            
        except Exception as e:
            print(f"❌ Error in generate: {e}")
            self._send_json_response({
                'error': 'Internal server error',
                'details': str(e)
            }, 500)
    
    def _handle_improve(self):
        """Handle script improvement request"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            request_data = json.loads(post_data.decode('utf-8'))
            
            current_script = request_data.get('script', '').strip()
            improvement_request = request_data.get('improvement', '').strip()
            
            if not current_script or not improvement_request:
                self._send_json_response({
                    'error': 'Both script and improvement request are required'
                }, 400)
                return
            
            print(f"🔧 Improving script: {improvement_request}")
            
            # Generate improved script
            start_time = time.time()
            improved_script = self._agent.improve_script(current_script, improvement_request)
            generation_time = time.time() - start_time
            
            response_data = {
                'improved_script': improved_script,
                'original_script': current_script,
                'improvement_request': improvement_request,
                'generation_time': round(generation_time, 2),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self._send_json_response(response_data)
            
        except Exception as e:
            print(f"❌ Error in improve: {e}")
            self._send_json_response({
                'error': 'Internal server error',
                'details': str(e)
            }, 500)
    
    def _handle_404(self):
        """Handle 404 errors"""
        self._send_json_response({
            'error': 'Endpoint not found',
            'message': 'Please check the API documentation'
        }, 404)
    
    def log_message(self, format, *args):
        """Override log message to reduce noise"""
        # Only log errors, not every request
        if "GET" not in format and "POST" not in format:
            super().log_message(format, *args)

def run_server(port=5001):
    """Run the HTTP server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, ManimAPIHandler)
    
    print("🚀 Starting Enhanced Manim API Server")
    print("=" * 60)
    print("✅ Model loaded with template-based generation")
    print("🌐 API server starting...")
    print(f"📍 API available at: http://localhost:{port}")
    print(f"🔗 Test endpoint: http://localhost:{port}/api/health")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        httpd.shutdown()

if __name__ == '__main__':
    run_server()