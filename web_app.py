#!/usr/bin/env python3
"""
Simple web interface for generating Manim scripts.
"""

from flask import Flask, request, jsonify, render_template_string
from agent import ManimAgent
from validator import ManimScriptValidator
import os

app = Flask(__name__)

# Initialize global variables
agent = None
validator = None

def initialize_model():
    """Initialize the model and validator."""
    global agent, validator
    
    try:
        agent = ManimAgent(llm_provider="custom")
        validator = ManimScriptValidator()
        return True
    except Exception as e:
        print(f"Error initializing model: {e}")
        return False

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🎬 Manim Script Generator</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .description {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .input-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #333;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            resize: vertical;
            box-sizing: border-box;
            font-family: inherit;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .button-group {
            text-align: center;
            margin: 20px 0;
        }
        button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            margin: 0 10px;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .script-output {
            background: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            white-space: pre-wrap;
            overflow-x: auto;
            max-height: 500px;
            overflow-y: auto;
        }
        .status {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-weight: bold;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.warning {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .loading {
            text-align: center;
            padding: 20px;
            font-size: 18px;
            color: #667eea;
        }
        .examples {
            margin-top: 20px;
            padding: 15px;
            background: #e9ecef;
            border-radius: 8px;
        }
        .examples h3 {
            margin-top: 0;
            color: #333;
        }
        .example-item {
            background: white;
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 5px;
            cursor: pointer;
            border: 1px solid #ddd;
            transition: background 0.2s;
        }
        .example-item:hover {
            background: #f0f0f0;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Manim Script Generator</h1>
        <div class="description">
            Generate Python Manim scripts from natural language using your custom trained model
        </div>
        
        <div class="input-group">
            <label for="request">✨ Describe your animation:</label>
            <textarea id="request" placeholder="e.g., Create a blue circle that rotates and transforms into a square"></textarea>
        </div>
        
        <div class="button-group">
            <button onclick="generateScript()">🎯 Generate Script</button>
            <button onclick="clearAll()">🗑️ Clear</button>
        </div>
        
        <div class="examples">
            <h3>💡 Example Requests:</h3>
            <div class="example-item" onclick="setRequest('Create a blue circle that rotates')">
                Create a blue circle that rotates
            </div>
            <div class="example-item" onclick="setRequest('Make a bouncing ball animation')">
                Make a bouncing ball animation
            </div>
            <div class="example-item" onclick="setRequest('Show the mathematical formula E=mc²')">
                Show the mathematical formula E=mc²
            </div>
            <div class="example-item" onclick="setRequest('Create a sine wave that draws itself')">
                Create a sine wave that draws itself
            </div>
            <div class="example-item" onclick="setRequest('Transform a circle into a square')">
                Transform a circle into a square
            </div>
        </div>
        
        <div id="result" style="display: none;"></div>
        
        <div class="footer">
            🤖 Powered by your custom trained Manim LLM | 🏠 Running locally
        </div>
    </div>

    <script>
        function setRequest(text) {
            document.getElementById('request').value = text;
        }
        
        function clearAll() {
            document.getElementById('request').value = '';
            document.getElementById('result').style.display = 'none';
        }
        
        async function generateScript() {
            const request = document.getElementById('request').value.trim();
            const resultDiv = document.getElementById('result');
            
            if (!request) {
                alert('Please enter an animation request');
                return;
            }
            
            // Show loading
            resultDiv.innerHTML = '<div class="loading">🤔 Generating your Manim script...</div>';
            resultDiv.style.display = 'block';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ request: request })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    let statusClass = 'success';
                    let statusText = '✅ Script generated successfully!';
                    
                    if (!data.is_valid) {
                        statusClass = 'warning';
                        statusText = '⚠️ Script generated with minor issues';
                    }
                    
                    resultDiv.innerHTML = `
                        <h3>🎯 Generated Script:</h3>
                        <div class="status ${statusClass}">${statusText}</div>
                        <div class="script-output">${data.script}</div>
                        <div style="margin-top: 15px;">
                            <button onclick="downloadScript('${data.script.replace(/'/g, "\\'")}', '${request.replace(/'/g, "\\'")}')">
                                📥 Download Script
                            </button>
                            <button onclick="copyScript('${data.script.replace(/'/g, "\\'")}')">
                                📋 Copy to Clipboard
                            </button>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <div class="status error">❌ Error: ${data.error}</div>
                    `;
                }
                
            } catch (error) {
                resultDiv.innerHTML = `
                    <div class="status error">❌ Connection error: ${error.message}</div>
                `;
            }
        }
        
        function downloadScript(script, request) {
            const blob = new Blob([script], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `manim_${request.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase()}.py`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
        
        function copyScript(script) {
            navigator.clipboard.writeText(script).then(() => {
                alert('📋 Script copied to clipboard!');
            }).catch(err => {
                console.error('Failed to copy: ', err);
                alert('❌ Failed to copy script');
            });
        }
        
        // Allow Enter key to generate
        document.getElementById('request').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                generateScript();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    """Generate script endpoint."""
    global agent, validator
    
    if not agent:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        data = request.json
        user_request = data.get('request', '').strip()
        
        if not user_request:
            return jsonify({'error': 'No request provided'}), 400
        
        # Generate script
        script = agent.generate_script(user_request)
        
        # Validate script
        is_valid, fixed_script, report = validator.validate_and_fix(script)
        
        final_script = fixed_script if is_valid else script
        
        return jsonify({
            'script': final_script,
            'is_valid': is_valid,
            'request': user_request,
            'validation_report': report
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': agent is not None
    })

if __name__ == '__main__':
    print("🌐 Starting Manim Script Generator Web Interface")
    print("=" * 50)
    
    # Check if model files exist
    model_files = ['best_model_epoch_1.pth', 'best_model_epoch_5.pth', 'best_model_epoch_10.pth', 'final_manim_model.pth']
    model_exists = any(os.path.exists(f) for f in model_files)
    
    if not model_exists:
        print("❌ No trained model found!")
        print("Please train the model first: python train_model.py")
        exit(1)
    
    # Initialize model
    if initialize_model():
        print("✅ Model initialized successfully!")
        print("🌐 Web interface starting...")
        print("📍 Open http://localhost:5000 in your browser")
        print("🛑 Press Ctrl+C to stop")
        
        app.run(debug=True, port=5000, host='0.0.0.0')
    else:
        print("❌ Failed to initialize model")
        print("Please make sure the model is trained: python train_model.py")