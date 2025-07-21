#!/usr/bin/env python3
"""
Main API server entry point.
Supports multiple API server configurations.
"""

import os
import sys
import argparse

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    parser = argparse.ArgumentParser(description='Start Manim LLM API Server')
    parser.add_argument('--server', choices=['fixed', 'original', 'simple'], 
                       default='fixed', help='API server type')
    parser.add_argument('--port', type=int, default=5001, help='Server port')
    parser.add_argument('--host', default='localhost', help='Server host')
    
    args = parser.parse_args()
    
    if args.server == 'fixed':
        from api.fixed_api_server import app
        app.run(host=args.host, port=args.port, debug=True)
    elif args.server == 'original':
        from api.api_server import app
        app.run(host=args.host, port=args.port, debug=True)
    elif args.server == 'simple':
        from api.simple_http_server import main as server_main
        sys.argv = ['simple_http_server.py', '--port', str(args.port)]
        server_main()

if __name__ == "__main__":
    main()