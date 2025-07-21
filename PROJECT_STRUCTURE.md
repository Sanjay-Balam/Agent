# 📁 Enhanced Multi-Domain LLM - Project Structure

This document describes the organized folder structure of the Enhanced Multi-Domain LLM project.

## 📂 Directory Structure

```
enhanced-manim-llm/
├── 📁 src/                          # Source code
│   ├── 📁 models/                   # Model definitions
│   │   ├── __init__.py
│   │   ├── model.py                 # Original Manim LLM
│   │   ├── enhanced_model.py        # Enhanced Multi-Domain LLM
│   │   ├── tokenizer.py             # Original tokenizer
│   │   ├── enhanced_tokenizer.py    # Enhanced tokenizer
│   │   ├── windows_enhanced_tokenizer.py  # Windows-compatible tokenizer
│   │   ├── agent.py                 # Original agent
│   │   ├── enhanced_agent.py        # Enhanced multi-domain agent
│   │   ├── inference.py             # Original inference
│   │   └── enhanced_inference.py    # Enhanced inference
│   │
│   ├── 📁 training/                 # Training components
│   │   ├── __init__.py
│   │   ├── train_model.py           # Original training script
│   │   ├── trainer.py               # Original trainer
│   │   ├── enhanced_trainer.py      # Enhanced trainer
│   │   ├── windows_enhanced_trainer.py     # Windows-compatible trainer
│   │   ├── data_generator.py        # Original data generator
│   │   ├── enhanced_data_generator.py      # Enhanced data generator
│   │   ├── evaluate_model.py        # Original evaluator
│   │   ├── enhanced_evaluator.py    # Enhanced evaluator
│   │   ├── start_training.py        # Quick training starter
│   │   ├── full_training_pipeline.py       # Complete training pipeline
│   │   ├── windows_compatible_training.py  # Windows-compatible pipeline
│   │   └── windows_training_pipeline.py   # Windows training pipeline
│   │
│   ├── 📁 api/                      # API and web components
│   │   ├── __init__.py
│   │   ├── api_server.py            # Original API server
│   │   ├── fixed_api_server.py      # Enhanced API server
│   │   ├── simple_http_server.py    # Simple HTTP server fallback
│   │   ├── web_app.py               # Web interface
│   │   └── simple_api_test.py       # API testing script
│   │
│   ├── 📁 knowledge/                # Knowledge bases
│   │   ├── __init__.py
│   │   ├── knowledge_base.py        # Original Manim knowledge base
│   │   ├── dsa_knowledge_base.py    # Data Structures & Algorithms
│   │   ├── system_design_knowledge_base.py  # System Design patterns
│   │   └── multi_domain_knowledge_base.py   # Unified knowledge base
│   │
│   ├── 📁 utils/                    # Utilities and helpers
│   │   ├── __init__.py
│   │   ├── validator.py             # Script validation
│   │   ├── generate_sample.py       # Sample generation
│   │   ├── batch_generate.py        # Batch generation
│   │   ├── examples.py              # Example scripts
│   │   ├── test_system.py           # System testing
│   │   └── install_and_train.py     # Installation helper
│   │
│   └── 📁 config/                   # Configuration files
│       ├── __init__.py
│       ├── requirements.txt         # Python dependencies
│       └── training_requirements.txt # Training-specific dependencies
│
├── 📁 scripts/                      # Entry point scripts
│   ├── train.py                     # Training entry point
│   └── api.py                       # API server entry point
│
├── 📁 docs/                         # Documentation
│   ├── README.md                    # Original README
│   ├── README_ENHANCED.md           # Enhanced README
│   ├── TRAINING_GUIDE.md           # Training guide
│   ├── GIT_SETUP.md                # Git setup guide
│   └── QUICK_SETUP_COMMANDS.md     # Quick setup commands
│
├── 📁 data/                         # Training data and tokenizers
│   ├── *.json                       # Training/validation data files
│   └── *.pkl                        # Tokenizer files
│
├── 📁 checkpoints/                  # Model checkpoints
│   └── *.pth                        # PyTorch model files
│
├── 📁 tests/                        # Test files
│   ├── create_a_blue_circle_that_moves_in_a_figure-8_patt.py
│   ├── generate_manim.py
│   └── test-new.py
│
├── 🔧 setup.py                      # Python package setup
├── 📋 PROJECT_STRUCTURE.md          # This file
└── 📋 README.md                     # Main project README
```

## 🎯 Key Components

### 🤖 Models (`src/models/`)
- **Enhanced Model**: Multi-domain transformer with domain adaptation
- **Original Model**: Basic Manim-only transformer
- **Tokenizers**: CS-specific vocabulary handling
- **Agents**: High-level interfaces for script generation
- **Inference**: Model inference pipelines

### 🏋️ Training (`src/training/`)
- **Data Generators**: Create training datasets across domains
- **Trainers**: Model training with multi-task learning
- **Evaluators**: Performance assessment and metrics
- **Pipelines**: Complete training orchestration
- **Windows Compatibility**: Special versions for Windows systems

### 🌐 API (`src/api/`)
- **Fixed API Server**: Enhanced Flask API with multi-domain support
- **Simple HTTP Server**: Fallback HTTP server without Flask
- **Web Interface**: Browser-based interaction
- **Testing Tools**: API endpoint testing

### 📚 Knowledge (`src/knowledge/`)
- **Manim Knowledge**: Animation templates and patterns
- **DSA Knowledge**: Data structures and algorithms
- **System Design**: Architectural patterns and concepts
- **Multi-Domain**: Unified knowledge management

### 🛠️ Utils (`src/utils/`)
- **Validation**: Script syntax and quality checking
- **Generation**: Sample and example creation
- **Testing**: System and component tests
- **Installation**: Setup and dependency management

## 🚀 Usage

### Installation
```bash
# Install the package
pip install -e .

# Or install with GPU support
pip install -e ".[gpu]"
```

### Training
```bash
# Enhanced model training
python scripts/train.py --model enhanced --auto

# Windows-compatible training
python scripts/train.py --model enhanced --windows --auto

# Original model training
python scripts/train.py --model original
```

### API Server
```bash
# Start enhanced API server
python scripts/api.py --server fixed --port 5001

# Start simple HTTP server
python scripts/api.py --server simple --port 8000
```

### Direct Module Usage
```python
from src.models import EnhancedManimAgent, MultiDomainLLM
from src.knowledge import get_knowledge_base
from src.training import EnhancedModelTrainer

# Initialize agent
agent = EnhancedManimAgent(llm_provider="enhanced")

# Generate script
script = agent.generate_script("Create a blue circle that rotates")
```

## 📋 Migration from Old Structure

The new structure maintains backward compatibility while providing:

1. **Better Organization**: Clear separation of concerns
2. **Easier Imports**: Proper Python package structure
3. **Entry Points**: Standardized script execution
4. **Documentation**: Comprehensive guides and examples
5. **Testing**: Organized test structure
6. **Configuration**: Centralized settings management

## 🔄 Backward Compatibility

Old scripts will continue to work with minimal changes:
- Most import statements updated automatically
- Entry point scripts provide same functionality
- Configuration files maintained in new locations
- Data and checkpoint paths preserved

## 🎉 Benefits

- **Professional Structure**: Industry-standard Python package layout
- **Easy Installation**: Standard pip installation with setup.py
- **Better Imports**: Clear module boundaries and dependencies
- **Documentation**: Comprehensive guides and examples
- **Cross-Platform**: Windows and Linux compatibility
- **Scalable**: Easy to extend with new domains and features