# 🚀 Enhanced Multi-Domain LLM - Usage Guide

## ✅ **Clean Structure Overview**

Your codebase is now professionally organized with **NO duplicate files** in the root directory. Everything is properly structured in appropriate folders.

```
enhanced-manim-llm/
├── 📁 src/                    # All source code (organized by function)
├── 📁 scripts/               # Entry point scripts  
├── 📁 docs/                  # Documentation
├── 📁 data/                  # Training data & tokenizers
├── 📁 checkpoints/          # Model checkpoints
├── 📁 tests/                # Test files
├── 🔧 setup.py              # Package installation
├── 📋 README.md             # Main documentation
└── 🧪 validate_structure.py # Structure validator
```

## 🎯 **Quick Start (3 Options)**

### **Option 1: Professional Installation (Recommended)**

```bash
# Install as Python package (handles all dependencies)
pip install -e .

# Train enhanced model
manim-llm-train --model enhanced --auto

# Start API server  
manim-llm-api --server fixed --port 5001
```

### **Option 2: Direct Script Usage**

```bash
# Install dependencies
pip install torch numpy tqdm matplotlib

# Train model
python scripts/train.py --model enhanced --auto

# Start API
python scripts/api.py --server fixed --port 5001
```

### **Option 3: Windows-Compatible**

```bash
# For Windows systems with Unicode issues
python scripts/train.py --model enhanced --windows --auto
```

## 📚 **Import Structure (For Developers)**

The new organized structure uses proper Python imports:

```python
# Model components
from src.models import EnhancedManimAgent, MultiDomainLLM
from src.models.enhanced_tokenizer import EnhancedManimTokenizer

# Training components  
from src.training import EnhancedModelTrainer, EnhancedDataGenerator

# Knowledge bases
from src.knowledge import get_knowledge_base, MultiDomainKnowledgeBase

# Utilities
from src.utils import ScriptValidator
```

## 🎨 **Example Usage**

### **Training a New Model**

```bash
# Full enhanced training (15K samples, 3 domains)
python scripts/train.py --model enhanced --epochs 15 --auto

# Custom training parameters
python scripts/train.py --model enhanced --batch-size 4 --epochs 20

# Windows-compatible training
python scripts/train.py --model enhanced --windows --auto
```

### **Using the API**

```bash
# Start API server
python scripts/api.py --server fixed --port 5001

# Test API
curl -X POST http://localhost:5001/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create a rotating blue circle", "validate": true}'
```

### **Programmatic Usage**

```python
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from models.enhanced_agent import EnhancedManimAgent

# Initialize agent
agent = EnhancedManimAgent(
    llm_provider="enhanced",
    model_path="checkpoints/enhanced_best_model.pth", 
    tokenizer_path="data/enhanced_tokenizer.pkl"
)

# Generate responses
manim_script = agent.generate_script("Create a sine wave animation")
dsa_explanation = agent.generate_script("Explain quicksort algorithm")
system_design = agent.generate_script("Design a scalable chat system")
```

## 🛠️ **Available Commands**

### **Training Commands**
```bash
# Enhanced model training
python scripts/train.py --model enhanced [OPTIONS]

# Original model training  
python scripts/train.py --model original [OPTIONS]

# Training options:
#   --auto              Run without prompts
#   --windows           Use Windows-compatible version
#   --epochs N          Number of training epochs
#   --batch-size N      Training batch size
#   --skip-data         Skip data generation if exists
#   --skip-tokenizer    Skip tokenizer building if exists
```

### **API Commands**
```bash
# Start API server
python scripts/api.py --server TYPE [OPTIONS]

# Server types:
#   --server fixed      Enhanced API server (recommended)
#   --server original   Original API server
#   --server simple     Simple HTTP server (no Flask)

# API options:
#   --port N           Server port (default: 5001)
#   --host HOST        Server host (default: localhost)
```

### **Validation Commands**
```bash
# Validate project structure
python validate_structure.py

# Install package
pip install -e .

# Install with GPU support
pip install -e ".[gpu]"

# Install development dependencies
pip install -e ".[dev]"
```

## 🎯 **Key Features**

### **✅ Properly Organized**
- **No duplicate files** in root directory
- **Clear separation** of concerns
- **Standard Python** package structure
- **Professional layout** for easy maintenance

### **✅ Cross-Platform Compatible**
- **Windows-specific** versions for Unicode issues
- **Linux/Unix** standard versions
- **Automatic detection** of system capabilities

### **✅ Multiple Domains**
- **🎨 Manim** - Mathematical animations
- **🧠 DSA** - Data structures & algorithms  
- **🏗️ System Design** - Architecture patterns

### **✅ Easy Installation**
- **Standard pip** installation
- **Entry point scripts** for common tasks
- **Dependency management** via setup.py

## 📊 **File Organization**

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `src/models/` | LLM models, agents, tokenizers | `enhanced_model.py`, `enhanced_agent.py` |
| `src/training/` | Training scripts, data generation | `enhanced_trainer.py`, `enhanced_data_generator.py` |
| `src/api/` | API servers, web interfaces | `fixed_api_server.py`, `web_app.py` |
| `src/knowledge/` | Domain knowledge bases | `multi_domain_knowledge_base.py` |
| `src/utils/` | Utilities, validation, testing | `validator.py`, `examples.py` |
| `src/config/` | Configuration, requirements | `requirements.txt` |
| `scripts/` | Entry point scripts | `train.py`, `api.py` |
| `docs/` | Documentation | `README_ENHANCED.md`, `TRAINING_GUIDE.md` |
| `data/` | Training data, tokenizers | `*.json`, `*.pkl` |
| `checkpoints/` | Model checkpoints | `*.pth` |

## 🚀 **Migration from Old Structure**

If you had the old flat structure, your existing commands now map to:

```bash
# Old → New
python full_training_pipeline.py --auto → python scripts/train.py --model enhanced --auto
python fixed_api_server.py → python scripts/api.py --server fixed
python windows_compatible_training.py --auto → python scripts/train.py --model enhanced --windows --auto
```

## 🎉 **Benefits of New Structure**

✅ **Professional** - Industry-standard Python package layout  
✅ **Clean** - No duplicate files, proper organization  
✅ **Maintainable** - Easy to find, modify, and extend code  
✅ **Installable** - Standard pip installation process  
✅ **Documented** - Comprehensive guides and examples  
✅ **Cross-Platform** - Works on Windows and Linux  
✅ **Extensible** - Easy to add new domains and features  

---

## 🏁 **Ready to Use!**

Your Enhanced Multi-Domain LLM is now professionally structured and ready for:

1. **Training**: `python scripts/train.py --model enhanced --auto`
2. **API Usage**: `python scripts/api.py --server fixed`
3. **Development**: `pip install -e .` then import from `src/`

**Enjoy your clean, professional AI codebase!** 🎯