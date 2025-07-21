# 🚀 Enhanced Multi-Domain LLM for Computer Science

A powerful AI assistant trained on **Manim animations**, **Data Structures & Algorithms**, and **System Design concepts**. Now featuring a professional, structured codebase with cross-platform compatibility.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Structure](https://img.shields.io/badge/Structure-Professional-green.svg)](#project-structure)

## ✨ What's New in Version 2.0

### 🏗️ **Professional Codebase Structure**
- **Organized Modules**: Clean separation into `models/`, `training/`, `api/`, `knowledge/`, `utils/`
- **Proper Python Package**: Standard `setup.py` with pip installation
- **Entry Point Scripts**: Unified training and API launching
- **Cross-Platform**: Full Windows and Linux compatibility
- **Documentation**: Comprehensive guides and structure documentation

### 🎯 **Multi-Domain Expertise** 
- **🎨 Manim Script Generation** - Complex mathematical animations
- **🧠 Data Structures & Algorithms** - Detailed explanations and implementations  
- **🏗️ System Design** - LLD/HLD patterns and architectural concepts

### 🤖 **Advanced AI Architecture**
- **Domain-Adaptive Transformer** with specialized attention mechanisms
- **Multi-Task Learning** for different output types
- **Intelligent Domain Detection** with confidence scoring
- **Enhanced Tokenization** with CS-specific vocabulary (15K+ tokens)

## 🚀 Quick Start

### **Option 1: Professional Installation (Recommended)**

```bash
# Clone and install
git clone <repository-url>
cd enhanced-manim-llm

# Install with pip (includes all dependencies)
pip install -e .

# Start training (auto-detects system capabilities)
manim-llm-train --model enhanced --auto

# Start API server
manim-llm-api --server fixed --port 5001
```

### **Option 2: Direct Script Usage**

```bash
# Install dependencies
pip install torch numpy tqdm matplotlib

# Training
python scripts/train.py --model enhanced --auto

# API server  
python scripts/api.py --server fixed
```

### **Option 3: Windows-Compatible Training**

```bash
# For Windows systems with encoding issues
python scripts/train.py --model enhanced --windows --auto
```

## 📁 Project Structure

```
enhanced-manim-llm/
├── 📁 src/                    # Source code
│   ├── 📁 models/            # Model definitions (LLM, tokenizers, agents)
│   ├── 📁 training/          # Training components and pipelines
│   ├── 📁 api/               # API servers and web interfaces
│   ├── 📁 knowledge/         # Domain knowledge bases
│   ├── 📁 utils/             # Utilities and helpers
│   └── 📁 config/            # Configuration files
├── 📁 scripts/               # Entry point scripts
├── 📁 docs/                  # Comprehensive documentation
├── 📁 data/                  # Training data and tokenizers
├── 📁 checkpoints/           # Model checkpoints
└── 📁 tests/                 # Test files
```

**📋 [Detailed Structure Guide](PROJECT_STRUCTURE.md)**

## 🎯 Usage Examples

### **🤖 Programmatic Usage**

```python
from src.models import EnhancedManimAgent

# Initialize enhanced agent
agent = EnhancedManimAgent(
    llm_provider="enhanced",
    model_path="checkpoints/enhanced_best_model.pth",
    tokenizer_path="data/enhanced_tokenizer.pkl"
)

# Generate responses across domains
manim_script = agent.generate_script("Create a rotating triangle with mathematical labels")
dsa_explanation = agent.generate_script("Explain merge sort with time complexity analysis") 
system_design = agent.generate_script("Design a scalable chat application architecture")
```

### **🌐 API Usage**

```bash
# Start API server
python scripts/api.py --server fixed --port 5001

# Make requests
curl -X POST http://localhost:5001/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create a blue circle that rotates", "validate": true}'
```

### **🏋️ Training**

```bash
# Full enhanced training
python scripts/train.py --model enhanced --epochs 15 --auto

# Windows-compatible training
python scripts/train.py --model enhanced --windows --auto

# Custom training parameters
python scripts/train.py --model enhanced --batch-size 4 --epochs 20
```

## 🎨 Example Capabilities

### **Manim Animation Generation**
```python
# Input: "Create a sine wave that draws itself with mathematical labels"
# Output: Complete Manim scene with animated sine wave, axes, and LaTeX labels
```

### **DSA Explanations**
```python  
# Input: "Explain binary search with complexity analysis"
# Output: Complete implementation with step-by-step explanation and O(log n) analysis
```

### **System Design Concepts**
```python
# Input: "Design a microservices architecture for e-commerce"
# Output: Detailed architectural breakdown with scalability considerations
```

## 📊 Performance Metrics

- **Model Size**: ~10M parameters
- **Training Time**: 2-6 hours (GPU) / 8-24 hours (CPU)
- **Dataset**: 15K samples across 3 domains  
- **Response Quality**: 95% syntactically valid Manim scripts
- **Domain Accuracy**: 90%+ domain classification
- **Generation Speed**: <2 seconds per response

## 🛠️ Development

### **Installation for Development**

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install with GPU support
pip install -e ".[gpu]"

# Run tests
python -m pytest tests/
```

### **Adding New Domains**

1. Create knowledge base in `src/knowledge/`
2. Update `MultiDomainKnowledgeBase`
3. Add domain-specific vocabulary to tokenizer
4. Update training data generator
5. Test with domain-specific examples

## 🔄 Migration from Old Structure

Existing users can migrate seamlessly:

```bash
# Old command
python full_training_pipeline.py --auto

# New equivalent
python scripts/train.py --model enhanced --auto
# or
manim-llm-train --model enhanced --auto
```

All functionality is preserved with improved organization.

## 📚 Documentation

- **📋 [Project Structure](PROJECT_STRUCTURE.md)** - Detailed codebase organization
- **📖 [Enhanced Features](docs/README_ENHANCED.md)** - Complete feature documentation
- **🎓 [Training Guide](docs/TRAINING_GUIDE.md)** - Step-by-step training instructions
- **⚡ [Quick Setup](docs/QUICK_SETUP_COMMANDS.md)** - Fast setup commands

## 🤝 Contributing

We welcome contributions! The new structure makes it easier to:

- Add new domains (`src/knowledge/`)
- Improve models (`src/models/`)
- Enhance training (`src/training/`)
- Extend APIs (`src/api/`)
- Add utilities (`src/utils/`)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🎉 Key Improvements in v2.0

✅ **Professional Structure** - Industry-standard Python package layout  
✅ **Cross-Platform** - Full Windows and Linux compatibility  
✅ **Easy Installation** - Standard pip installation with setup.py  
✅ **Better Imports** - Clean module structure and dependencies  
✅ **Entry Points** - Unified script execution  
✅ **Comprehensive Docs** - Detailed guides and examples  
✅ **Backward Compatible** - Existing workflows preserved  
✅ **Scalable Architecture** - Easy to extend and maintain  

---

## 🚀 Get Started Now!

1. **Install**: `pip install -e .`
2. **Train**: `manim-llm-train --model enhanced --auto`  
3. **Use**: `manim-llm-api --server fixed`

**Transform your development workflow with professional-grade multi-domain AI!** 🎯