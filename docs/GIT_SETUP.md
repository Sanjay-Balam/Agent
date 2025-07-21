# Git LFS Setup Guide for Manim Agent

## Step 1: Install Git LFS

### On Windows:
1. **Download Git LFS**: Go to https://git-lfs.github.io/
2. **Install the executable** or use package manager:
   ```powershell
   # Using Chocolatey
   choco install git-lfs
   
   # Or using Scoop
   scoop install git-lfs
   
   # Or download from: https://github.com/git-lfs/git-lfs/releases
   ```

### On Linux/WSL:
```bash
# Ubuntu/Debian
sudo apt install git-lfs

# CentOS/RHEL
sudo yum install git-lfs
```

### On macOS:
```bash
# Using Homebrew
brew install git-lfs
```

## Step 2: Set Up Repository with LFS

After installing Git LFS, run these commands in your PowerShell/Terminal:

```bash
# Navigate to your manim_agent directory
cd "C:\Users\ivect\Desktop\Practice\manim_agent"

# Initialize Git LFS (one-time setup)
git lfs install

# Initialize Git repository if not already done
git init

# Configure Git LFS to track large model files
git lfs track "*.pth"
git lfs track "*.pkl"

# Verify LFS tracking
git lfs track

# Add the .gitattributes file (this is created automatically by LFS)
git add .gitattributes

# Add all files (including large model files)
git add .

# Configure Git user (replace with your info)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Create initial commit
git commit -m "Initial commit: Manim AI Agent with LFS for model files

🤖 Features:
- Custom trained LLM for Manim script generation  
- Flask API backend with CORS support
- Next.js frontend with modern UI
- Model files handled with Git LFS
- Script validation and intelligent fallbacks

📦 Includes:
- Trained model files (19M+ parameters)
- Complete inference pipeline
- Web interface components
- API documentation"

# Add GitHub remote (replace with your repository URL)
git remote add origin https://github.com/YOUR_USERNAME/manim-ai-agent.git

# Push to GitHub (this will upload large files to LFS)
git push -u origin main
```

## Step 3: Verify LFS Setup

After pushing, verify that LFS is working:

```bash
# Check LFS files
git lfs ls-files

# Should show:
# best_model_epoch_1.pth
# best_model_epoch_2.pth  
# best_model_epoch_10.pth
# tokenizer.pkl
```

## Step 4: Create GitHub Repository

1. **Go to GitHub.com**
2. **Click "New repository"**
3. **Repository name**: `manim-ai-agent` (or your preferred name)
4. **Description**: "AI-powered Manim script generator with custom LLM"
5. **Set to Public** (or Private if you prefer)
6. **Don't initialize** with README, .gitignore, or license (we already have them)
7. **Click "Create repository"**

## Important Notes:

### ✅ **What Gets Uploaded:**
- All your Python code (.py files)
- Configuration files
- README and documentation
- **Large model files** (handled efficiently by LFS)
- Frontend Next.js code

### 🎯 **LFS Benefits:**
- Model files are versioned but don't bloat Git history
- Fast cloning (models download separately when needed)
- Efficient storage on GitHub
- Anyone who clones gets the complete working system

### 📋 **For Others Who Clone:**
```bash
git clone https://github.com/YOUR_USERNAME/manim-ai-agent.git
cd manim-ai-agent

# LFS files download automatically, but you can also:
git lfs pull

# Then install dependencies and run:
pip install -r requirements.txt
python api_server.py
```

## Troubleshooting:

### If LFS installation fails:
1. Make sure Git is updated to latest version
2. Restart your terminal/PowerShell after installing LFS
3. Try manual installation from: https://github.com/git-lfs/git-lfs/releases

### If files are too large for GitHub:
- GitHub LFS has a 2GB per file limit
- Your model files (~233MB each) are well within limits
- Total LFS storage: 1GB free, then $5/month for 50GB

## File Sizes:
- `best_model_epoch_1.pth`: ~233MB ✅
- `best_model_epoch_2.pth`: ~233MB ✅  
- `best_model_epoch_10.pth`: ~233MB ✅
- `tokenizer.pkl`: ~10KB ✅

All files are within GitHub LFS limits!