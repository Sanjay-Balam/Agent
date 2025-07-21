# Quick Setup Commands

## Step 1: Install Git LFS (Windows)

**Option A - Download and Install:**
1. Go to: https://git-lfs.github.io/
2. Download and run the installer
3. Restart PowerShell

**Option B - Using Package Manager (if you have Chocolatey):**
```powershell
choco install git-lfs
```

## Step 2: Run These Commands in PowerShell

```powershell
# Navigate to your manim_agent directory
cd "C:\Users\ivect\Desktop\Practice\manim_agent"

# Initialize Git LFS
git lfs install

# Initialize Git repository
git init

# Configure Git LFS to track large files
git lfs track "*.pth"
git lfs track "*.pkl"

# Verify what LFS is tracking
git lfs track

# Configure Git user
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add all files (LFS will handle the large ones)
git add .

# Create initial commit
git commit -m "🎬 Initial commit: Manim AI Agent

🤖 Features:
- Custom trained LLM (19M+ parameters)  
- Flask API backend with CORS
- Next.js frontend with modern UI
- Intelligent script generation & fallbacks
- Git LFS for model file management

📦 Model Files (LFS):
- best_model_epoch_10.pth (~233MB)
- best_model_epoch_1.pth (~233MB) 
- best_model_epoch_2.pth (~233MB)
- tokenizer.pkl (~10KB)

🚀 Ready to generate Manim scripts from natural language!"

# Check file sizes and LFS status
git lfs ls-files
```

## Step 3: Create GitHub Repository

1. Go to GitHub.com
2. Click "New repository"
3. Name: `manim-ai-agent`
4. Description: `AI-powered Manim script generator with custom LLM`
5. Public/Private (your choice)
6. **Don't** initialize with README (we have one)
7. Click "Create repository"

## Step 4: Connect and Push

```powershell
# Add GitHub remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/manim-ai-agent.git

# Push to GitHub (this uploads everything including LFS files)
git push -u origin main
```

## ✅ Verification

After pushing, verify everything worked:

```powershell
# Check LFS files were uploaded
git lfs ls-files

# Should show:
# 28a8b1c8f7... * best_model_epoch_1.pth
# 4f5d3a2b1c... * best_model_epoch_2.pth  
# 9e7c5d4f2a... * best_model_epoch_10.pth
# a1b2c3d4e5... * tokenizer.pkl
```

## 🎯 Result

✅ **Complete working repository on GitHub**  
✅ **Large model files handled efficiently with LFS**  
✅ **Anyone can clone and get the full working system**  
✅ **No file size issues or missing models**

## For Others to Clone:

```bash
git clone https://github.com/YOUR_USERNAME/manim-ai-agent.git
cd manim-ai-agent

# Install dependencies and run
pip install -r requirements.txt
python api_server.py
```

The model files will download automatically with LFS! 🎉

# for training
 python windows_compatible_training.py --auto

 python windows_enhanced_trainer.py