import sys
sys.path.append("..")
import os
from agent import ManimAgent

agent = ManimAgent('custom')
script = agent.generate_script("Create a blue circle")
print(script)

print("CWD:", os.getcwd())
print("Files in CWD:", os.listdir(".."))

model_path = "best_model_epoch_10.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")