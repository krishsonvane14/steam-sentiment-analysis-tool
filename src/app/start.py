import subprocess
import sys
import os
import webbrowser

# Use python3 ./start.py in src/app to start both the frontend and backend

backend_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.main:app", "--reload"],
    cwd=".",  
)

base_dir = os.path.dirname(__file__)
frontend_folder = os.path.join(base_dir, "frontend")
print("Contents:", os.listdir(frontend_folder))
frontend_proc = subprocess.Popen(
    [sys.executable, "-m", "http.server", "5500"],
    cwd=frontend_folder 
)
webbrowser.open("http://localhost:5500/index.html")


print("Backend and frontend started.")

# backend_proc.wait()
# frontend_proc.wait()
