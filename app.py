import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stbesa.gradio_app import app

if __name__ == "__main__":
    print("Starting ST-BESA Platform...")
    print("Please check the console for the local URL (usually http://127.0.0.1:7860)")
    app.launch(inbrowser=True)
