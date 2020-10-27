from pathlib import Path
from config import FACESWAP_PATH

import subprocess

class Model:

    def __init__(self):
        pass

    def center_box(self, row, col, image):
        pass

    def extract(self, input_dir, output_dir):
        """Extracts faces from photos"""
        subprocess.run([
            "python",
            FACESWAP_PATH,
            "extract",
            "-i",
            input_dir,
            "-o",
            output_dir
        ])
        
    
    def convert(self, input_dir, output_dir):
        """Swaps the faces from the input photos to the output photos"""
        subprocess.run([
            "python",
            FACESWAP_PATH,
            "convert",
            "-i",
            input_dir,
            "-o",
            output_dir
        ])

