import argparse
import json
import os
from datasets import load_dataset
dataset = load_dataset("BioDEX/BioDEX-Reactions", split="test")

print(dataset)