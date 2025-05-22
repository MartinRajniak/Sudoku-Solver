#!/bin/bash

# 1> output.log redirects all standard output to output.log
# 2> error.log redirects all stderr to error.log
python main.py --experiment "complete_dataset" --trial 4 1> output.log 2> error.log