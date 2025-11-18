# Programmable VNFPC

This repository contains artefacts from the paper submitted to **NOMS 2026** (IEEE/IFIP Network Operations and Management Symposium).

## Paper Information
- **Conference**: NOMS 2026
- **Website**: https://noms2026.ieee-noms.org/
- **Paper Title**: Mind the Path: Virtual Network Function Placement and Chaining in a Programmable Data Plane Era

## Overview

This project implements and evaluates programmable Virtual Network Function Placement and Chaining (VNFPC) solutions using both Deep Q-Network (DQN) and Mixed Integer Linear Programming (MILP) approaches.

## Requirements

- Python 3.x
- pip
- git (optional but recommended)
- venv (optional but recommended)

## Installation

### Step 1: Clone the Repository

Option 1: Using git (recommended)
```bash

git clone https://github.com/wevertoncordeiro/programmable-vnfpc.git
cd programmable-vnfpc

```
Option 2: Download manually from GitHub and extract to directory

### Step 2: Configure Environment
Option 1: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Option 2: Without Virtual Environment
```bash
# Install dependencies directly
pip install -r requirements.txt
```

## Running Experiments

### Complete Workflow
Run the complete experimental workflow with the following commands:
```bash
# Run DQN and MILP solvers for 5 different seeds
for i in 1 2 3 4 5; do
    ./run_solver_dqn.sh inputs/inputs_seed$i
    ./run_solver_milp.sh inputs/inputs_seed$i
done

# Create results directory and copy outputs
mkdir results_tmp
cp outputs_dqn/*.json results_tmp/
cp outputs_milp/*.json results_tmp/

# Generate plots
python plot.py results_tmp
```

### Step-by-Step Execution
You can also run components individually:

### Run Solvers for Specific Seed
```bash
# Run DQN solver for seed 1
./run_solver_dqn.sh inputs/inputs_seed1

# Run MILP solver for seed 1  
./run_solver_milp.sh inputs/inputs_seed1
```


### Collect Results
```bash
# Create results directory
mkdir results_tmp

# Copy DQN results
cp outputs_dqn/*.json results_tmp/

# Copy MILP results
cp outputs_milp/*.json results_tmp/
```

### Generate Plots
```bash
# Generate plots from results
python plot.py results_tmp
```

### Generate other workloads
See instructions on wokload_generator/
```bash
cd workload_generator/
pip instal -r requirements.txt
#download Brite from https://www.cs.bu.edu/brite/download.html and install on ./BRITE directory
python3 workload_generator.py -h #to obtain instructions
```

## Project Structure
```text
programmable-vnfpc/
├── inputs/                 # Input data files
│   ├── inputs_seed1/      # Input files for seed 1
│   ├── inputs_seed2/      # Input files for seed 2
│   ├── inputs_seed3/      # Input files for seed 3
│   ├── inputs_seed4/      # Input files for seed 4
│   └── inputs_seed5/      # Input files for seed 5
├── outputs_dqn/           # DQN solver output files (.json)
├── outputs_milp/          # MILP solver output files (.json)
├── run_solver_dqn.sh      # DQN solver execution script
├── run_solver_milp.sh     # MILP solver execution script
├── plot.py               # Results visualization and plotting script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

