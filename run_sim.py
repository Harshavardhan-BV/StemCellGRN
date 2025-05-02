#!/usr/bin/env python3
import os
import glob
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Run simulation')
parser.add_argument('--n_thr', type=int, default=-1, help='Number of threads')
parser.add_argument('topo', type=str, default='toggle-n', help='Topology')
args = parser.parse_args()

def clear_output():
    # Clear the output folder
    os.system('rm -rf *.topo *_finFlagFreq.csv *_nodes.txt *_rand/')

def cp_output(topo):
    # Copy the output to the output folder
    os.makedirs('Output', exist_ok=True)
    if (topo == 'rand-n') or (topo =='tcell-rand'):
        os.system('cp -r *_rand/ Output/')
    elif (topo== 'tcell') or (topo== 'tcell-embedded') or (topo== 'tcell-cyt'):
        os.system('cp *_finFlagFreq.csv Output/')
        os.system('cp *_nodes.txt Output/')
    else:
        os.system('cp *_finFlagFreq.csv Output/')

def cp_topo(topo):
    # Check if the topology is valid
    if args.topo not in maptop.keys():
        raise ValueError('Invalid topology, choose from:', list(maptop.keys()))
    else:
        topo = maptop[args.topo]
    # Copy the topology files
    os.system(f'cp ./TOPO/{topo} .')

def run_sim(topo, n_thr):
    if (topo == 'rand-n') or (topo =='tcell-rand'):
        # Run the script_rand.jl
        cmd = f'julia -t {n_thr} script_rand.jl'
    elif  (topo == 'tcell-cyt'):
         cmd = f'julia -t {n_thr} script_sig.jl'
    else:
        cmd = f'julia -t {n_thr} script.jl'
    subprocess.run(cmd, shell=True)

# Automatically set the number of threads if not provided
if args.n_thr <= 0:
    args.n_thr = os.cpu_count()-2

maptop = {'toggle-n': 'T[1-9].topo',
          'rand-n': 'T[1-9].topo',
          'impure-n': 'Impure*.topo',
          'embedded-n': 'Embedded_T[1-9].topo',
          'team-n': 'Team*.topo',
          'tcell': 'TCellDiff*.topo',
          'tcell-rand': 'TCellDiff*.topo',
          'tcell-embedded': 'Embedded_TCellDiff*.topo',
          'tcell-cyt': 'Cyt_T[1-9]*.topo'
          }

print('Running simulation for', args.topo)

clear_output()
cp_topo(args.topo)
run_sim(args.topo, args.n_thr)
cp_output(args.topo)
clear_output()

