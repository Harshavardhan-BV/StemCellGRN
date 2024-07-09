# StemCellGRN
Boolean simulations of toggle-n networks to represent stem cell differentiation.

## Requirements
- Some codes present in other repositories are given as submodules. 
    - If you are using the code for the first time, you can clone the repository using the following command:
    ```bash
    git clone --recurse-submodules https://github.com/Harshavardhan-BV/StemCellGRN.git
    ```
    - If you have already cloned the repository, you can update the submodules using the following command:
    ```bash
    git submodule update --init --recursive
    ```
- Python 
    - [requirements.txt](./requirements.txt)
    ```bash
    pip install -r requirements.txt
    ```
- julia
    - [dependencies.jl](./dependencies.jl)
    ```bash
    cd Boolean.jl
    julia dependencyInstaller.jl
    ```

## Structure of the repo
- [TOPO](./TOPO): Contains the topology files for the networks
- [Output](./Output): Contains the output files generated from the simulations. (Note: The output files for Embedded networks is too large to be uploaded to the repository and can accessed [here](https://drive.google.com/file/d/1SDYOVoQ6g1Czj7pzK_fXcqQCCpDE8cU2/view?usp=sharing) instead)
- [Analysis](./Analysis): Contains the scripts for the analysis of the output files
- [Analysed_data](./Analysed_data): Contains the analysed data files generated from the analysis scripts
- [figures](./figures): Contains the figures generated from the analysis scripts
- [writing](./writing): Contains the main figures and latex files for the manuscript

The simulations for signalling and state transitions are performed by Hanuma Sai. The structure is not compatible with the rest of the repo. Refer to:
- [Sig_ActA_LS](./Sig_ActA_LS) for simulations/analysis for toggle-n networks with cytokine signalling that activates A
- [Sig_ActA_Inh_LS](./Sig_ActA_LS) for simulations/analysis for toggle-n networks with cytokine signalling that activates A and inhibits the rest
- [Transdiff_p](./Transdiff_p) for simulations/analysis for state transition dynamics between the steady states of toggle-n networks

## Usage
### 1. Network generation
Generate the topology files for the networks and save them in the [TOPO](./TOPO) folder. Generates the following:
- toggle-n: Mutually inhibitory networks with n nodes. `n` = 2, 3, ..., 8 with no self-regulation, self-activation and self-inhibition. Alter the `range` if you want networks of different sizes
- team-n: Fully connected networks with n teams with within team activation and across team inhibition. `ng.team_n` generates where members in a team are equal. `ng.team_n_rand` generates where members in a team are randomly split by multinomial sampling. No. of teams, `n` = 2, 3,..., 8 and Avg. no. of members in a team, `m` = 2, ..., 10.
- embedded-n: Toggle-n networks embedded in random networks. No. of nodes in toggle-n network, `n` = 2, 3, ..., 8, No. of nodes in random network, `size` = 10, 15, 20 and Avg. number of edges per node, `density` = 2, 4, 6.
- impure-n: n-node networks (no self reg) with inhibition replaced by activations (impurity) selecting only the non-isomorphic networks. No. of nodes, n = 2, 3, ..., 8. The maximum number of netwerks generated for each impurity, `nnets` = 50 for `n` = 3, 4, 5 and 100 for `n` = 6. 

Refer to docstrings (or tooltips of the functions in your IDE) of (net_gen.py)[./RASSPE/net_gen.py] for details about the parameters and adjust if necessary.
```bash
python SC_net_topo.py
```

### 2. Perform simulations
Run scripts to perform the simulations for the different types of networks. Enter the simulation type as an argument. The following simulation_types are available:
- toggle-n 
- impure-n
- embedded-n 
- team-n
- rand-n: Toggle-n networks with edge-weights randomly sampled from U(0, 1). The min and max can be changed in the `script_rand.jl` file.
```bash
python run_sim.py <simulation_type>
```

### 3. Analyse the data
```bash
cd Analysis
```
#### 3.1. Plot the networks (Figure 1A/1B/S1A/S1B)
- Plots the toggle-n network as graphs.
- Plots the adjacency matrices.
```bash
cd Analysis
python plot_network.py
```
- Note: The plotting function for graphs with self-regulation is not stable and requires manual tweaking of parameters.
- The schematic plots in Figure S4 are made manually with inkscape using elements generated from the above script.

#### 3.2 Toggle-n (Figure 1C/S1C/S2C/2A)
- Iterates through the output files for toggle-n networks and calculate the frequency of $n_{high} = {0,1,2,...n}$ states for each network. 
- Plots the frequency of $n_{high}$ states for each network. 
- Saves the dataframe with frequencies for each $n_{high}$ for all the networks.
- Plots the F(k) vs n with k = 1 for all networks, k = n/2, n/2+1, n/2-1 for n even and k = (n+1)/2, (n-1)/2 for n odd.
```bash
python Fn.py
```

#### 3.3 Random edge-weights (Figure 2B)
- Analysis and plotting for the edge-weights simulations.
```bash
python Fn_rand.py
```

#### 3.4 Toggle-n with impurities (Figure 3B)
- Analysis and plotting for the edge-weights simulations.
```bash
python Fn_impure.py
```

#### 3.5 Toggle-n embedded in random networks (Figure 3)
- Analysis and plotting for the embedded networks simulations.
```bash
python Fn_embedded.py
```
- Plotting with statistical tests (Warning: statannotation requires older python and dependency versions)
```bash
python Fn_embedded_statannot.py
```

#### 3.6 Team-n networks (Figure 4)
- Analysis and plotting for the team-n networks simulations.
```bash
python Fn_team.py
```

#### 3.7 Signalling (Figure 5)
Refer to [Sig_ActA_LS](./Sig_ActA_LS) and [Sig_ActA_Inh_LS](./Sig_ActA_LS) for the analysis scripts.

#### 3.8 State transition analysis (Figure S3)
Refer to [Transdiff_p](./Transdiff_p) for the analysis scripts.