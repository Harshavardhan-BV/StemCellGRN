import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab
import pandas as pd
import ssl
import os
sns.set_context('poster')

TTri_Boolean_init = pd.read_csv(os.getcwd() + '/Toggle_Triad_initFinFlagFreq.csv');
TTri_SA_Boolean_init = pd.read_csv(os.getcwd() + '/Toggle_Triad_Self_Activation_initFinFlagFreq.csv');

TTetra_Boolean_init = pd.read_csv(os.getcwd() + '/Toggle_Tetrahedron_initFinFlagFreq.csv');
TTetra_SA_Boolean_init = pd.read_csv(os.getcwd() + '/Toggle_Tetrahedron_Self_Activation_initFinFlagFreq.csv');
TTetra_SI_Boolean_init = pd.read_csv(os.getcwd() + '/Toggle_Tetrahedron_Self_Inhibition_initFinFlagFreq.csv');

TP_Boolean_init = pd.read_csv(os.getcwd() + '/Toggle_Pentagon_initFinFlagFreq.csv');
TP_SA_Boolean_init = pd.read_csv(os.getcwd() + '/Toggle_Pentagon_Self_Activation_initFinFlagFreq.csv');

THex_Boolean_init = pd.read_csv(os.getcwd() + '/Toggle_Hexagon_initFinFlagFreq.csv');
THex_SA_Boolean_init = pd.read_csv(os.getcwd() + '/Toggle_Hexagon_Self_Activation_initFinFlagFreq.csv');
THex_SI_Boolean_init = pd.read_csv(os.getcwd() + '/Toggle_Hexagon_Self_Inhibition_initFinFlagFreq.csv');

THept_Boolean_init = pd.read_csv(os.getcwd() + '/Toggle_Heptagon_initFinFlagFreq.csv');
THept_SA_Boolean_init = pd.read_csv(os.getcwd() + '/Toggle_Heptagon_Self_Activation_initFinFlagFreq.csv');

TO_Boolean_init = pd.read_csv(os.getcwd() + '/Toggle_Octagon_initFinFlagFreq.csv');
TO_SA_Boolean_init = pd.read_csv(os.getcwd() + '/Toggle_Octagon_Self_Activation_initFinFlagFreq.csv');
TO_SI_Boolean_init = pd.read_csv(os.getcwd() + '/Toggle_Octagon_Self_Inhibition_initFinFlagFreq.csv');



TTri_Boolean = pd.read_csv(os.getcwd() + '/Toggle_Triad_FinFlagFreq.csv')
TTri_SA_Boolean = pd.read_csv(os.getcwd() + '/Toggle_Triad_Self_Activation_FinFlagFreq.csv');

TTetra_Boolean = pd.read_csv(os.getcwd() + '/Toggle_Tetrahedron_FinFlagFreq.csv');
TTetra_SA_Boolean = pd.read_csv(os.getcwd() + '/Toggle_Tetrahedron_Self_Activation_FinFlagFreq.csv');
TTetra_SI_Boolean = pd.read_csv(os.getcwd() + '/Toggle_Tetrahedron_Self_Inhibition_FinFlagFreq.csv');

TP_Boolean = pd.read_csv(os.getcwd() + '/Toggle_Pentagon_FinFlagFreq.csv');
TP_SA_Boolean = pd.read_csv(os.getcwd() + '/Toggle_Pentagon_Self_Activation_FinFlagFreq.csv');

THex_Boolean = pd.read_csv(os.getcwd() + '/Toggle_Hexagon_FinFlagFreq.csv');
THex_SA_Boolean = pd.read_csv(os.getcwd() + '/Toggle_Hexagon_Self_Activation_FinFlagFreq.csv');
THex_SI_Boolean = pd.read_csv(os.getcwd() + '/Toggle_Hexagon_Self_Inhibition_FinFlagFreq.csv');

THept_Boolean = pd.read_csv(os.getcwd() + '/Toggle_Heptagon_FinFlagFreq.csv');
THept_SA_Boolean = pd.read_csv(os.getcwd() + '/Toggle_Heptagon_Self_Activation_FinFlagFreq.csv');

TO_Boolean = pd.read_csv(os.getcwd() + '/Toggle_Octagon_FinFlagFreq.csv');
TO_SA_Boolean = pd.read_csv(os.getcwd() + '/Toggle_Octagon_Self_Activation_FinFlagFreq.csv');
TO_SI_Boolean = pd.read_csv(os.getcwd() + '/Toggle_Octagon_Self_Inhibition_FinFlagFreq.csv');

def state_trans_matrix(state_trans_df, unique_df):
    STM = np.zeros((len(unique_df), len(unique_df)))
    for i in range(len(unique_df)):
        for j in range(len(state_trans_df)):
            if (unique_df[i] == state_trans_df.init[j]):
                for k in range(len(unique_df)):
                    if (state_trans_df.states[j] == unique_df[k]):
                        STM[k,i] = state_trans_df.Avg0[j]
    STM = STM/np.sum(STM, axis=0)
    return STM

def hamming_vs_relfreq(state_trans_df, init_state, max_hamming_distance):
    distance = np.arange(0, max_hamming_distance+1)
    rel_freq = np.zeros(distance.shape[0])
    for i in range(len(state_trans_df)):
        if (init_state == state_trans_df.init[i]):
            hamming_distance = 0
            for j in range(len(init_state)):
                if (init_state[j] != state_trans_df.states[i][j]):
                    hamming_distance = hamming_distance + 1
            rel_freq[hamming_distance] = rel_freq[hamming_distance] + state_trans_df.Avg0[i]
    return distance, rel_freq

def hamming_vs_relfreq_swarm(state_trans_df, init_state):
    data = []
    sum_rel_freq = 0
    for i in range(len(state_trans_df)):
        if (init_state == state_trans_df.init[i]):
            sum_rel_freq = sum_rel_freq + state_trans_df.Avg0[i]
    for i in range(len(state_trans_df)):
        if init_state == state_trans_df.init[i]:
            hamming_distance = 0
            for j in range(len(init_state)):
                if (init_state[j] != state_trans_df.states[i][j]):
                    hamming_distance = hamming_distance + 1
            data.append((hamming_distance, state_trans_df.Avg0[i]/sum_rel_freq))
    return data

TTri_single_positive = []
TTri_double_positive = []
for i in range(len(TTri_SA_Boolean)):
    count = 0
    for j in range(3):
        if int(TTri_SA_Boolean.states[i][j+1]) == 1:
            count = count + 1
    if count == 1:
        TTri_single_positive.append(TTri_SA_Boolean.states[i])
    if count == 2:
        TTri_double_positive.append(TTri_SA_Boolean.states[i])

TTetra_single_positive = []
TTetra_double_positive = []
TTetra_triple_positive = []
for i in range(len(TTetra_SA_Boolean)):
    count = 0
    for j in range(4):
        if int(TTetra_SA_Boolean.states[i][j+1]) == 1:
            count = count + 1
    if count == 1:
        TTetra_single_positive.append(TTetra_SA_Boolean.states[i])
    if count == 2:
        TTetra_double_positive.append(TTetra_SA_Boolean.states[i])
    if count == 3:
        TTetra_triple_positive.append(TTetra_SA_Boolean.states[i])


TP_double_positive = []
TP_triple_positive = []
for i in range(len(TP_SA_Boolean)):
    count = 0
    for j in range(5):
        if int(TP_SA_Boolean.states[i][j+1]) == 1:
            count = count + 1
    if count == 2:
        TP_double_positive.append(TP_SA_Boolean.states[i])
    if count == 3:
        TP_triple_positive.append(TP_SA_Boolean.states[i])


THex_double_positive = []
THex_triple_positive = []
THex_tetra_positive = []
for i in range(len(THex_SA_Boolean)):
    count = 0
    for j in range(6):
        if int(THex_SA_Boolean.states[i][j+1]) == 1:
            count = count + 1
    if count == 2:
        THex_double_positive.append(THex_SA_Boolean.states[i])
    if count == 3:
        THex_triple_positive.append(THex_SA_Boolean.states[i])
    if count == 4:
        THex_tetra_positive.append(THex_SA_Boolean.states[i])


THept_triple_positive = []
THept_tetra_positive = []
for i in range(len(THept_SA_Boolean)):
    count = 0
    for j in range(7):
        if int(THept_SA_Boolean.states[i][j+1]) == 1:
            count = count + 1
    if count == 3:
        THept_triple_positive.append(THept_SA_Boolean.states[i])
    if count == 4:
        THept_tetra_positive.append(THept_SA_Boolean.states[i])


TO_triple_positive = []
TO_tetra_positive = []
TO_penta_positive = []
for i in range(len(TO_SA_Boolean)):
    count = 0
    for j in range(8):
        if int(TO_SA_Boolean.states[i][j+1]) == 1:
            count = count + 1
    if count == 3:
        TO_triple_positive.append(TO_SA_Boolean.states[i])
    if count == 4:
        TO_tetra_positive.append(TO_SA_Boolean.states[i])
    if count == 5:
        TO_penta_positive.append(TO_SA_Boolean.states[i])


states_TTri = ["'000'"] + TTri_single_positive + TTri_double_positive + ["'111'"]
TTri_STM = state_trans_matrix(TTri_Boolean_init, states_TTri)
states_TTri_SA = ["'000'"] + TTri_single_positive + TTri_double_positive + ["'111'"]
TTri_SA_STM = state_trans_matrix(TTri_SA_Boolean_init, states_TTri_SA)
#TTri_SI_STM = state_trans_matrix(TTri_SI_Boolean_init, TTri_SI_Boolean)


states_TTetra = ["'0000'"] + TTetra_double_positive + ["'1111'"]
TTetra_STM = state_trans_matrix(TTetra_Boolean_init, states_TTetra)
states_TTetra_SA = ["'0000'"] + TTetra_single_positive + TTetra_double_positive + TTetra_triple_positive + ["'1111'"]
TTetra_SA_STM = state_trans_matrix(TTetra_SA_Boolean_init, states_TTetra_SA)
states_TTetra_SI = ["'0000'"] + TTetra_double_positive + ["'1111'"]
TTetra_SI_STM = state_trans_matrix(TTetra_SI_Boolean_init, states_TTetra_SI)

states_TP = ["'00000'"] + TP_double_positive + TP_triple_positive + ["'11111'"]
TP_STM = state_trans_matrix(TP_Boolean_init, states_TP)
states_TP_SA = ["'00000'"] + TP_double_positive + TP_triple_positive + ["'11111'"]
TP_SA_STM = state_trans_matrix(TP_SA_Boolean_init, states_TP_SA)
#TP_SI_STM = state_trans_matrix(TP_SI_Boolean_init, TP_SI_Boolean)

states_THex = ["'000000'"] + THex_triple_positive + ["'111111'"]
THex_STM = state_trans_matrix(THex_Boolean_init, states_THex)
states_THex_SA = ["'000000'"] + THex_double_positive + THex_triple_positive + THex_tetra_positive + ["'111111'"]
THex_SA_STM = state_trans_matrix(THex_SA_Boolean_init, states_THex_SA)
states_THex_SI = ["'000000'"] + THex_triple_positive + ["'111111'"]
THex_SI_STM = state_trans_matrix(THex_SI_Boolean_init, states_THex_SI)


states_THept = ["'0000000'"] + THept_triple_positive + THept_tetra_positive + ["'1111111'"]
THept_STM = state_trans_matrix(THept_Boolean_init, states_THept)
states_THept_SA = ["'0000000'"] + THept_triple_positive + THept_tetra_positive + ["'1111111'"]
THept_SA_STM = state_trans_matrix(THept_SA_Boolean_init, states_THept_SA)
#THept_SI_STM = state_trans_matrix(THept_SI_Boolean_init, THept_SI_Boolean)

states_TO = ["'00000000'"] + TO_tetra_positive + ["'11111111'"]
TO_STM = state_trans_matrix(TO_Boolean_init, states_TO)
states_TO_SA = ["'00000000'"] + TO_triple_positive + TO_tetra_positive + TO_penta_positive + ["'11111111'"]
TO_SA_STM = state_trans_matrix(TO_SA_Boolean_init, states_TO_SA)
states_TO_SI = ["'00000000'"] + TO_tetra_positive + ["'11111111'"]
TO_SI_STM = state_trans_matrix(TO_SI_Boolean_init, states_TO_SI)

TTri_diatance_1, TTri_rel_freq_1 = hamming_vs_relfreq(TTri_Boolean_init, "'100'", 3)
TTri_diatance_2, TTri_rel_freq_2 = hamming_vs_relfreq(TTri_Boolean_init, "'110'", 3)
TTetra_diatance, TTetra_rel_freq = hamming_vs_relfreq(TTetra_Boolean_init, "'1100'", 4)
TP_diatance_2, TP_rel_freq_2 = hamming_vs_relfreq(TP_Boolean_init, "'11000'", 5)
TP_diatance_3, TP_rel_freq_3 = hamming_vs_relfreq(TP_Boolean_init, "'11100'", 5)
THex_distance, THex_rel_freq = hamming_vs_relfreq(THex_Boolean_init, "'111000'", 6)
THept_distance_3, THept_rel_freq_3 = hamming_vs_relfreq(THex_Boolean_init, "'1110000'", 7)
THept_distance_4, THept_rel_freq_4 = hamming_vs_relfreq(THex_Boolean_init, "'1111000'", 7)
TO_distance_4, TO_rel_freq_4 = hamming_vs_relfreq(TO_Boolean_init, "'11110000'", 8)



plt.figure(1)
data_1 = hamming_vs_relfreq_swarm(TO_Boolean_init, "'11110000'")
#data_2 = hamming_vs_relfreq_swarm(TP_Boolean_init, "'11100'")
# Create a DataFrame from the generated data

swarm_data_1 = pd.DataFrame(data_1, columns=['Hamming Distance', 'Value'])
#swarm_data_2 = pd.DataFrame(data_2, columns=['Hamming Distance', 'Value'])

# Create the swarm plot
sns.swarmplot(x='Hamming Distance', y='Value', data=swarm_data_1, color='blue', label='Tetra positive initial state',size=10)
#sns.swarmplot(x='Hamming Distance', y='Value', data=swarm_data_2, color='red', label='Triple positive initial state',size=10)
# Adding labels and title
plt.xlabel('Hamming distance')
plt.ylabel('Relative frequency')
plt.xlim(-0.5,4.5)
plt.xticks([0, 1, 2, 3, 4], [0, 2, 4, 6, 8])
plt.title('T8')
blue_patch = plt.Line2D([0], [0], marker='o', color='blue', label='Tetra positive initial state')
#red_patch = plt.Line2D([0], [0], marker='o', color='red', label='Triple positive initial state')
plt.legend(handles=[blue_patch])
plt.show()
# plt.savefig('/Volumes/A/MTech Project/Manuscript/TO_Hamming.svg')
# Display the plot


plt.figure(2)
data_1 = hamming_vs_relfreq_swarm(TTetra_Boolean_init, "'1100'")
#data_2 = hamming_vs_relfreq_swarm(TP_Boolean_init, "'11100'")
# Create a DataFrame from the generated data

swarm_data_1 = pd.DataFrame(data_1, columns=['Hamming Distance', 'Value'])
#swarm_data_2 = pd.DataFrame(data_2, columns=['Hamming Distance', 'Value'])

# Create the swarm plot
sns.swarmplot(x='Hamming Distance', y='Value', data=swarm_data_1, color='blue', label='Double positive initial state',size=10)
#sns.swarmplot(x='Hamming Distance', y='Value', data=swarm_data_2, color='red', label='Triple positive initial state',size=10)
# Adding labels and title
plt.xlabel('Hamming distance')
plt.ylabel('Relative frequency')
plt.xlim(-0.5,2.5)
plt.xticks([0, 1, 2], [0, 2, 4])
plt.title('T4')
blue_patch = plt.Line2D([0], [0], marker='o', color='blue', label='Double positive initial state')
#red_patch = plt.Line2D([0], [0], marker='o', color='red', label='Triple positive initial state')
plt.legend(handles=[blue_patch])
plt.show()


plt.figure(3)
data_1 = hamming_vs_relfreq_swarm(THex_Boolean_init, "'111000'")
#data_2 = hamming_vs_relfreq_swarm(TP_Boolean_init, "'11100'")
# Create a DataFrame from the generated data

swarm_data_1 = pd.DataFrame(data_1, columns=['Hamming Distance', 'Value'])
#swarm_data_2 = pd.DataFrame(data_2, columns=['Hamming Distance', 'Value'])

# Create the swarm plot
sns.swarmplot(x='Hamming Distance', y='Value', data=swarm_data_1, color='blue', label='Triple positive initial state',size=10)
#sns.swarmplot(x='Hamming Distance', y='Value', data=swarm_data_2, color='red', label='Triple positive initial state',size=10)
# Adding labels and title
plt.xlabel('Hamming distance')
plt.ylabel('Relative frequency')
plt.xlim(-0.5,2.5)
plt.xticks([0, 1, 2, 3], [0, 2, 4, 6])
plt.title('T6')
blue_patch = plt.Line2D([0], [0], marker='o', color='blue', label='Triple positive initial state')
#red_patch = plt.Line2D([0], [0], marker='o', color='red', label='Triple positive initial state')
plt.legend(handles=[blue_patch])
plt.show()


plt.figure(4)
data_1 = hamming_vs_relfreq_swarm(TTri_Boolean_init, "'100'")
data_2 = hamming_vs_relfreq_swarm(TTri_Boolean_init, "'110'")
# Create a DataFrame from the generated data

swarm_data_1 = pd.DataFrame(data_1, columns=['Hamming Distance', 'Value'])
swarm_data_2 = pd.DataFrame(data_2, columns=['Hamming Distance', 'Value'])

# Create the swarm plot
sns.swarmplot(x='Hamming Distance', y='Value', data=swarm_data_1, color='blue', label='Single positive initial state',size=10)
sns.swarmplot(x='Hamming Distance', y='Value', data=swarm_data_2, color='red', label='Double positive initial state',size=10)
# Adding labels and title
plt.xlabel('Hamming distance')
plt.ylabel('Relative frequency')
plt.xlim(-0.5,3.5)
plt.xticks([0, 1, 2, 3], [0, 1, 2, 3])
plt.title('T3')
blue_patch = plt.Line2D([0], [0], marker='o', color='blue', label='Single positive initial state')
red_patch = plt.Line2D([0], [0], marker='o', color='red', label='Double positive initial state')
plt.legend(handles=[blue_patch, red_patch])
plt.show()


plt.figure(5)
data_1 = hamming_vs_relfreq_swarm(TP_Boolean_init, "'11000'")
data_2 = hamming_vs_relfreq_swarm(TP_Boolean_init, "'11100'")
# Create a DataFrame from the generated data

swarm_data_1 = pd.DataFrame(data_1, columns=['Hamming Distance', 'Value'])
swarm_data_2 = pd.DataFrame(data_2, columns=['Hamming Distance', 'Value'])

# Create the swarm plot
sns.swarmplot(x='Hamming Distance', y='Value', data=swarm_data_1, color='blue', label='Double positive initial state',size=10)
sns.swarmplot(x='Hamming Distance', y='Value', data=swarm_data_2, color='red', label='Triple positive initial state',size=10)
# Adding labels and title
plt.xlabel('Hamming distance')
plt.ylabel('Relative frequency')
plt.xlim(-0.5,5.5)
plt.xticks([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
plt.title('T5')
blue_patch = plt.Line2D([0], [0], marker='o', color='blue', label='Double positive initial state')
red_patch = plt.Line2D([0], [0], marker='o', color='red', label='Triple positive initial state')
plt.legend(handles=[blue_patch, red_patch])
plt.show()


plt.figure(6)
data_1 = hamming_vs_relfreq_swarm(THept_Boolean_init, "'1110000'")
data_2 = hamming_vs_relfreq_swarm(THept_Boolean_init, "'1111000'")
# Create a DataFrame from the generated data

swarm_data_1 = pd.DataFrame(data_1, columns=['Hamming Distance', 'Value'])
swarm_data_2 = pd.DataFrame(data_2, columns=['Hamming Distance', 'Value'])

# Create the swarm plot
sns.swarmplot(x='Hamming Distance', y='Value', data=swarm_data_1, color='blue', label='Triple positive initial state',size=10)
sns.swarmplot(x='Hamming Distance', y='Value', data=swarm_data_2, color='red', label='Tetra positive initial state',size=10)
# Adding labels and title
plt.xlabel('Hamming distance')
plt.ylabel('Relative frequency')
plt.xlim(-0.5,7.5)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7])
plt.title('T7')
blue_patch = plt.Line2D([0], [0], marker='o', color='blue', label='Triple positive initial state')
red_patch = plt.Line2D([0], [0], marker='o', color='red', label='Tetra positive initial state')
plt.legend(handles=[blue_patch, red_patch])
plt.show()
# x = np.arange(len(TTri_diatance_1))
# bar_width = 0.35
# plt.bar(x - bar_width/2, TTri_rel_freq_1, width=bar_width, label='Single positive initial state', color = 'blue')
# plt.bar(x + bar_width/2, TTri_rel_freq_2, width=bar_width, label='Double positive initial state', color = 'red')
# plt.legend()
# plt.xlabel('Hamming distance from the initial state', fontsize = 20)
# plt.ylabel('Relative frequency', fontsize = 20)
# plt.xticks(x, TTri_diatance_1, fontsize=16)
# plt.yticks(fontsize=16)
# plt.show()
#
# plt.bar(TTetra_diatance, TTetra_rel_freq, width=0.35, label = 'Double positive initial state', color = 'blue')
# plt.legend()
# plt.xlabel('Hamming distance from the initial state', fontsize = 20)
# plt.ylabel('Relative frequency', fontsize = 20)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.show()
#



plt.figure(7)
TTetra_df = pd.DataFrame(data = TTetra_STM, index = states_TTetra, columns = states_TTetra)
heatmap = sns.heatmap(TTetra_df, linewidths=0.5, linecolor='black',cmap = 'Reds')
heatmap.set_xlabel("Initial state")
heatmap.set_ylabel("Final state")
cbar = heatmap.collections[0].colorbar
plt.title('T4')
# cbar.ax.tick_params(labelsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
plt.show()
#plt.savefig('/Volumes/A/MTech Project/Manuscript/TTetra_STM.svg')



plt.figure(8)
TTri_df = pd.DataFrame(data = TTri_STM, index = states_TTri, columns = states_TTri)
heatmap = sns.heatmap(TTri_df, linewidths=0.5, linecolor='black',cmap = 'Reds')
heatmap.set_xlabel("Initial state")
heatmap.set_ylabel("Final state")
cbar = heatmap.collections[0].colorbar
plt.title('T3')
# cbar.ax.tick_params(labelsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
plt.show()

plt.figure(9)
TP_df = pd.DataFrame(data = TP_STM, index = states_TP, columns = states_TP)
heatmap = sns.heatmap(TP_df, linewidths=0.5, linecolor='black',cmap = 'Reds')
heatmap.set_xlabel("Initial state")
heatmap.set_ylabel("Final state")
cbar = heatmap.collections[0].colorbar
plt.title('T5')
# cbar.ax.tick_params(labelsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
plt.show()


plt.figure(10)
THex_df = pd.DataFrame(data = THex_STM, index = states_THex, columns = states_THex)
heatmap = sns.heatmap(THex_df, linewidths=0.5, linecolor='black',cmap = 'Reds')
heatmap.set_xlabel("Initial state")
heatmap.set_ylabel("Final state")
cbar = heatmap.collections[0].colorbar
plt.title('T6')
# cbar.ax.tick_params(labelsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
plt.show()


plt.figure(11)
THept_df = pd.DataFrame(data = THept_STM, index = states_THept, columns = states_THept)
heatmap = sns.heatmap(THept_df, linewidths=0.5, linecolor='black',cmap = 'Reds')
heatmap.set_xlabel("Initial state")
heatmap.set_ylabel("Final state")
cbar = heatmap.collections[0].colorbar
plt.title('T7')
# cbar.ax.tick_params(labelsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
plt.show()


plt.figure(12)
TO_df = pd.DataFrame(data = TO_STM, index = states_TO, columns = states_TO)
heatmap = sns.heatmap(TO_df, linewidths=0.5, linecolor='black',cmap = 'Reds')
heatmap.set_xlabel("Initial state")
heatmap.set_ylabel("Final state")
cbar = heatmap.collections[0].colorbar
plt.title('T8')
# cbar.ax.tick_params(labelsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
plt.show()