import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab
import pandas as pd
from matplotlib.ticker import FuncFormatter
import ssl
import os
sns.set_context('poster')

def f1_state_frequency(network, state):
    STM = np.zeros((11, 10))
    for i in range(10):
        for j in range(11):
            # state_trans_df = pd.read_csv('/Volumes/A/MTech Project/Boolean/Boolean.jl-main/Sig_LS/Sig_' + str(j) + '/' + str(i+1) + '/' + network + '_finFlagFreq.csv')
            state_trans_df = pd.read_csv(os.getcwd() + '/' + network + '_Sig_Str_'+ str(j) + '_LS_' + str(i+1) + '_finFlagFreq.csv')
            for k in range(len(state_trans_df)):
                if state == state_trans_df.states[k]:
                    STM[10 - j,i] = state_trans_df.Avg0[k]
    return STM


TS_STM = f1_state_frequency('Toggle_Switch', "'101'")
# TS_SI_STM = f1_state_frequency('Toggle_Switch_Self_Inhibition', "'101'")
# TS_SA_STM = f1_state_frequency('Toggle_Switch_Self_Activation', "'101'")
TTri_STM = f1_state_frequency('Toggle_Triad', "'1001'")
# TTri_SA_STM = f1_state_frequency('Toggle_Triad_Self_Activation', "'1001'")
# TTri_SI_STM = f1_state_frequency('Toggle_Triad_Self_Inhibition', "'1001'")
TTetra_STM = f1_state_frequency('Toggle_Tetrahedron', "'10001'")
# TTetra_SI_STM = f1_state_frequency('Toggle_Tetrahedron_Self_Inhibition', "'10001'")
# TTetra_SA_STM = f1_state_frequency('Toggle_Tetrahedron_Self_Activation', "'10001'")
TP_STM = f1_state_frequency('Toggle_Pentagon', "'100001'")
# TP_SA_STM = f1_state_frequency('Toggle_Pentagon_Self_Activation', "'100001'")
# TP_SI_STM = f1_state_frequency('Toggle_Pentagon_Self_Inhibition', "'100001'")
THex_STM = f1_state_frequency('Toggle_Hexagon', "'1000001'")
# THex_SI_STM = f1_state_frequency('Toggle_Hexagon_Self_Inhibition', "'1000001'")
# THex_SA_STM = f1_state_frequency('Toggle_Hexagon_Self_Activation', "'1000001'")
THept_STM = f1_state_frequency('Toggle_Heptagon', "'10000001'")
# THept_SA_STM = f1_state_frequency('Toggle_Heptagon_Self_Activation', "'10000001'")
# THept_SI_STM = f1_state_frequency('Toggle_Heptagon_Self_Inhibition', "'10000001'")
TO_STM = f1_state_frequency('Toggle_Octagon', "'100000001'")
# TO_SI_STM = f1_state_frequency('Toggle_Octagon_Self_Inhibition', "'100000001'")
# TO_SA_STM = f1_state_frequency('Toggle_Octagon_Self_Activation', "'100000001'")


TS_df = pd.DataFrame(data = TS_STM, index = ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0'], columns = ['1', '2', '3' ,'4', '5', '6', '7', '8', '9', '10'])
TTri_df = pd.DataFrame(data = TTri_STM, index = ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0'], columns = ['1', '2', '3' ,'4', '5', '6', '7', '8', '9', '10'])
TTetra_df = pd.DataFrame(data = TTetra_STM, index = ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0'], columns = ['1', '2', '3' ,'4', '5', '6', '7', '8', '9', '10'])
TP_df = pd.DataFrame(data = TP_STM, index = ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0'], columns = ['1', '2', '3' ,'4', '5', '6', '7', '8', '9', '10'])
THex_df = pd.DataFrame(data = THex_STM, index = ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0'], columns = ['1', '2', '3' ,'4', '5', '6', '7', '8', '9', '10'])
THept_df = pd.DataFrame(data = THept_STM, index = ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0'], columns = ['1', '2', '3' ,'4', '5', '6', '7', '8', '9', '10'])
TO_df = pd.DataFrame(data = TO_STM, index = ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0'], columns = ['1', '2', '3' ,'4', '5', '6', '7', '8', '9', '10'])


plt.figure(1)
heatmap = sns.heatmap(TS_df, annot=True,cmap='Reds')
heatmap.set_xlabel("Edge weight")
heatmap.set_ylabel("Strength of signalling")
cbar = heatmap.collections[0].colorbar
plt.title('T2')
plt.tight_layout()
plt.savefig('/Volumes/A/MTech Project/Boolean/Boolean.jl-main/Sig_LS/Results/TS_Sig_LS.svg')

plt.figure(2)
heatmap = sns.heatmap(TTri_df, annot=True,cmap='Reds')
heatmap.set_xlabel("Edge weight")
heatmap.set_ylabel("Strength of signalling")
cbar = heatmap.collections[0].colorbar
plt.title('T3')
plt.tight_layout()
plt.show()
#plt.savefig('/Volumes/A/MTech Project/Boolean/Boolean.jl-main/Sig_LS/Results/TTri_Sig_LS.svg')

plt.figure(3)
heatmap = sns.heatmap(TTetra_df, annot=True,cmap='Reds')
# heatmap.set_xlabel("Edge weight")
# heatmap.set_ylabel("Strength of signalling")
cbar = heatmap.collections[0].colorbar
plt.title('T4')
plt.tight_layout()
plt.show()
# plt.savefig('/Volumes/A/MTech Project/Boolean/Boolean.jl-main/Sig_LS/Results/TTetra_Sig_LS.svg')

plt.figure(4)
heatmap = sns.heatmap(TP_df, annot=True,cmap='Reds')
# heatmap.set_xlabel("Edge weight")
# heatmap.set_ylabel("Strength of signalling")
cbar = heatmap.collections[0].colorbar
plt.title('T5')
plt.tight_layout()
plt.show()
# plt.savefig('/Volumes/A/MTech Project/Boolean/Boolean.jl-main/Sig_LS/Results/TP_Sig_LS.svg')


plt.figure(5)
heatmap = sns.heatmap(THex_df, annot=True,cmap='Reds')
heatmap.set_xlabel("Edge weight")
heatmap.set_ylabel("Strength of signalling")
cbar = heatmap.collections[0].colorbar
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.title('T6')
plt.tight_layout()
plt.show()
# plt.savefig('/Volumes/A/MTech Project/Boolean/Boolean.jl-main/Sig_LS/Results/THex_Sig_LS.svg')


plt.figure(6)
heatmap = sns.heatmap(THept_df, annot=True,cmap='Reds')
heatmap.set_xlabel("Edge weight")
heatmap.set_ylabel("Strength of signalling")
cbar = heatmap.collections[0].colorbar
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.title('T7')
plt.tight_layout()
plt.show()
# plt.savefig('/Volumes/A/MTech Project/Boolean/Boolean.jl-main/Sig_LS/Results/THept_Sig_LS.svg')

plt.figure(7)
heatmap = sns.heatmap(TO_df, annot=True,cmap='Reds')
heatmap.set_xlabel("Edge weight")
heatmap.set_ylabel("Strength of signalling")
cbar = heatmap.collections[0].colorbar
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.title('T8')
plt.tight_layout()
plt.show()
# plt.savefig('/Volumes/A/MTech Project/Boolean/Boolean.jl-main/Sig_LS/Results/TO_Sig_LS.svg')

categories = ['T3', 'T4', 'T5', 'T6', 'T7', 'T8']
# xpos = range(len(categories))
# ypos = [2, 3, 4, 5, 6, 7]
# zpos = [3, 2, 3, 2, 3, 2]

def rel_freq_thrd_1(STM):
    x = []
    y = []
    for i in range(STM.shape[0]):
        for j in range(STM.shape[1]):
            if STM[i, j] == 1:
                y.append(10-i)
                x.append(j+1)
                break
    for i in range(STM.shape[1]):
        for j in range(STM.shape[0]):
            if STM[10-j,i] == 1:
                y.append(j)
                x.append(i+1)
                break
    return x,y

TTri_X, TTri_Y = rel_freq_thrd_1(TTri_STM)
TTetra_X, TTetra_Y = rel_freq_thrd_1(TTetra_STM)
TP_X, TP_Y = rel_freq_thrd_1(TP_STM)
THex_X, THex_Y = rel_freq_thrd_1(THex_STM)
THept_X, THept_Y = rel_freq_thrd_1(THept_STM)
TO_X, TO_Y = rel_freq_thrd_1(TO_STM)


plt.figure(8)
# plt.figure(figsize=(5,5))
# plt.plot(TTri_X,TTri_Y, label = categories[0])
# plt.plot(TTetra_X, TTetra_Y, label = categories[1])
plt.plot(TP_X, TP_Y, label = categories[2])
plt.plot(THex_X, THex_Y, label = categories[3])
plt.plot(THept_X, THept_Y, label = categories[4])
plt.plot(TO_X, TO_Y, label = categories[5])
#plt.axis('equal')
plt.xticks([1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10])
plt.yticks([1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10])
plt.xlabel('Edge weight')
plt.ylabel('Strength of signaling')
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig('/Volumes/A/MTech Project/Boolean/Boolean.jl-main/Sig_LS/Results/Sig_Line_Plot.svg')
