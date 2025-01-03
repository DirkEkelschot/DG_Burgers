
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import seaborn as sns


edge_points = np.loadtxt('ref_basis_points.out')

nodal_l = edge_points[:,0]
nodal_r = edge_points[:,1]

modal_l = edge_points[:,2]
modal_r = edge_points[:,3]

plt.figure(1)
nodal          = np.loadtxt('new_nodal_basis.out')
nodal_deriv    = np.loadtxt('new_diff_nodal_basis.out')


mako_cmap = sns.color_palette("plasma", as_cmap=True)


xnodal_nodes = nodal[0]
print(xnodal_nodes)

nummodes = len(nodal)-1
P        = nummodes - 1
nq       = len(nodal[0])
num_lines      = nummodes;
colors         = [mako_cmap((i) / num_lines) for i in range(num_lines)]

basis_sum = np.zeros((nq,1))
for i in range(0,nummodes):
    for j in range(0,nq):
        basis_sum[j] = basis_sum[j]+1.0*nodal[i+1][j];



for i in range(0,nummodes):
    # plt.plot(xnodal_nodes,nodal_deriv[i],'-',color=colors[i],linewidth=3)
    plt.plot(xnodal_nodes,nodal[i+1],'-',color=colors[i])

plt.plot(xnodal_nodes,basis_sum,'--',color='k')

for i in range(0,len(nodal_l)):
    plt.plot(-0.25,nodal_l[i],'ok')
    plt.plot(0.25,nodal_r[i],'ok')







plt.figure(2)
nodal          = np.loadtxt('new_modal_basis.out')
nodal_deriv    = np.loadtxt('new_diff_modal_basis.out')


mako_cmap = sns.color_palette("plasma", as_cmap=True)


xnodal_nodes = nodal[0]
print(xnodal_nodes)

nummodes = len(nodal)-1
P        = nummodes - 1
nq       = len(nodal[0])

num_lines      = nummodes;
colors         = [mako_cmap((i) / num_lines) for i in range(num_lines)]


basis_sum = np.zeros((nq,1))
for i in range(0,nummodes):
    for j in range(0,nq):
        basis_sum[j] = basis_sum[j]+1.0*nodal[i+1][j];

for i in range(0,nummodes):
    plt.plot(xnodal_nodes,nodal[i+1],'-',color=colors[i])
    plt.plot(xnodal_nodes,nodal_deriv[i],'-',color=colors[i])
    print(nodal_deriv[i])

plt.plot(xnodal_nodes,basis_sum,'--',color='k')

for i in range(0,len(modal_l)):
    plt.plot(-0.25,modal_l[i],'ok')
    plt.plot(0.25,modal_r[i],'ok')


plt.show()