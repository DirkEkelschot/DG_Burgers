
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

import xml.etree.ElementTree as ET

# with open('inputs.xml', 'r') as f:
#     inputs = f.read()

# print(inputs.find('PolynomialOrder')
      
# Parse the XML file
tree = ET.parse('inputs.xml')

# Get the root element
root = tree.getroot()

# Access the PARAMETERS element
parameters = root.find('PARAMETERS')

# Extract and print each parameter
inputs_map = {}
for param in parameters.findall('P'):
    key, value = param.text.split('=')
    key = key.strip()
    value = value.strip()
    print(f"{key}: {value}")
    inputs_map[key] = value

args = sys.argv;

print(inputs_map['PolynomialOrder'])
P   = int(float(inputs_map['PolynomialOrder']))
Nel = int(float(inputs_map['nElements']))
npo = int(float(inputs_map['nQuadrature']))

print(inputs_map)
compare = 1
conv = np.loadtxt('dgdataEuler.in')
conv_o = np.loadtxt('solution_'+inputs_map['BasisType']+'.out')
x = conv[:,0]
rho = conv[:,1]
rhou = conv[:,2]
rhoE = conv[:,3]

rho_o = conv_o[:,1]
rhou_o = conv_o[:,2]
rhoE_o = conv_o[:,3]
#pres_o = conv_o[:,4]
x_p = []
rho_p = []
rhou_p = []
rhoE_p = []

x_p_i = []
rho_p_i = []
rhou_p_i = []
rhoE_p_i = []
#pres_p = []
# plt.figure(1)
for i in range(0,Nel):
    xplot=np.zeros((npo,1))
    rhoplot=np.zeros((npo,1))
    rhouplot=np.zeros((npo,1))
    rhoEplot=np.zeros((npo,1))

    rhoplot_o=np.zeros((npo,1))
    rhouplot_o=np.zeros((npo,1))
    rhoEplot_o=np.zeros((npo,1))
    #pressureplot_o = np.zeros((npo,1))
    for j in range(0,npo):
        xplot[j]=x[i*npo+j]
        rhoplot[j]=rho[i*npo+j]
        rhouplot[j]=rhou[i*npo+j]
        rhoEplot[j]=rhoE[i*npo+j]
        rhoplot_o[j]=rho_o[i*npo+j]
        rhouplot_o[j]=rhou_o[i*npo+j]
        rhoEplot_o[j]=rhoE_o[i*npo+j]
        #pressureplot_o[j]=pres_o[i*npo+j]
        x_p.append(x[i*npo+j])
        rho_p.append(rho_o[i*npo+j])
        rhou_p.append(rhou_o[i*npo+j])
        rhoE_p.append(rhoE_o[i*npo+j])
        x_p_i.append(x[i*npo+j])
        rho_p_i.append(rho[i*npo+j])
        rhou_p_i.append(rhou[i*npo+j])
        rhoE_p_i.append(rhoE[i*npo+j])
        #pres_p.append(pres_o[i*npo+j])
    # plt.plot(xplot,rhoplot,'-ob')
    # plt.plot(xplot,rhouplot,'-or')    
    # plt.plot(xplot,rhoEplot,'--ok')
    # plt.plot(xplot,rhoplot_o,'--ob')
    # plt.plot(xplot,rhouplot_o,'--or')
    # plt.plot(xplot,rhoEplot_o,'--ok')
    # plt.plot(xplot,fplot,'-og')





fig, axs = plt.subplots(3, 1)
axs[0].plot(x_p, rho_p)
axs[0].plot(x_p, rho_p_i,'--')
axs[0].plot(x_p, rho_p,'.')
axs[0].set_title(r'$\rho$')
axs[0].grid()
axs[1].plot(x_p, rhou_p)
axs[1].plot(x_p, rhou_p_i,'--')
axs[1].plot(x_p, rhou_p,'.')
axs[1].set_title(r'$\rho u$')
axs[1].grid()
axs[2].plot(x_p, rhoE_p)
axs[2].plot(x_p, rhoE_p_i,'--')
axs[2].plot(x_p, rhoE_p,'.')
axs[2].set_title(r'$\rho E$')
axs[2].grid()
# axs[3].plot(x_p, pres_p, 'tab:red')
# axs[3].set_title(r'$p$')
# axs[3].grid()



# plt.figure(2)

# plt.plot(x_p, rhou_p_i,'--')
# plt.plot(x_p, rhou_p,'.')
# plt.plot(x_p, rhou_p)
plt.show()




