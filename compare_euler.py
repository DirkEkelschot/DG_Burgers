
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse


# args = sys.argv;

# P = int(args[1])
# Nel = int(args[2])
# npo = 2*P + 1



order = [1,2,3,4]

for p in order:

    npo = 2*p + 1

    lf  = np.loadtxt('dgdataEulerLaxFriedrichs_p'+str(p)+'.out')
    roe = np.loadtxt('dgdataEulerRoe_p'+str(p)+'.out')
    x_lf = lf[:,0]
    rho_lf = lf[:,1]
    rhou_lf = lf[:,2]
    rhoE_lf = lf[:,3]

    x_roe = roe[:,0]
    rho_roe = roe[:,1]
    rhou_roe = roe[:,2]
    rhoE_roe = roe[:,3]

    plt.figure(p)
    fig, axs = plt.subplots(3, 1)
    # plt.plot(x_lf,rho_lf);
    # plt.plot(x_roe,rho_roe);

    axs[0].plot(x_lf, rho_lf,label="Lax-Friedrichs p="+str(p))
    axs[0].plot(x_roe, rho_roe,label="Roe p="+str(p))
    # axs[0].plot(x_p, rho_p_i,'--','tab:red')
    axs[0].set_title(r'$\rho$')
    axs[0].grid()
    axs[0].legend()
    axs[1].plot(x_lf, rhou_lf,label="Lax-Friedrichs p="+str(p))
    axs[1].plot(x_roe, rhou_roe,label="Roe p="+str(p))
    # axs[1].plot(x_p, rhou_p_i,'--','tab:red')
    axs[1].set_title(r'$\rho u$')
    axs[1].grid()
    axs[1].legend()
    axs[2].plot(x_lf, rhoE_lf,label="Lax-Friedrichs p="+str(p))
    axs[2].plot(x_roe, rhoE_roe,label="Roe p="+str(p))

    # axs[2].plot(x_p, rhoE_p_i,'--','tab:red')
    axs[2].set_title(r'$\rho E$')
    axs[2].grid()
    axs[2].legend()

    plt.savefig("compare_p"+str(p)+".png", dpi=150)
    # axs[3].plot(x_p, pres_p, 'tab:red')
    # axs[3].set_title(r'$p$')
    # axs[3].grid()

plt.show()


# x_p = []
# rho_p = []
# rhou_p = []
# rhoE_p = []

# x_p_i = []
# rho_p_i = []
# rhou_p_i = []
# rhoE_p_i = []
# #pres_p = []
# plt.figure(1)
# for i in range(0,Nel):
#     xplot=np.zeros((npo,1))
#     rhoplot=np.zeros((npo,1))
#     rhouplot=np.zeros((npo,1))
#     rhoEplot=np.zeros((npo,1))

#     rhoplot_o=np.zeros((npo,1))
#     rhouplot_o=np.zeros((npo,1))
#     rhoEplot_o=np.zeros((npo,1))
#     #pressureplot_o = np.zeros((npo,1))
#     for j in range(0,npo):
#         xplot[j]=x[i*npo+j]
#         rhoplot[j]=rho[i*npo+j]
#         rhouplot[j]=rhou[i*npo+j]
#         rhoEplot[j]=rhoE[i*npo+j]
#         rhoplot_o[j]=rho_o[i*npo+j]
#         rhouplot_o[j]=rhou_o[i*npo+j]
#         rhoEplot_o[j]=rhoE_o[i*npo+j]
#         #pressureplot_o[j]=pres_o[i*npo+j]
#         x_p.append(x[i*npo+j])
#         rho_p.append(rho_o[i*npo+j])
#         rhou_p.append(rhou_o[i*npo+j])
#         rhoE_p.append(rhoE_o[i*npo+j])
#         x_p_i.append(x[i*npo+j])
#         rho_p_i.append(rho[i*npo+j])
#         rhou_p_i.append(rhou[i*npo+j])
#         rhoE_p_i.append(rhoE[i*npo+j])
#         #pres_p.append(pres_o[i*npo+j])
#     plt.plot(xplot,rhoplot,'-ob')
#     plt.plot(xplot,rhouplot,'-or')    
#     plt.plot(xplot,rhoEplot,'--ok')
#     plt.plot(xplot,rhoplot_o,'--ob')
#     plt.plot(xplot,rhouplot_o,'--or')
#     plt.plot(xplot,rhoEplot_o,'--ok')
#     # plt.plot(xplot,fplot,'-og')



# fig, axs = plt.subplots(3, 1)
# axs[0].plot(x_p, rho_p)
# # axs[0].plot(x_p, rho_p_i,'--','tab:red')
# axs[0].set_title(r'$\rho$')
# axs[0].grid()
# axs[1].plot(x_p, rhou_p)
# # axs[1].plot(x_p, rhou_p_i,'--','tab:red')
# axs[1].set_title(r'$\rho u$')
# axs[1].grid()
# axs[2].plot(x_p, rhoE_p)
# # axs[2].plot(x_p, rhoE_p_i,'--','tab:red')
# axs[2].set_title(r'$\rho E$')
# axs[2].grid()
# # axs[3].plot(x_p, pres_p, 'tab:red')
# # axs[3].set_title(r'$p$')
# # axs[3].grid()





plt.show()




