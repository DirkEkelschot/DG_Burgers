
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse


args = sys.argv;

P = int(args[1])
Nel = int(args[2])
npo = P + 1


    
conv = np.loadtxt('dgdata.in')

x = conv[:,0]
u = conv[:,1]

plt.figure(1)
for i in range(0,Nel):
    xplot=np.zeros((npo,1))
    uplot=np.zeros((npo,1))
    for j in range(0,npo):
        xplot[j]=x[i*npo+j]
        uplot[j]=u[i*npo+j]
        
    plt.plot(xplot,uplot,'-ob')




# times = ['0.000000','0.010000','0.050000']

# for i in (times):

#     print('dgdata'+str(i)+'.out')
#     conv = np.loadtxt('dgdata'+i+'.out')

#     xout = conv[:,0]
#     uout = conv[:,1]
#     for i in range(0,Nel):
#         xplot=np.zeros((npo,1))
#         uplot=np.zeros((npo,1))
#         for j in range(0,npo):
#             xplot[j]=xout[i*npo+j]
#             uplot[j]=uout[i*npo+j]
            
#         # plt.plot(xplot,uplot,'--o',color="tab:orange")
#         plt.plot(xplot,uplot,'--o',color="tab:orange")


conv = np.loadtxt('dgdata.out')

xout = conv[:,0]
uout = conv[:,1]
for i in range(0,Nel):
    xplot=np.zeros((npo,1))
    uplot=np.zeros((npo,1))
    for j in range(0,npo):
        xplot[j]=xout[i*npo+j]
        uplot[j]=uout[i*npo+j]
        
    # plt.plot(xplot,uplot,'--o',color="tab:orange")
    plt.plot(xplot,uplot,'-o',color="tab:orange")


# conv = np.loadtxt('dgdatamodal.out')

# xout = conv[:,0]
# uout = conv[:,1]
# for i in range(0,Nel):
#     xplot=np.zeros((npo,1))
#     uplot=np.zeros((npo,1))
#     for j in range(0,npo):
#         xplot[j]=xout[i*npo+j]
#         uplot[j]=uout[i*npo+j]
        
#     # plt.plot(xplot,uplot,'--o',color="tab:orange")
#     plt.plot(xplot,uplot,'--o',color="tab:red")


# conv = np.loadtxt('dgdata2_10.out')

# xout = conv[:,0]
# uout = conv[:,1]
# for i in range(0,Nel):
#     xplot=np.zeros((npo,1))
#     uplot=np.zeros((npo,1))
#     for j in range(0,npo):
#         xplot[j]=xout[i*npo+j]
#         uplot[j]=uout[i*npo+j]
        
#     # plt.plot(xplot,uplot,'--o',color="tab:orange")
#     plt.plot(xplot,uplot,'--o',color="tab:red")


# conv = np.loadtxt('dgdata3_10.out')

# xout = conv[:,0]
# uout = conv[:,1]
# for i in range(0,Nel):
#     xplot=np.zeros((npo,1))
#     uplot=np.zeros((npo,1))
#     for j in range(0,npo):
#         xplot[j]=xout[i*npo+j]
#         uplot[j]=uout[i*npo+j]
        
#     # plt.plot(xplot,uplot,'--o',color="tab:orange")
#     plt.plot(xplot,uplot,'--o',color="tab:green")


# conv = np.loadtxt('dgdata4_10.out')

# xout = conv[:,0]
# uout = conv[:,1]
# for i in range(0,Nel):
#     xplot=np.zeros((npo,1))
#     uplot=np.zeros((npo,1))
#     for j in range(0,npo):
#         xplot[j]=xout[i*npo+j]
#         uplot[j]=uout[i*npo+j]
        
#     # plt.plot(xplot,uplot,'--o',color="tab:orange")
#     plt.plot(xplot,uplot,'--o',color="tab:purple")

plt.show()




