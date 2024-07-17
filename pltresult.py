
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
f = conv[:,2]
plt.figure(1)
for i in range(0,Nel):
    xplot=np.zeros((npo,1))
    uplot=np.zeros((npo,1))
    fplot=np.zeros((npo,1))
    for j in range(0,npo):
        xplot[j]=x[i*npo+j]
        uplot[j]=u[i*npo+j]
        fplot[j]=f[i*npo+j]
        
    plt.plot(xplot,uplot,'-ob')
    # plt.plot(xplot,fplot,'-og')




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
fout = conv[:,2]

# r2out = conv[:,4]
for i in range(0,Nel):
    xplot=np.zeros((npo,1))
    uplot=np.zeros((npo,1))
    fplot=np.zeros((npo,1))

    for j in range(0,npo):
        xplot[j]=xout[i*npo+j]
        uplot[j]=uout[i*npo+j]
        fplot[j]=fout[i*npo+j]

    # plt.plot(xplot,uplot,'--o',color="tab:orange")
    plt.plot(xplot,uplot,'-o',color="tab:orange")
    # plt.plot(xplot,fplot,'-o',color="tab:red")





plt.figure(20)

convrhs = np.loadtxt('dgRHSdata.out')

outrhs  = convrhs[:,0]
outrhs0 = convrhs[:,1]
outrhs1 = convrhs[:,2]
for i in range(0,Nel):
    xplot=np.zeros((npo,1))
    outrhsplot=np.zeros((npo,1))
    outrhs0plot=np.zeros((npo,1))
    outrhs1plot=np.zeros((npo,1))
    for j in range(0,npo):
        xplot[j]=x[i*npo+j]
        outrhsplot[j]=outrhs[i*npo+j]
        outrhs0plot[j]=outrhs0[i*npo+j]
        outrhs1plot[j]=outrhs1[i*npo+j]
    # plt.plot(xplot,uplot,'--o',color="tab:orange")
    plt.plot(xplot,outrhsplot,'-o',color="tab:green")
    plt.plot(xplot,outrhs0plot,'-o',color="tab:blue")
    plt.plot(xplot,outrhs1plot,'-o',color="tab:pink",linewidth=3)
    plt.plot(xplot,outrhs0plot+outrhs1plot,'--o',color="tab:purple")

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




