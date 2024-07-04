
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse



#print(len(conv[:,0]),len(conv[0,:]))
#l1,=plt.loglog(conv[:,11-3],conv[:,1],'-r',label="res vs #steps")
#l1,=plt.loglog(conv[:,0],conv[:,1],'-b',label="res vs time")
#
#plt.legend(handles=[l1])
#
#plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
#plt.grid()
args = sys.argv;

P = int(args[1])
Nel = int(args[2])
npo = P + 1



nodal = np.loadtxt('nodal_basis.out')
offset = 10*(P+1)
sum = np.zeros((offset,1))
plt.figure(1)
for i in range(0,(P+1)):
    # sum = sum + nodal[i*(P+1):(i+1)*(P+1),1]
    
    plt.plot(nodal[i*offset:(i+1)*offset,0],nodal[i*offset:(i+1)*offset,1])

    for j in range(0,offset):
        sum[j] = sum[j] + nodal[i*offset+j,1]

# print(sum)
plt.plot(nodal[0:offset,0],sum,'r')


modal = np.loadtxt('modal_basis.out')
offset = 10*(P+1)
sum = np.zeros((offset,1))
plt.figure(10)
for i in range(0,(P+1)):
    # sum = sum + nodal[i*(P+1):(i+1)*(P+1),1]
    
    plt.plot(modal[i*offset:(i+1)*offset,0],modal[i*offset:(i+1)*offset,1])

    for j in range(0,offset):
        sum[j] = sum[j] + modal[i*offset+j,1]


plt.figure(1010)
for i in range(0,(P+1)):
    # sum = sum + nodal[i*(P+1):(i+1)*(P+1),1]
    
    plt.plot(modal[i*offset:(i+1)*offset,0],modal[i*offset:(i+1)*offset,2]*1.0/20.0)

# print(sum)
# plt.plot(modal[0:offset,0],sum,'r')


M=[[0.25, -1.04083e-17, 4.44523e-18, -1.86483e-17, -8.99888e-18, -3.96818e-17, 2.7039e-17, -3.6929e-17, 3.56042e-17, -1.14992e-16, 1.67814e-17, -1.19828e-16, 3.47368e-17, -7.1991e-17, 2.27682e-17, -9.0314e-17, 5.62701e-17, -1.15034e-16, 1.18829e-16, -1.23599e-17, 2.1684e-18],
[-1.04083e-17, 0.0833333, -1.72388e-17, -3.25261e-18, -3.23092e-17, 1.0842e-17, -3.81216e-17, 2.89905e-17, -7.75865e-17, 2.61716e-17, -1.17118e-16, 2.64308e-17, -9.48254e-17, 2.9165e-17, -8.13152e-17, 4.01155e-17, -1.03216e-16, 8.81456e-17, -6.27753e-17, 5.74627e-17, 2.81893e-18],
[4.44523e-18, -1.72388e-17, 0.05, -2.84061e-17, 1.02999e-17, -3.38271e-17, 1.69559e-17, -6.93466e-17, 2.30274e-17, -8.89707e-17, 2.84908e-17, -9.8144e-17, 2.33527e-17, -9.45424e-17, 4.14165e-17, -9.37835e-17, 6.91721e-17, -6.63532e-17, 5.05238e-17, -3.75134e-17, -5.96311e-18],
[-1.86483e-17, -3.25261e-18, -2.84061e-17, 0.0357143, -3.11166e-17, 1.76725e-17, -6.32751e-17, 1.52212e-17, -8.12728e-17, 2.50874e-17, -7.76526e-17, 2.56718e-17, -9.75359e-17, 3.50197e-17, -1.03216e-16, 6.60279e-17, -6.46184e-17, 4.04407e-17, -4.4886e-17, -3.36103e-18, 1.12757e-17],
[-8.99888e-18, -3.23092e-17, 1.02999e-17, -3.11166e-17, 0.0277778, -5.78964e-17, 1.71727e-17, -7.47676e-17, 1.8799e-17, -7.18403e-17, 2.2094e-17, -8.12305e-17, 3.52789e-17, -1.043e-16, 5.95227e-17, -7.55689e-17, 4.05492e-17, -4.53197e-17, -5.20417e-18, -1.0842e-19, -8.89046e-18],
[-3.96818e-17, 1.0842e-17, -3.38271e-17, 1.76725e-17, -5.78964e-17, 0.0227273, -7.07561e-17, 2.0859e-17, -6.70698e-17, 1.66306e-17, -7.65684e-17, 3.26107e-17, -9.0597e-17, 5.87638e-17, -7.84962e-17, 3.64292e-17, -5.58364e-17, -1.51788e-18, -4.01155e-18, -1.00831e-17, 1.69136e-17],
[2.7039e-17, -3.81216e-17, 1.69559e-17, -6.32751e-17, 1.71727e-17, -7.07561e-17, 0.0192308, -6.65938e-17, 1.87329e-17, -7.24485e-17, 2.67984e-17, -8.63923e-17, 5.50537e-17, -6.72866e-17, 3.69052e-17, -5.91551e-17, -3.10185e-18, -1.54618e-17, -6.13762e-18, 1.31612e-17, -1.05828e-17],
[-3.6929e-17, 2.89905e-17, -6.93466e-17, 1.52212e-17, -7.47676e-17, 2.0859e-17, -6.65938e-17, 0.0166667, -7.03885e-17, 2.92497e-17, -8.18387e-17, 4.94582e-17, -6.39917e-17, 3.49536e-17, -5.01562e-17, -8.25026e-19, -2.01238e-17, -7.98076e-18, 9.09697e-19, -6.0292e-18, 2.14011e-17],
[3.56042e-17, -7.75865e-17, 2.30274e-17, -8.12728e-17, 1.8799e-17, -6.70698e-17, 1.87329e-17, -7.03885e-17, 0.0147059, -8.02547e-17, 5.05424e-17, -6.11304e-17, 3.05508e-17, -4.74457e-17, -1.58397e-18, -1.30765e-17, -6.13762e-18, -3.53553e-18, -7.87234e-18, 1.05591e-17, -1.19923e-17],
[-1.14992e-16, 2.61716e-17, -8.89707e-17, 2.50874e-17, -7.18403e-17, 1.66306e-17, -7.24485e-17, 2.92497e-17, -8.02547e-17, 0.0131579, -5.88536e-17, 3.05931e-17, -4.46929e-17, -4.83657e-18, -1.19923e-17, -5.92078e-18, 2.10232e-18, -6.13762e-18, 6.00545e-18, -1.32934e-17, 3.9182e-17],
[1.67814e-17, -1.17118e-16, 2.84908e-17, -7.76526e-17, 2.2094e-17, -7.65684e-17, 2.67984e-17, -8.18387e-17, 5.05424e-17, -5.88536e-17, 0.0119048, -4.26566e-17, -3.45086e-18, -9.88999e-18, -9.67315e-18, 3.12044e-18, -5.98686e-18, 1.11435e-17, -1.15163e-17, 3.4237e-17, -1.02152e-17],
[-1.19828e-16, 2.64308e-17, -9.8144e-17, 2.56718e-17, -8.12305e-17, 3.26107e-17, -8.63923e-17, 4.94582e-17, -6.11304e-17, 3.05931e-17, -4.26566e-17, 0.0108696, -8.76345e-18, -7.93843e-18, 4.96358e-18, -8.80579e-18, 1.21193e-17, -1.15163e-17, 3.86823e-17, -8.58895e-18, 4.75727e-17],
[3.47368e-17, -9.48254e-17, 2.33527e-17, -9.75359e-17, 3.52789e-17, -9.0597e-17, 5.50537e-17, -6.39917e-17, 3.05508e-17, -4.46929e-17, -3.45086e-18, -8.76345e-18, 0.01, 5.78861e-18, -7.87234e-18, 1.29443e-17, -1.47028e-17, 3.89652e-17, -8.1976e-18, 5.18672e-17, -1.7847e-17],
[-7.1991e-17, 2.9165e-17, -9.45424e-17, 3.50197e-17, -1.043e-16, 5.87638e-17, -6.72866e-17, 3.49536e-17, -4.74457e-17, -4.83657e-18, -9.88999e-18, -7.93843e-18, 5.78861e-18, 0.00925926, 1.36609e-17, -1.33357e-17, 3.98986e-17, -1.10589e-17, 5.20417e-17, -1.72388e-17, 3.67545e-17],
[2.27682e-17, -8.13152e-17, 4.14165e-17, -1.03216e-16, 5.95227e-17, -7.84962e-17, 3.69052e-17, -5.01562e-17, -1.58397e-18, -1.19923e-17, -9.67315e-18, 4.96358e-18, -7.87234e-18, 1.36609e-17, 0.00862069, 4.08744e-17, -1.02999e-17, 5.25838e-17, -1.98409e-17, 3.68629e-17, -1.95156e-17],
[-9.0314e-17, 4.01155e-17, -9.37835e-17, 6.60279e-17, -7.55689e-17, 3.64292e-17, -5.91551e-17, -8.25026e-19, -1.30765e-17, -5.92078e-18, 3.12044e-18, -8.80579e-18, 1.29443e-17, -1.33357e-17, 4.08744e-17, 0.00806452, 5.33427e-17, -1.97325e-17, 3.77302e-17, -2.20093e-17, 3.4586e-17],
[5.62701e-17, -1.03216e-16, 6.91721e-17, -6.46184e-17, 4.05492e-17, -5.58364e-17, -3.10185e-18, -2.01238e-17, -6.13762e-18, 2.10232e-18, -5.98686e-18, 1.21193e-17, -1.47028e-17, 3.98986e-17, -1.02999e-17, 5.33427e-17, 0.00757576, 3.8706e-17, -2.1684e-17, 3.50197e-17, -5.74627e-18],
[-1.15034e-16, 8.81456e-17, -6.63532e-17, 4.04407e-17, -4.53197e-17, -1.51788e-18, -1.54618e-17, -7.98076e-18, -3.53553e-18, -6.13762e-18, 1.11435e-17, -1.15163e-17, 3.89652e-17, -1.10589e-17, 5.25838e-17, -1.97325e-17, 3.8706e-17, 0.00714286, 3.59955e-17, -5.74627e-18, 2.61293e-17],
[1.18829e-16, -6.27753e-17, 5.05238e-17, -4.4886e-17, -5.20417e-18, -4.01155e-18, -6.13762e-18, 9.09697e-19, -7.87234e-18, 6.00545e-18, -1.15163e-17, 3.86823e-17, -8.1976e-18, 5.20417e-17, -1.98409e-17, 3.77302e-17, -2.1684e-17, 3.59955e-17, 0.00675676, 2.72135e-17, -5.42101e-18],
[-1.23599e-17, 5.74627e-17, -3.75134e-17, -3.36103e-18, -1.0842e-19, -1.00831e-17, 1.31612e-17, -6.0292e-18, 1.05591e-17, -1.32934e-17, 3.4237e-17, -8.58895e-18, 5.18672e-17, -1.72388e-17, 3.68629e-17, -2.20093e-17, 3.50197e-17, -5.74627e-18, 2.72135e-17, 0.00641026, 5.09575e-18],
[2.1684e-18, 2.81893e-18, -5.96311e-18, 1.12757e-17, -8.89046e-18, 1.69136e-17, -1.05828e-17, 2.14011e-17, -1.19923e-17, 3.9182e-17, -1.02152e-17, 4.75727e-17, -1.7847e-17, 3.67545e-17, -1.95156e-17, 3.4586e-17, -5.74627e-18, 2.61293e-17, -5.42101e-18, 5.09575e-18, 0.0125]]

plt.figure(1000)
plt.contourf(M)



radau = np.loadtxt('radaum_basis.out')
offset = 10*(P+1)
sum = np.zeros((offset,1))
plt.figure(20)
for i in range(0,(P+1)):
    # sum = sum + nodal[i*(P+1):(i+1)*(P+1),1]
    
    plt.plot(radau[i*offset:(i+1)*offset,0],radau[i*offset:(i+1)*offset,1],label='R'+str(i))

    for j in range(0,offset):
        sum[j] = sum[j] + radau[i*offset+j,1]

plt.figure(2020)
for i in range(0,(P+1)):
    # sum = sum + nodal[i*(P+1):(i+1)*(P+1),1]
    
    plt.plot(radau[i*offset:(i+1)*offset,0],radau[i*offset:(i+1)*offset,2],label='R'+str(i))

    for j in range(0,offset):
        sum[j] = sum[j] + radau[i*offset+j,1]

# print(sum)
# plt.plot(radau[0:offset,0],sum,'r')
plt.legend()



radau = np.loadtxt('radaup_basis.out')
offset = 10*(P+1)
sum = np.zeros((offset,1))
zeros = np.zeros((offset,1))
plt.figure(21)
for i in range(0,(P+1)):
    # sum = sum + nodal[i*(P+1):(i+1)*(P+1),1]
    
    plt.plot(radau[i*offset:(i+1)*offset,0],radau[i*offset:(i+1)*offset,1],label='R'+str(i))
    plt.plot(radau[i*offset:(i+1)*offset,0],zeros,'ok')
    for j in range(0,offset):
        sum[j] = sum[j] + radau[i*offset+j,1]

# print(sum)
# plt.plot(radau[0:offset,0],sum,'r')
plt.legend()



plt.show()





    
# conv = np.loadtxt('dgdata.in')

# x = conv[:,0]
# u = conv[:,1]

# plt.figure(1)
# for i in range(0,Nel):
#     xplot=np.zeros((npo,1))
#     uplot=np.zeros((npo,1))
#     for j in range(0,npo):
#         xplot[j]=x[i*npo+j]
#         uplot[j]=u[i*npo+j]
        
#     plt.plot(xplot,uplot,'-ob')




# # times = ['0.000000','0.010000','0.050000']

# # for i in (times):

# #     print('dgdata'+str(i)+'.out')
# #     conv = np.loadtxt('dgdata'+i+'.out')

# #     xout = conv[:,0]
# #     uout = conv[:,1]
# #     for i in range(0,Nel):
# #         xplot=np.zeros((npo,1))
# #         uplot=np.zeros((npo,1))
# #         for j in range(0,npo):
# #             xplot[j]=xout[i*npo+j]
# #             uplot[j]=uout[i*npo+j]
            
# #         # plt.plot(xplot,uplot,'--o',color="tab:orange")
# #         plt.plot(xplot,uplot,'--o',color="tab:orange")


# conv = np.loadtxt('dgdata.out')

# xout = conv[:,0]
# uout = conv[:,1]
# for i in range(0,Nel):
#     xplot=np.zeros((npo,1))
#     uplot=np.zeros((npo,1))
#     for j in range(0,npo):
#         xplot[j]=xout[i*npo+j]
#         uplot[j]=uout[i*npo+j]
        
#     # plt.plot(xplot,uplot,'--o',color="tab:orange")
#     plt.plot(xplot,uplot,'-o',color="tab:orange")


# # conv = np.loadtxt('dgdatamodal.out')

# # xout = conv[:,0]
# # uout = conv[:,1]
# # for i in range(0,Nel):
# #     xplot=np.zeros((npo,1))
# #     uplot=np.zeros((npo,1))
# #     for j in range(0,npo):
# #         xplot[j]=xout[i*npo+j]
# #         uplot[j]=uout[i*npo+j]
        
# #     # plt.plot(xplot,uplot,'--o',color="tab:orange")
# #     plt.plot(xplot,uplot,'--o',color="tab:red")


# # conv = np.loadtxt('dgdata2_10.out')

# # xout = conv[:,0]
# # uout = conv[:,1]
# # for i in range(0,Nel):
# #     xplot=np.zeros((npo,1))
# #     uplot=np.zeros((npo,1))
# #     for j in range(0,npo):
# #         xplot[j]=xout[i*npo+j]
# #         uplot[j]=uout[i*npo+j]
        
# #     # plt.plot(xplot,uplot,'--o',color="tab:orange")
# #     plt.plot(xplot,uplot,'--o',color="tab:red")


# # conv = np.loadtxt('dgdata3_10.out')

# # xout = conv[:,0]
# # uout = conv[:,1]
# # for i in range(0,Nel):
# #     xplot=np.zeros((npo,1))
# #     uplot=np.zeros((npo,1))
# #     for j in range(0,npo):
# #         xplot[j]=xout[i*npo+j]
# #         uplot[j]=uout[i*npo+j]
        
# #     # plt.plot(xplot,uplot,'--o',color="tab:orange")
# #     plt.plot(xplot,uplot,'--o',color="tab:green")


# # conv = np.loadtxt('dgdata4_10.out')

# # xout = conv[:,0]
# # uout = conv[:,1]
# # for i in range(0,Nel):
# #     xplot=np.zeros((npo,1))
# #     uplot=np.zeros((npo,1))
# #     for j in range(0,npo):
# #         xplot[j]=xout[i*npo+j]
# #         uplot[j]=uout[i*npo+j]
        
# #     # plt.plot(xplot,uplot,'--o',color="tab:orange")
# #     plt.plot(xplot,uplot,'--o',color="tab:purple")

# plt.show()




