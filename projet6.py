import numpy as np
import matplotlib.pyplot as plt
ni,nh,no=13,6,1
wih=np.zeros([ni,nh])
who=np.zeros([nh,no])
ivec=np.zeros(ni)
sh=np.zeros(nh)
so=np.zeros(no)
err=np.zeros(no)
deltao=np.zeros(no)
deltah=np.zeros(nh)
eta=0.1
b=0.

# fonction d'activation sigmoidale
def actv(a):
    return 1./(1.+np.exp(-a))     

	
# derive de fonction d'activation sigmoidale
def dactv(s):
    return s*(1.-s)        

	
#feed forward
def ffnn(ivec):
    for ih in range(0,nh):
        s=0.
        for ii in range(0,ni):
            s+=wih[ii,ih]*ivec[ii]
        
        sh[ih]=actv(s)
        #print(sh[ih])
    for io in range(0,no):
        s=b
        for ih in range(0,nh):
            s+=who[ih,io]*sh[ih]
        
        
        so[io]=actv(s)
    return

	
#retropropagation
def backprop(err):
    for io in range(0,no):
        deltao[io]=err[io]*dactv(so[io])
        for ih in range(0,nh):
            who[ih,io]+=eta*deltao[io]*sh[ih]
    for ih in range(0,nh):
        som=0.
        for io in range(0,no):
            som+=deltao[io]*who[ih,io]
        deltah[ih]=dactv(sh[ih])*som
        for ii in range(0,ni):
            wih[ii,ih]+=eta*deltah[ih]*ivec[ii]
    return
        
#fonction de melange
def randomize(n):
    dumvec=np.zeros(n)
    for k in range(0,n):
        dumvec[k]=np.random.uniform()
    return np.argsort(dumvec)

# phase d'entrainement
nset=290
niter=1000
sol=np.zeros(nset)
oset=np.zeros([nset,no])
tset=np.zeros([nset,ni])
rmserr=np.zeros(niter)
datalist=[]
#lecture de fichier
fichier=open('data.txt' ,mode='r')
ligne=fichier.read().split('\n')
for i in range(0,nset):
    data=ligne[i].split()
    datalist.append(data)
    print(i)
fichier.close()
donnee=np.asarray(datalist)
tset=donnee[:,:13]
oset=donnee[:,13:14]
tset=tset.astype(np.float)
oset=oset.astype(np.int)
#normalisation
for k in range(0,nset):
    tset[k]=tset[k]/np.max(tset[k])


for ii in range(0,ni):
    for ih in range(0,nh):
        wih[ii,ih]=np.random.uniform(-0.5,0.5)


for ih in range(0,nh):
    for io in range(0,no):
        who[ih,io]=np.random.uniform(-0.5,0.5)

Ec=np.zeros(niter)
for iteration in range(0,niter):
    somme=0.
    rvec=randomize(nset)
    for itrain in range(0,nset):
        itt=rvec[itrain]
        ivec=tset[itt,:]
        ffnn(ivec)
        for io in range(0,no):
            sol[itt]=so[io]
            err[io]=oset[itt,io]-so[io]
            
            somme+=err[io]**2
        
        backprop(err)
    print(iteration)
    rmserr[iteration]=np.sqrt(somme/nset/no)
    tabs=(sol>0.5)*1
    Ec[iteration]=np.sum(np.abs(tabs-oset.reshape((nset,))))/nset
tabx=np.arange(0,niter)
plt.plot(tabx,rmserr,'b')
plt.plot(tabx,Ec,'r')
plt.xlabel('iteration')
plt.ylabel('erreurs')
plt.show()    


#phase test
N=1000
datalist=[]
prediction=np.zeros(N)
fichier=open('Game.txt' ,mode='r')
ligne=fichier.read().split('\n')
for i in range(0,N):
    data=ligne[i].split()
    datalist.append(data)
    
fichier.close()
donnee=np.asarray(datalist)
Tset=donnee[:,:13]
Tset=Tset.astype(np.float)

for k in range(0,N):
    Tset[k]=Tset[k]/np.max(Tset[k])
    ffnn(Tset[k])
    prediction[k]=so
    print(so)
tabp=(prediction>0.5)*1
#ecriture des resultats
fichier=open('guess.txt',mode='w')
for n in range(0,N):
    x=tabp[n].astype(np.str)
    fichier.write(x)
    fichier.write('\n')
fichier.close()


