import numpy as np
from numpy import math
import scipy
import matplotlib
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.optimize import fsolve
from numpy import logspace
from pylab import scatter
from scipy.integrate import solve_ivp
import multiprocessing
import itertools
from scipy.signal import argrelextrema

Npoints = 10


Ks_Values = np.logspace(-2,2,Npoints)
Vd_Values = np.logspace(-4,0,Npoints)
Vd_Values = np.flipud(Vd_Values)

number_of_cores = multiprocessing.cpu_count()
tmax=240
n=6 
threshold_kaiC=600

# another set of initial conditon: it was chosen by runing the model and taking the final time point => less transcient behavior
xs_init_specific = np.array([4.21811090e-01, 3.18514069e-01, 1.03447154e-01, 1.80699948e-02,
       1.77761154e-03, 9.38839685e-05, 2.17937539e-06, 4.13557615e-01,
       2.62823939e-01, 7.00568065e-02, 9.44938276e-03, 6.36138005e-04,
       1.71753079e-05, 1.18224008e-01, 5.74320453e-02, 1.19432807e-02,
       1.18434228e-03, 4.49361010e-05, 1.83337815e-02, 6.12219211e-03,
       7.58483947e-04, 3.40062566e-05, 1.03304086e-03, 2.54137898e-04,
       1.63472376e-05, 1.32441351e-05, 2.00282786e-06, 1.66226735e-08,
       7.77991846e-02, 5.11617512e-02, 1.36804116e-02, 1.83262850e-03,
       1.24267335e-04, 3.67027164e-06, 4.65797378e-02, 2.63876338e-02,
       5.78152876e-03, 5.57969833e-04, 2.02333300e-05, 9.08985862e-03,
       3.39960695e-03, 4.64410243e-04, 2.29040217e-05, 7.33322497e-04,
       1.88505024e-04, 1.25769993e-05, 1.29315682e-05, 2.01525870e-06,
       2.03873193e-08, 5.28761957e-03, 2.93386807e-03, 6.16543305e-04,
       5.93038200e-05, 2.45457706e-06, 1.72526840e-03, 7.28010967e-04,
       1.11950433e-04, 6.24593316e-06, 2.04609446e-04, 5.48260861e-05,
       3.90043467e-06, 5.21446223e-06, 8.28501986e-07, 1.07232018e-08,
       1.34872294e-04, 6.36474085e-05, 1.05611452e-05, 7.15608661e-07,
       2.67667365e-05, 7.72721452e-06, 6.26116839e-07, 1.08396722e-06,
       1.87547278e-07, 3.01528189e-09, 1.48816428e-06, 5.40336334e-07,
       7.70961579e-08, 1.22875565e-07, 2.19348810e-08, 5.80571895e-10,
       7.44721044e-09, 3.61041496e-09, 4.49711102e-11, 1.29240879e-11,
       1.36012106e-01, 8.20178510e-02, 2.39187278e-02, 3.95816873e-03,
       3.78933000e-04, 1.97791487e-05, 4.61642123e-07, 2.43565662e-01,
       1.05619602e-01, 2.17531172e-02, 2.47012051e-03, 1.49657199e-04,
       3.85304466e-06, 1.74874922e-01, 6.51338012e-02, 1.01168340e-02,
       7.12388181e-04, 1.85446548e-05, 3.56966165e-02, 1.07577923e-02,
       1.19387939e-03, 4.72146913e-05, 2.14440922e-03, 5.03266842e-04,
       3.09079302e-05, 2.81570295e-05, 4.12991813e-06, 3.62527554e-08,
       2.86683189e-02, 1.49716438e-02, 3.45059011e-03, 4.26146274e-04,
       2.79082085e-05, 8.28791276e-07, 4.25446204e-02, 1.63080155e-02,
       2.44928759e-03, 1.75542994e-04, 5.24110141e-06, 1.57062942e-02,
       5.15417313e-03, 5.99325306e-04, 2.39268137e-05, 1.46295226e-03,
       3.58183154e-04, 2.26737653e-05, 2.69380845e-05, 4.07926868e-06,
       4.39153651e-08, 2.46561745e-03, 1.03552958e-03, 1.78078851e-04,
       1.50466886e-05, 5.99016807e-07, 2.42621078e-03, 8.17993204e-04,
       9.26586669e-05, 3.48094348e-06, 3.90573177e-04, 9.85009385e-05,
       6.36915253e-06, 1.06718236e-05, 1.65617558e-06, 2.27833751e-08,
       1.13890336e-04, 3.56524818e-05, 4.04409717e-06, 2.13504836e-07,
       4.77397594e-05, 1.22859256e-05, 8.29041329e-07, 2.17485379e-06,
       3.45345156e-07, 6.46201507e-09, 2.21867993e-06, 5.89837650e-07,
       4.67906353e-08, 2.30414150e-07, 4.08558994e-08, 1.05812440e-09,
       1.13578499e-08, 2.60086605e-09, 1.14279896e-10, 7.53901963e-12,
       1.71734123e-16])



def hit_ground(t, y,params): 
    threshold_kaiC=600
    return y[0] - threshold_kaiC
hit_ground.terminal = True


def estimate_period(timelapse,t):
    """ estimate the period of the time series generated with one "species" of the timelapse function, by searching peaks in the time series"""

    nbr_timepts = len(t)
    
    t_notrans=t[int(nbr_timepts/2):nbr_timepts]
    timelapse_notrans = timelapse[int(nbr_timepts/2):nbr_timepts]
    locmax = argrelextrema(timelapse_notrans, np.greater)[0]
    mean_val = np.mean(timelapse_notrans)
    heights = np.array([timelapse_notrans[locmax[k]]-mean_val  for k in range(len(locmax))])
    condition_height=np.array(heights>.1*mean_val)
    #locmin = argrelextrema(timelapse_notrans, np.less)
    
    if len(heights)>1:
        locmax = locmax[condition_height]
        if len(locmax>1):
            number_of_max_peaks=np.shape(locmax)[0]
            periods_locmax=t_notrans[locmax[1:number_of_max_peaks]]-t_notrans[locmax[0:(number_of_max_peaks-1)]]   
            #print(t_notrans[locmax[0][-1]]-t_notrans[locmin[0][-1]])        
            return np.mean(periods_locmax)
        else:
            return np.nan
    else:
        return np.nan

#to reconstruction a matrix containing periods of oscillations obtained from scanning over two parameters:
#for some scans, we saved extra information. when this is the case, the value of myindex should be adapted accordingly
def matrix_reconstruction(preoscillations,myindex=2):
    oscillations = np.zeros((Npoints,Npoints))
    for kk in range(len(preoscillations)):
        index1 = int(preoscillations[kk,1]) # line index is V_V
        index2 = int(preoscillations[kk,0]) # column index is Ks_V
        oscillations[index1,index2] = preoscillations[kk,myindex]
    return oscillations

######
######


def def_ijk_combinations(n):

    ijk_combinations = [np.array([-1,-1,-1])]
    for kk in range(n+1):
        for jj in range(n+1-kk):
            for ii in range(n+1-kk-jj):
                ijk_combinations_new = [np.array([ii,jj,kk])]
                #print(type(ijk_combinations_new))
                ijk_combinations = np.concatenate((ijk_combinations,ijk_combinations_new),axis=0)
    ijk_combinations = ijk_combinations[1:]
    return ijk_combinations



def compute_index(i,j,k):
    return i*(n+1)**2 + j*(n+1)+k

def compute_index_KaiB(i,j,k):
    return (n+1)**3 +  i*(n+1)**2 + j*(n+1)+k

def def_C_indices_double(n):
    C_indices = [compute_index(x[0],x[1],x[2]) for x in ijk_combinations]
    len_C_indices = len(C_indices)
    C_indices_KaiB = [compute_index_KaiB(x[0],x[1],x[2]) for x in ijk_combinations]
    C_indices_double = np.asarray(np.reshape(np.array([C_indices,C_indices_KaiB]),(len_C_indices*2,)))
    C_indices_double = C_indices_double.tolist()

    return C_indices, C_indices_double

def my_new_indices(kk):
    return C_indices_double.index(kk)

#because the ijk combinations are used to define other functions, we compute them in this python file:
ijk_combinations = def_ijk_combinations(n)
C_indices, C_indices_double = def_C_indices_double(n)
number_states_without_B = len(C_indices)


def def_only_T_phospho(n):
    only_T_phospho = np.array([-1])
    for kk in ijk_combinations: 
        i,j,k = kk
        if kk[1]==0 and kk[2]==0:
            only_T_phospho  = np.concatenate((only_T_phospho,np.array([my_new_indices(compute_index(i,j,k))])))
    only_T_phospho = only_T_phospho[2:]

    return np.concatenate((only_T_phospho,only_T_phospho+number_states_without_B))

only_T = def_only_T_phospho(n)

def def_only_S_phospho(n):
    only_S_phospho = np.array([-1])
    for kk in ijk_combinations: 
        i,j,k = kk
        if kk[0]==0 and kk[2]==0:

            only_S_phospho  = np.concatenate((only_S_phospho,np.array([my_new_indices(compute_index(i,j,k))])))
    only_S_phospho = only_S_phospho[2:]
    return np.concatenate((only_S_phospho,only_S_phospho+number_states_without_B))

only_S = def_only_S_phospho(n)

def def_only_D_phospho(n):
    only_D_phospho = np.array([-1])
    for kk in ijk_combinations: 
        i,j,k = kk
        if kk[0]==0 and kk[1]==0:

            only_D_phospho  = np.concatenate((only_D_phospho,np.array([my_new_indices(compute_index(i,j,k))])))
    only_D_phospho = only_D_phospho[2:]
    return  np.concatenate((only_D_phospho,only_D_phospho+number_states_without_B))
only_D = def_only_D_phospho(n)

def def_only_U(n):
    return[0,number_states_without_B]
only_U = def_only_U(n)

contain_S_phospho = np.array([-1])
weight_contain_S_phospho = np.array([-1])
#we loop over KaiCijk configurations:
for kk in ijk_combinations: 
    i,j,k = kk
    if kk[1]!=0:

        contain_S_phospho   = np.concatenate((contain_S_phospho ,np.array([my_new_indices(compute_index(i,j,k))])))
        weight_contain_S_phospho =np.concatenate((weight_contain_S_phospho,np.array([j])))
#we remove the artificial"-1"
contain_S_phospho = contain_S_phospho[1:]#start at one and not 2 because here we don't get 0,0,0 by default

weight_contain_S_phospho = weight_contain_S_phospho[1:]
#we copy info for KaiB*KaiC variables
contain_S_phospho = np.concatenate((contain_S_phospho,contain_S_phospho+number_states_without_B))
weight_contain_S_phospho  = np.concatenate((weight_contain_S_phospho ,weight_contain_S_phospho ))



def def_full_phospohate(n):
    ijk_combinations = def_ijk_combinations(n)
    phospho_level = [x[0]+x[1]+2*x[2] for x in ijk_combinations]
    full_phospohate = np.concatenate((phospho_level,phospho_level))
    return full_phospohate

full_phospohate = def_full_phospohate(n)



######################################################
######################################################
# definitions to encode the model in a compact manner
######################################################
######################################################

def kaiA(kaiA0, kaiBC): #KaiBKaiC is sum of all KaiBKaiC(i,j,k)return max(0, kaiA0 - 6*2*sum(C[i],i=N*N,2*N*N-1))
    return max(0, kaiA0-n*2*kaiBC) # here we set m=2

def DeltaG(i,j,k,paramsDelta):
    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta
    return i*deltaGpT+j*deltaGpS+k*deltaGpSpT+(n-i-j-k)*deltaGU

def KA(i,j,k,paramsDelta):
    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta
    return np.exp(-( DeltaG(i,j,k,paramsDelta)))

def FA(i,j,k,kaiBC,kaiA0,km,paramsDelta):
    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta
    return (kaiA(kaiA0,kaiBC)/(kaiA(kaiA0,kaiBC)+(km*KA(i,j,k,paramsDelta))+km))



def FB(i,j,k,kaiBC,kaiA0,km,paramsDelta):
    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta
    return 1/(1+(1/KA(i,j,k,paramsDelta))+kaiA(kaiA0,kaiBC)/(km*KA(i,j,k,paramsDelta)))


def kxyUT(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact):
    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta
    k0UT,k0TD,k0US,k0SD,k0DS,k0DT,k0TU,k0SU = paramsk0
    kactUT,kactTD,kactUS,kactSD,kactDS,kactDT,kactTU,kactSU = paramskact
    kxy= FA(i,j,k,kaiBC,kaiA0,km,paramsDelta)*kactUT +(1-FA(i,j,k,kaiBC,kaiA0,km,paramsDelta))*k0UT
    return kxy


def kxyUS(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact):
    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta
    k0UT,k0TD,k0US,k0SD,k0DS,k0DT,k0TU,k0SU = paramsk0
    kactUT,kactTD,kactUS,kactSD,kactDS,kactDT,kactTU,kactSU = paramskact
    kxy= FA(i,j,k,kaiBC,kaiA0,km,paramsDelta)*kactUS +(1-FA(i,j,k,kaiBC,kaiA0,km,paramsDelta))*k0US
    return kxy

def kxyTD(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact):
    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta
    k0UT,k0TD,k0US,k0SD,k0DS,k0DT,k0TU,k0SU = paramsk0
    kactUT,kactTD,kactUS,kactSD,kactDS,kactDT,kactTU,kactSU = paramskact
    kxy= FA(i,j,k,kaiBC,kaiA0,km,paramsDelta)*kactTD +(1-FA(i,j,k,kaiBC,kaiA0,km,paramsDelta))*k0TD
    return kxy


def kxyTU(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact):
    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta
    k0UT,k0TD,k0US,k0SD,k0DS,k0DT,k0TU,k0SU = paramsk0
    kactUT,kactTD,kactUS,kactSD,kactDS,kactDT,kactTU,kactSU = paramskact
    kxy= FA(i,j,k,kaiBC,kaiA0,km,paramsDelta)*kactTU +(1-FA(i,j,k,kaiBC,kaiA0,km,paramsDelta))*k0TU
    return kxy

def kxyDT(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact):
    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta
    k0UT,k0TD,k0US,k0SD,k0DS,k0DT,k0TU,k0SU = paramsk0
    kactUT,kactTD,kactUS,kactSD,kactDS,kactDT,kactTU,kactSU = paramskact
    kxy= FA(i,j,k,kaiBC,kaiA0,km,paramsDelta)*kactDT+(1-FA(i,j,k,kaiBC,kaiA0,km,paramsDelta))*k0DT
    return kxy

def kxyDS(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact):
    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta
    k0UT,k0TD,k0US,k0SD,k0DS,k0DT,k0TU,k0SU = paramsk0
    kactUT,kactTD,kactUS,kactSD,kactDS,kactDT,kactTU,kactSU = paramskact
    kxy= FA(i,j,k,kaiBC,kaiA0,km,paramsDelta)*kactDS +(1-FA(i,j,k,kaiBC,kaiA0,km,paramsDelta))*k0DS
    return kxy


def kxySD(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact):
    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta
    k0UT,k0TD,k0US,k0SD,k0DS,k0DT,k0TU,k0SU = paramsk0
    kactUT,kactTD,kactUS,kactSD,kactDS,kactDT,kactTU,kactSU = paramskact
    kxy= FA(i,j,k,kaiBC,kaiA0,km,paramsDelta)*kactSD +(1-FA(i,j,k,kaiBC,kaiA0,km,paramsDelta))*k0SD
    return kxy


def kxySU(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact):
    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta
    k0UT,k0TD,k0US,k0SD,k0DS,k0DT,k0TU,k0SU = paramsk0
    kactUT,kactTD,kactUS,kactSD,kactDS,kactDT,kactTU,kactSU = paramskact
    kxy= FA(i,j,k,kaiBC,kaiA0,km,paramsDelta)*kactSU +(1-FA(i,j,k,kaiBC,kaiA0,km,paramsDelta))*k0SU
    return kxy       

    #todo : to speed up the code: an update could be to define sets of indices to avoid using "if"s in the ode solver.

def Cip(C,n,i,j,k):
    if (i==n or (i+j+k)==n): # we can't add a phosphate group at T if the hexamer is already saturated. 
        return 0.0
    else:
        #print('hi',((n+1)**2)*(i+1)+(n+1)*j+k)
        return (i+1)*C[my_new_indices(((n+1)**2)*(i+1)+(n+1)*j+k)]
        
def Cim(C,n,i,j,k):
    if (i==0):
        return 0.0
    else:
        return C[my_new_indices(((n+1)**2)*(i-1)+(n+1)*j+k)]
def Cjp(C,n,i,j,k):
    if (j==n or (i+j+k)==n):
        return 0.0
    else:
        return (j+1)*C[my_new_indices(((n+1)**2)*i+(n+1)*(j+1)+k)]
def Cjm(C,n,i,j,k): #0 0 1 -- -2+1 ((n+1)**2)*i+(n+1)*(j+1)+k
    if (j==0):
        return 0.0
    else:
        return C[my_new_indices(((n+1)**2)*i+(n+1)*(j-1)+k)]
    
    
def Cikp(C,n,i,j,k): 
    if (i==n ) or (k==0):#todo or i+j+k==n+1
        return 0.0
    else:
        return (i+1)*C[my_new_indices(((n+1)**2)*(i+1)+(n+1)*j+k-1)]
def Cikm(C,n,i,j,k):
    if (i==0) or (k==n):#todo or (i+j+k)==n+1):
        return 0.0
    else:
        return (k+1)*C[my_new_indices(((n+1)**2)*(i-1)+(n+1)*j+k+1)]
def Cjkp(C,n,i,j,k):
    if j==n  or (k==0):#todo or (i+j+k)==n+1)
        return 0.0
    else:
        return (j+1)*C[my_new_indices(((n+1)**2)*i+(n+1)*(j+1)+k-1)]
def Cjkm(C,n,i,j,k):
    if (j==0) or (k==n ):#todo or (i+j+k)==n+1
        return 0.0
    else:
        return (k+1)*C[my_new_indices(((n+1)**2)*i+(n+1)*(j-1)+k+1)]
    
    
def kaiBCip(C,n,i,j,k):
    if (i==n) or i+j+k==n:
        return 0.0
    else:
        return (i+1)*C[my_new_indices(((n+1)**3)+((n+1)**2)*(i+1)+(n+1)*j+k)]
def kaiBCim(C,n,i,j,k):
    if (i==0):
        return 0.0
    else:
        return C[my_new_indices(((n+1)**3)+((n+1)**2)*(i-1)+(n+1)*j+k)]
def kaiBCjp(C,n,i,j,k):
    if (j==n) or i+j+k==n:
        return 0.0
    else:
        #print(((n+1)**3)+((n+1)**2)*i+(n+1)*(j+1)+k)
        return (j+1)*C[my_new_indices(((n+1)**3)+((n+1)**2)*i+(n+1)*(j+1)+k)]
def kaiBCjm(C,n,i,j,k):
    if (j==0):
        return 0.0
    else:
        #print(((n+1)**3)+((n+1)**2)*i+(n+1)*(j-1)+k)
        return C[my_new_indices(((n+1)**3)+((n+1)**2)*i+(n+1)*(j-1)+k)]
def kaiBCikp(C,n,i,j,k):
    if (i==n or (i+j+k)==n+1) or (k==0):
        return 0.0
    else:
        #print(((n+1)**3)+((n+1)**2)*(i+1)+(n+1)*j+k-1)
        return (i+1)*C[my_new_indices(((n+1)**3)+((n+1)**2)*(i+1)+(n+1)*j+k-1)]
def kaiBCikm(C,n,i,j,k):
    if (i==0) or (k==n or (i+j+k)==n+1):
        return 0.0
    else:
        return (k+1)*C[my_new_indices(((n+1)**3)+((n+1)**2)*(i-1)+(n+1)*j+k+1)]
def kaiBCjkp(C,n,i,j,k):
    if (j==n or (i+j+k)==n+1) or (k==0):
        return 0.0
    else:
        return (j+1)*C[my_new_indices(((n+1)**3)+((n+1)**2)*i+(n+1)*(j+1)+k-1)]
def kaiBCjkm(C,n,i,j,k):
    if (j==0) or (k==n ):#todo or (i+j+k)==n+1
        return 0.0
    else:
        return (k+1)*C[my_new_indices(((n+1)**3)+((n+1)**2)*i+(n+1)*(j-1)+k+1)]


######################################################
######################################################
# in vitro model
######################################################
######################################################

def dCdt(t,C,params):
    # C = [KaiCijk (7³ element) KaiBCijk (7³)] C m => m = i*(n+1)^2 + j*(n+1) + k 
    konB, koffB, paramsDelta, km, kaiC, kaiA0, paramsk0, paramskact = params 

    k0UT, k0TD, k0US, k0SD, k0DS, k0DT, k0TU, k0SU = paramsk0
    kactUT, kactTD, kactUS, kactSD, kactDS, kactDT, kactTU, kactSU = paramsk0

    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta

    dkaiC = np.empty(int(len(C)/2))
    dkaiBC=np.empty(int(len(C)/2))

    kaiBC = np.sum(C[int(len(C)/2):])# 

    for i,j,k in ijk_combinations:


                kUTeval = kxyUT(i-1, j, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kUSeval  = kxyUS(i, j-1, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTDeval = kxyTD(i+1, j, k-1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTUeval = kxyTU(i+1, j, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDTeval = kxyDT(i-1, j, k+1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDSeval = kxyDS(i, j-1, k+1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSDeval = kxySD(i, j+1, k-1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSUeval = kxySU(i, j+1, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                
                kUTe = kxyUT(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kUSe = kxyUS(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTDe = kxyTD(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTUe = kxyTU(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDTe = kxyDT(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDSe = kxyDS(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSDe = kxySD(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSUe = kxySU(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                
                dkaiC[my_new_indices(((n+1)**2)*i+(n+1)*j+k)] = \
                +kUTeval*(n+1-i-j-k)*Cim(C,n,i,j,k)\
                +kUSeval*(n+1-i-j-k)*Cjm(C,n,i,j,k)\
                +kTDeval*Cikp(C,n,i,j,k)\
                +kTUeval*Cip(C,n,i,j,k)\
                +kDTeval*Cikm(C,n,i,j,k)\
                +kDSeval*Cjkm(C,n,i,j,k)\
                +kSDeval*Cjkp(C,n,i,j,k)\
                +kSUeval*Cjp(C,n,i,j,k)\
                -((n-i-j-k)*kUTe+(n-i-j-k)*kUSe+i*kTDe +i*kTUe+k*kDTe+k*kDSe+j*kSDe+j*kSUe+ konB*FB(i,j,k,kaiBC,kaiA0,km,paramsDelta))*C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]\
                +koffB*C[my_new_indices(((n+1)**3)+((n+1)**2)*i+(n+1)*j+k)]

                dkaiBC[my_new_indices(((n+1)**2)*i+(n+1)*j+k)] = \
                +kUTeval*(n+1-i-j-k)*kaiBCim(C,n,i,j,k)\
                +kUSeval*(n+1-i-j-k)*kaiBCjm(C,n,i,j,k)\
                +kTDeval*kaiBCikp(C,n,i,j,k)\
                +kTUeval*kaiBCip(C,n,i,j,k)\
                +kDTeval*kaiBCikm(C,n,i,j,k)\
                +kDSeval*kaiBCjkm(C,n,i,j,k)\
                +kSDeval*kaiBCjkp(C,n,i,j,k)\
                +kSUeval*kaiBCjp(C,n,i,j,k)\
                -((n-i-j-k)*kUTe+(n-i-j-k)*kUSe+i*kTDe+i*kTUe+k*kDTe+k*kDSe+j*kSDe+j*kSUe+ koffB)*C[my_new_indices(((n+1)**3)+((n+1)**2)*i+(n+1)*j+k)]\
                +konB*FB(i,j,k,kaiBC,kaiA0,km,paramsDelta)*C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]

    der = np.concatenate((dkaiC,dkaiBC))
    return der

######################################################
######################################################
# in vivo ptr model
######################################################
######################################################


def dCdt_ptr(t,C,params):
     # C = [KaiCijk (7³ element) KaiBCijk (7³)]
    konB, koffB, paramsDelta, km, kaiA0, paramsk0, paramskact, params_vivo = params 
    k0UT, k0TD, k0US, k0SD, k0DS, k0DT, k0TU, k0SU = paramsk0
    kactUT, kactTD, kactUS, kactSD, kactDS, kactDT, kactTU, kactSU = paramsk0

    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta
    
    Vsptr, Vs, Vm, Km, Ks, K, V, Vd, Ki= params_vivo

    dkaiC = np.empty(int(len(C)/2))
    dkaiBC=np.empty(int(len(C)/2))
    
    kaiBC = np.sum(C[int(len(C)/2):])# 
    mRNA = C[-1]
         
    dmRNA = Vsptr - Vm*mRNA/(Km+mRNA)
   
    for i,j,k in ijk_combinations:


                kUTeval = kxyUT(i-1, j, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kUSeval  = kxyUS(i, j-1, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTDeval = kxyTD(i+1, j, k-1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTUeval = kxyTU(i+1, j, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDTeval = kxyDT(i-1, j, k+1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDSeval = kxyDS(i, j-1, k+1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSDeval = kxySD(i, j+1, k-1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSUeval = kxySU(i, j+1, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                
                kUTe = kxyUT(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kUSe = kxyUS(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTDe = kxyTD(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTUe = kxyTU(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDTe = kxyDT(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDSe = kxyDS(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSDe = kxySD(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSUe = kxySU(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                
                dkaiC[my_new_indices(((n+1)**2)*i+(n+1)*j+k)] = \
                +kUTeval*(n+1-i-j-k)*Cim(C,n,i,j,k)\
                +kUSeval*(n+1-i-j-k)*Cjm(C,n,i,j,k)\
                +kTDeval*Cikp(C,n,i,j,k)\
                +kTUeval*Cip(C,n,i,j,k)\
                +kDTeval*Cikm(C,n,i,j,k)\
                +kDSeval*Cjkm(C,n,i,j,k)\
                +kSDeval*Cjkp(C,n,i,j,k)\
                +kSUeval*Cjp(C,n,i,j,k)\
                -((n-i-j-k)*kUTe+(n-i-j-k)*kUSe+i*kTDe +i*kTUe+k*kDTe+k*kDSe+j*kSDe+j*kSUe+ konB*FB(i,j,k,kaiBC,kaiA0,km,paramsDelta))*C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]\
                +koffB*C[my_new_indices(((n+1)**3)+((n+1)**2)*i+(n+1)*j+k)]- V*C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]/(C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]+K)- Vd*C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]

                dkaiBC[my_new_indices(((n+1)**2)*i+(n+1)*j+k)] = \
                +kUTeval*(n+1-i-j-k)*kaiBCim(C,n,i,j,k)\
                +kUSeval*(n+1-i-j-k)*kaiBCjm(C,n,i,j,k)\
                +kTDeval*kaiBCikp(C,n,i,j,k)\
                +kTUeval*kaiBCip(C,n,i,j,k)\
                +kDTeval*kaiBCikm(C,n,i,j,k)\
                +kDSeval*kaiBCjkm(C,n,i,j,k)\
                +kSDeval*kaiBCjkp(C,n,i,j,k)\
                +kSUeval*kaiBCjp(C,n,i,j,k)\
                -((n-i-j-k)*kUTe+(n-i-j-k)*kUSe+i*kTDe+i*kTUe+k*kDTe+k*kDSe+j*kSDe+j*kSUe+ koffB)*C[my_new_indices(((n+1)**3)+((n+1)**2)*i+(n+1)*j+k)]\
                +konB*FB(i,j,k,kaiBC,kaiA0,km,paramsDelta)*C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]
                
                if i==0 and j==0 and k==0:
                    dkaiC[0]=dkaiC[0]+Ks*mRNA
                    
    der = np.concatenate((dkaiC,dkaiBC))
    der = np.append(der,dmRNA)
    return der




######################################################
######################################################
# scan function ptr 
######################################################
######################################################



def scan2d_invivo_ptr_Ks_Vd(x):
    my_x = x[0]
    params = x[1]
    n= x[2]
    print('myx is ', my_x)

    konB_V, koffB_V, paramsDelta, km_V, kaiA0_V, paramsk0, paramskact, params_vivo = params 
    k0UT_V, k0TD_V, k0US_V, k0SD_V, k0DS_V, k0DT_V, k0TU_V, k0SU_V = paramsk0
    kactUT_V, kactTD_V, kactUS_V, kactSD_V, kactDS_V, kactDT_V, kactTU_V, kactSU_V = paramsk0
    deltaGpT_V,deltaGpS_V,deltaGpSpT_V,deltaGU_V = paramsDelta
    Vsptr_V, Vs_V, Vm_V, Km_V, Ks_V, K_V, V_V, Vd_V, Ki_V= params_vivo
    
    # change values of Ks and Vd
    Ks_V = Ks_Values[my_x[0]]
    Vd_V = Vd_Values[my_x[1]]

    # update parameters accordingly
    paramsk0_V = k0UT_V,k0TD_V,k0US_V,k0SD_V,k0DS_V,k0DT_V,k0TU_V,k0SU_V
    paramskact_V = kactUT_V, kactTD_V, kactUS_V, kactSD_V, kactDS_V, kactDT_V, kactTU_V, kactSU_V
    paramsDelta_V = deltaGpT_V, deltaGpS_V, deltaGpSpT_V, deltaGU_V
    params_vivo_V = Vsptr_V, Vs_V, Vm_V, Km_V, Ks_V, K_V, V_V, Vd_V, Ki_V
    params_V = konB_V, koffB_V, paramsDelta_V, km_V, kaiA0_V, paramsk0_V, paramskact_V, params_vivo_V
    print('kaiA0 is ', kaiA0_V)
    try:
        print('I am trying')
        xs = solve_ivp(dCdt_ptr,[0,tmax],xs_init_specific,args=(params_V,), events=hit_ground)
        kaictot_v = np.sum(xs.y.T[:,0:-1],axis=1)
        full_phospohate = def_full_phospohate(n)
        perc_phospho_kaic = np.dot(xs.y.T[:,0:-1],full_phospohate)/kaictot_v/n
        myperiod = estimate_period(perc_phospho_kaic,xs.t)
        if (1 < myperiod <100):
            line = '%d,%d,%.3E' % (my_x[0],my_x[1],myperiod) +  '\n'
            return line
        else:
            line =  '%d,%d,%.3E' % (my_x[0],my_x[1],np.nan) +  '\n'
            return line
    except:
        print('i coudn t run it')
        myperiod = -1


######################################################
######################################################
# in vivo wt model
######################################################
######################################################




def dCdt_wt(t,C,params):
        
    konB, koffB, paramsDelta, km, kaiA0, paramsk0, paramskact, params_vivo = params 
    k0UT, k0TD, k0US, k0SD, k0DS, k0DT, k0TU, k0SU = paramsk0
    kactUT, kactTD, kactUS, kactSD, kactDS, kactDT, kactTU, kactSU = paramskact
    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta
        
    Vsptr, Vs, Vm, Km, Ks, K, V, Vd, Ki = params_vivo
    dkaiC = np.empty(int(len(C)/2))
    dkaiBC=np.empty(int(len(C)/2))
    
    kaiBC = np.sum(C[int(len(C)/2):])# 
    mRNA = C[-1]
         
    #dmRNA = Vsptr - Vm*mRNA/(Km+mRNA)
    #feedback = np.sum(C[number_states_without_B+1:])#(.9*(C[0] + C[number_states_without_B ]) + 0.5*(np.sum(C[only_D]))+np.sum(C[only_S])+0.3*np.sum(C[only_D]))
    #S dominated feedback 
    #feedback = np.sum(C[only_S_B])
    #feedback = np.dot(C[contain_S_phospho],weight_contain_S_phospho)
    removeNotB = int(len(contain_S_phospho)/2)
    feedback = np.dot(C[contain_S_phospho[removeNotB:]],weight_contain_S_phospho[removeNotB:])#todo+.9*C[number_states_without_B]
    dmRNA = Vs*Ki**4/(Ki**4 + feedback**4) - Vm*mRNA/(Km+mRNA)
    
    for i,j,k in ijk_combinations:


                kUTeval = kxyUT(i-1, j, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kUSeval  = kxyUS(i, j-1, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTDeval = kxyTD(i+1, j, k-1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTUeval = kxyTU(i+1, j, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDTeval = kxyDT(i-1, j, k+1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDSeval = kxyDS(i, j-1, k+1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSDeval = kxySD(i, j+1, k-1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSUeval = kxySU(i, j+1, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                
                kUTe = kxyUT(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kUSe = kxyUS(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTDe = kxyTD(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTUe = kxyTU(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDTe = kxyDT(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDSe = kxyDS(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSDe = kxySD(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSUe = kxySU(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                
                dkaiC[my_new_indices(((n+1)**2)*i+(n+1)*j+k)] = \
                +kUTeval*(n+1-i-j-k)*Cim(C,n,i,j,k)\
                +kUSeval*(n+1-i-j-k)*Cjm(C,n,i,j,k)\
                +kTDeval*Cikp(C,n,i,j,k)\
                +kTUeval*Cip(C,n,i,j,k)\
                +kDTeval*Cikm(C,n,i,j,k)\
                +kDSeval*Cjkm(C,n,i,j,k)\
                +kSDeval*Cjkp(C,n,i,j,k)\
                +kSUeval*Cjp(C,n,i,j,k)\
                -((n-i-j-k)*kUTe+(n-i-j-k)*kUSe+i*kTDe +i*kTUe+k*kDTe+k*kDSe+j*kSDe+j*kSUe+ konB*FB(i,j,k,kaiBC,kaiA0,km,paramsDelta))*C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]\
                +koffB*C[my_new_indices(((n+1)**3)+((n+1)**2)*i+(n+1)*j+k)]- V*C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]/(C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]+K)- Vd*C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]

                dkaiBC[my_new_indices(((n+1)**2)*i+(n+1)*j+k)] = \
                +kUTeval*(n+1-i-j-k)*kaiBCim(C,n,i,j,k)\
                +kUSeval*(n+1-i-j-k)*kaiBCjm(C,n,i,j,k)\
                +kTDeval*kaiBCikp(C,n,i,j,k)\
                +kTUeval*kaiBCip(C,n,i,j,k)\
                +kDTeval*kaiBCikm(C,n,i,j,k)\
                +kDSeval*kaiBCjkm(C,n,i,j,k)\
                +kSDeval*kaiBCjkp(C,n,i,j,k)\
                +kSUeval*kaiBCjp(C,n,i,j,k)\
                -((n-i-j-k)*kUTe+(n-i-j-k)*kUSe+i*kTDe+i*kTUe+k*kDTe+k*kDSe+j*kSDe+j*kSUe+ koffB)*C[my_new_indices(((n+1)**3)+((n+1)**2)*i+(n+1)*j+k)]\
                +konB*FB(i,j,k,kaiBC,kaiA0,km,paramsDelta)*C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]
                
                if i==0 and j==0 and k==0:
                    dkaiC[0]=dkaiC[0]+Ks*mRNA
                    
    der = np.concatenate((dkaiC,dkaiBC))
    der = np.append(der,dmRNA)
    return der

######################################################
######################################################
# in vivo wt model - another feedback function (not main )
######################################################
######################################################

def dCdt_wt_unphospho_only_feedback(t,C,params):

    konB, koffB, paramsDelta, km, kaiA0, paramsk0, paramskact, params_vivo = params 
    k0UT, k0TD, k0US, k0SD, k0DS, k0DT, k0TU, k0SU = paramsk0
    kactUT, kactTD, kactUS, kactSD, kactDS, kactDT, kactTU, kactSU = paramskact
    deltaGpT,deltaGpS,deltaGpSpT,deltaGU = paramsDelta

    Vsptr, Vs, Vm, Km, Ks, K, V, Vd, Ki = params_vivo
    dkaiC = np.empty(int(len(C)/2))
    dkaiBC=np.empty(int(len(C)/2))

    kaiBC = np.sum(C[int(len(C)/2):])# 
    mRNA = C[-1]

    #dmRNA = Vsptr - Vm*mRNA/(Km+mRNA)
    #feedback = np.sum(C[number_states_without_B+1:])#(.9*(C[0] + C[number_states_without_B ]) + 0.5*(np.sum(C[only_D]))+np.sum(C[only_S])+0.3*np.sum(C[only_D]))
    #S dominated feedback 
    #feedback = np.sum(C[only_S_B])
    #feedback = np.dot(C[contain_S_phospho],weight_contain_S_phospho)
    feedback = .9*(C[0])
    dmRNA = Vs*Ki**4/(Ki**4 + feedback**4) - Vm*mRNA/(Km+mRNA)

    for i,j,k in ijk_combinations:


                kUTeval = kxyUT(i-1, j, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kUSeval  = kxyUS(i, j-1, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTDeval = kxyTD(i+1, j, k-1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTUeval = kxyTU(i+1, j, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDTeval = kxyDT(i-1, j, k+1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDSeval = kxyDS(i, j-1, k+1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSDeval = kxySD(i, j+1, k-1, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSUeval = kxySU(i, j+1, k, kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)

                kUTe = kxyUT(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kUSe = kxyUS(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTDe = kxyTD(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kTUe = kxyTU(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDTe = kxyDT(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kDSe = kxyDS(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSDe = kxySD(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)
                kSUe = kxySU(i,j,k,kaiBC,kaiA0,km,paramsDelta,paramsk0,paramskact)

                dkaiC[my_new_indices(((n+1)**2)*i+(n+1)*j+k)] = \
                +kUTeval*(n+1-i-j-k)*Cim(C,n,i,j,k)\
                +kUSeval*(n+1-i-j-k)*Cjm(C,n,i,j,k)\
                +kTDeval*Cikp(C,n,i,j,k)\
                +kTUeval*Cip(C,n,i,j,k)\
                +kDTeval*Cikm(C,n,i,j,k)\
                +kDSeval*Cjkm(C,n,i,j,k)\
                +kSDeval*Cjkp(C,n,i,j,k)\
                +kSUeval*Cjp(C,n,i,j,k)\
                -((n-i-j-k)*kUTe+(n-i-j-k)*kUSe+i*kTDe +i*kTUe+k*kDTe+k*kDSe+j*kSDe+j*kSUe+ konB*FB(i,j,k,kaiBC,kaiA0,km,paramsDelta))*C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]\
                +koffB*C[my_new_indices(((n+1)**3)+((n+1)**2)*i+(n+1)*j+k)]- V*C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]/(C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]+K)- Vd*C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]

                dkaiBC[my_new_indices(((n+1)**2)*i+(n+1)*j+k)] = \
                +kUTeval*(n+1-i-j-k)*kaiBCim(C,n,i,j,k)\
                +kUSeval*(n+1-i-j-k)*kaiBCjm(C,n,i,j,k)\
                +kTDeval*kaiBCikp(C,n,i,j,k)\
                +kTUeval*kaiBCip(C,n,i,j,k)\
                +kDTeval*kaiBCikm(C,n,i,j,k)\
                +kDSeval*kaiBCjkm(C,n,i,j,k)\
                +kSDeval*kaiBCjkp(C,n,i,j,k)\
                +kSUeval*kaiBCjp(C,n,i,j,k)\
                -((n-i-j-k)*kUTe+(n-i-j-k)*kUSe+i*kTDe+i*kTUe+k*kDTe+k*kDSe+j*kSDe+j*kSUe+ koffB)*C[my_new_indices(((n+1)**3)+((n+1)**2)*i+(n+1)*j+k)]\
                +konB*FB(i,j,k,kaiBC,kaiA0,km,paramsDelta)*C[my_new_indices(((n+1)**2)*i+(n+1)*j+k)]

                if i==0 and j==0 and k==0:
                    dkaiC[0]=dkaiC[0]+Ks*mRNA

    der = np.concatenate((dkaiC,dkaiBC))
    der = np.append(der,dmRNA)
    return der





######################################################
######################################################
# scan function wt
######################################################
######################################################



def scan2d_invivo_wt_Ks_Vd(x):
    my_x = x[0]
    params = x[1]
    n= x[2]
    print('myx is ', my_x)

    konB_V, koffB_V, paramsDelta, km_V, kaiA0_V, paramsk0, paramskact, params_vivo = params 
    k0UT_V, k0TD_V, k0US_V, k0SD_V, k0DS_V, k0DT_V, k0TU_V, k0SU_V = paramsk0
    kactUT_V, kactTD_V, kactUS_V, kactSD_V, kactDS_V, kactDT_V, kactTU_V, kactSU_V = paramskact
    deltaGpT_V,deltaGpS_V,deltaGpSpT_V,deltaGU_V = paramsDelta
    Vsptr_V, Vs_V, Vm_V, Km_V, Ks_V, K_V, V_V, Vd_V, Ki_V= params_vivo
    
    # change values of Ks and Vd
    Ks_V = Ks_Values[my_x[0]]
    Vd_V = Vd_Values[my_x[1]]

    # update parameters accordingly
    paramsk0_V = k0UT_V,k0TD_V,k0US_V,k0SD_V,k0DS_V,k0DT_V,k0TU_V,k0SU_V
    paramskact_V = kactUT_V, kactTD_V, kactUS_V, kactSD_V, kactDS_V, kactDT_V, kactTU_V, kactSU_V
    paramsDelta_V = deltaGpT_V, deltaGpS_V, deltaGpSpT_V, deltaGU_V
    params_vivo_V = Vsptr_V, Vs_V, Vm_V, Km_V, Ks_V, K_V, V_V, Vd_V, Ki_V
    params_V = konB_V, koffB_V, paramsDelta_V, km_V, kaiA0_V, paramsk0_V, paramskact_V, params_vivo_V
    print('kaiA0 is ', kaiA0_V, 'Vs is ', Vs_V, 'Ki is ', Ki_V)
    try:
        print('I am trying')
        xs = solve_ivp(dCdt_wt,[0,tmax],xs_init_specific,args=(params_V,), events=hit_ground)
        kaictot_v = np.sum(xs.y.T[:,0:-1],axis=1)
        full_phospohate = def_full_phospohate(n)
        perc_phospho_kaic = np.dot(xs.y.T[:,0:-1],full_phospohate)/kaictot_v/n
        myperiod = estimate_period(perc_phospho_kaic,xs.t)
        if (1 < myperiod <100):
            line = '%d,%d,%.3E' % (my_x[0],my_x[1],myperiod) +  '\n'
            return line
        else:
            line =  '%d,%d,%.3E' % (my_x[0],my_x[1],np.nan) +  '\n'
            return line
    except:
        print('i coudn t run it')
        myperiod = -1





######################################################
######################################################
# plot result of the scan function
######################################################
######################################################
        
def twodscan_figure(Npoints, oscillations,file_to_save_figure,name_x,name_y,name_values_x,name_values_y, vmin=0, vmax=60):
    fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(16, 8),tight_layout=True)
    #the main figure
    pos = axes.imshow(oscillations, vmin=vmin, vmax=vmax)
    axes.set_xticks(np.arange(-.5, Npoints, 1), minor=True)
    axes.set_yticks(np.arange(-.5, Npoints, 1), minor=True)
    axes.grid(which ='minor',color='black', linestyle='-', linewidth=1)

    axes.set_xlabel(name_x)
    axes.set_ylabel(name_y)
    axes.set_xticks(range(0, Npoints, 1))
    axes.set_yticks(range(0, Npoints, 1))
    #use the values of konB for the ticks on the x axis: 
    xlabels = ['INF' if np.isinf(i) else '%d' % i if int(i) == i else '%.1f' % i 
                   if round(i, 1) == i else '%.2f' % i 
                   if round(i, 2) == i else '%.2E' % i for i in name_values_x]
    ylabels = ['INF' if np.isinf(i) else '%d' % i if int(i) == i else '%.1f' % i 
               if round(i, 1) == i else '%.2f' % i 
               if round(i, 2) == i else '%.2E' % i for i in name_values_y]

    if Npoints > 10:
        tobekept = np.linspace(0,Npoints-1,5).astype(int) 
        my_chosen_ticks= np.ones(Npoints-len(tobekept))
        jj = 0 
        for kk in range(Npoints):
            if kk not in ll:
                my_chosen_ticks[jj]=kk
                jj = jj+1
        my_chosen_ticks = my_chosen_ticks.astype(int) 
        for i in my_chosen_ticks: 
            xlabels[i]=''
    axes.set_xticklabels(xlabels, rotation=90)
    axes.set_yticklabels(ylabels)

    cax = fig.add_axes([.75, 0.15, 0.03, 0.8])
    #https://towardsdatascience.com/the-many-ways-to-call-axes-in-matplotlib-2667a7b06e06
    cax.set_xlabel('Period (min)')
    fig.colorbar(pos, cax=cax)
    #
  
    plt.savefig(file_to_save_figure)
    plt.show()              





############## function to test the multiprocessing 

def test_f(x_params):
    x,a,b = x_params
    print('hello test', x)
    line = '%d,%d' % (a*x[0],x[1]+b) +  '\n'
    print('line', line)
    return line

