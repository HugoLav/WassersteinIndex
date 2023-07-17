# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 18:07:36 2023

Compute the index for the situation described in Section 5.3. Running the code produces the txt file which is then used to produce the figure of the article.


"""


# ----------------------------------------------------------------------------

# Imports

# ----------------------------------------------------------------------------

from time import time

from math import *

import scipy.integrate as integrate
import scipy.special as special
import numpy as np


# ----------------------------------------------------------------------------

# Some auxilliary functions

# ----------------------------------------------------------------------------



# inverse of exponential integral

def invexp1(y, tol = 1e-8, maxCount = 50):
    """
    Inverse of the exponentional integral function found via Newton's method
    """
    
    if y > 1e3:
        return np.exp(-np.euler_gamma - y)
    elif y < 1e-10:
        return np.real(special.lambertw(1/y))
    else:
        # Find initial guess: use of LambertW
        x = np.real(special.lambertw(1/y))
        # Plus small loop to start with the right location 
        while special.exp1(x) < y:
            x/=2
        # Newton's loop
        count = 1
        while (abs(special.exp1(x) - y) > tol) and (count <= maxCount) :
            f = special.exp1(x) - y 
            df =  - np.exp(-x)/x
            x -= f/df
            count += 1
        return x
    
# additive
def integrand_add(s,z,d):
    
    inner = invexp1(  d*(1-z)*special.exp1(s) + z*special.exp1(s/d) )
    outer = d*(1.-z)*np.exp(-s) + z*np.exp(-s/d)
    
    return inner * outer

# generalized mix dependence-independence (nEx + nInd = d)
def integrand_mix(s,groupV):
    
    inner = invexp1( np.sum([special.exp1(s/group) for group in groupV]))
    outer = np.sum([np.exp(-s/group) for group in groupV])
    
    return inner * outer

def upperbound(s, m):
    
    inner = invexp1( m*special.exp1(s))
    outer = np.exp(-s)
    
    return inner * outer


# ----------------------------------------------------------------------------

# keep number of independent components fixed and let number of comonotone repetitions group

# ----------------------------------------------------------------------------



start = time()

nameFileMfix = "mix_mfix.txt" 


indV = [1,2,5,10]
nInd = len(indV)
comV = range(2,11)
nCom = len(comV)
m2 = 1

indexM = np.zeros((nInd, nCom))
upperV = np.zeros(nInd)

for indIndex in range(nInd):
    
    ind = indV[indIndex]
    upperV[indIndex] = integrate.quad(upperbound, 0., 500., args=(ind))[0]/m2
    
    for comIndex in range(nCom):
    
        com = comV[comIndex]
        d = ind * com
        wassInd = 2*d*m2 - 2*integrate.quad(integrand_add, 0., 500., args=(0,d))[0]
    
        groupV = np.repeat(com, ind)
        wass = 2*d*m2 - 2*integrate.quad(integrand_mix, 0., 500., args= groupV)[0]
    
        indexM[indIndex, comIndex] = 1 - wass/wassInd
        
        
        

# Writing the results in a file ready for a tikz reading

fileObject = open(nameFileMfix, 'w')



# First line: name of the columns
fileObject.write("x ")

for j in range(nInd):
    fileObject.write("index"+str(j+1)+" ")
    
for j in range(nInd):
    fileObject.write("limit"+str(j+1)+" ")

fileObject.write("\n")



# Next lines: values of the index
for i in range(nCom):

    fileObject.write(str(comV[i])+" ")
    for j in range(nInd):
        fileObject.write(str(indexM[j,i])+" ")
    for j in range(nInd):
        fileObject.write(str(upperV[j])+" ")
        
    fileObject.write("\n")    


fileObject.close()      

# ----------------------------------------------------------------------------

# keep number of comonotone ripetitions fixed and let number of independent components increase

# ----------------------------------------------------------------------------

nameFileNfix = "mix_nfix.txt" 

comV = [1,2,5,10]
nCom = len(comV)
indV = range(2,11)
nInd = len(indV)
m2 = 1

indexM = np.zeros((nCom, nInd))

for comIndex in range(nCom):
    
    com = comV[comIndex]
    
    for indIndex in range(nInd):
    
        ind = indV[indIndex]
        d = ind * com
        wassInd = 2*d*m2 - 2*integrate.quad(integrand_add, 0., 500., args=(0,d))[0]
    
        groupV = np.repeat(com, ind)
        wass = 2*d*m2 - 2*integrate.quad(integrand_mix, 0., 500., args= groupV)[0]
    
        indexM[comIndex, indIndex] = 1 - wass/wassInd


# Writing the results in a file ready for a tikz reading

fileObject = open(nameFileNfix, 'w')



# First line: name of the columns
fileObject.write("x ")

for j in range(nCom):
    fileObject.write("index"+str(j+1)+" ")
    

fileObject.write("limits"+str(j+1)+" ")
fileObject.write("\n")



# Next lines: values of the index
for i in range(nInd):

    fileObject.write(str(indV[i])+" ")
    for j in range(nCom):
        fileObject.write(str(indexM[j,i])+" ")
    fileObject.write("0. ")
    fileObject.write("\n")    


fileObject.close() 

end = time()
print("Elapsed time: "+str(round(end - start,2))+"s." )
