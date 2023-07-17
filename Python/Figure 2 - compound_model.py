# -*- coding: utf-8 -*-

"""

Created on Wed Jun 23 13:48:13 2021


Compute the index for the compound model. Running the code produces the txt file which is then used to produce the figure of the article.


"""





# ----------------------------------------------------------------------------

# Imports

# ----------------------------------------------------------------------------



print("Module importation...")



from time import time



from math import *



import scipy.integrate as integrate
import scipy.special as special
import numpy as np




# Name of the file to store the results

nameFile = "data_compound.txt"





# ----------------------------------------------------------------------------

# Auxilliary function: inverse exponential integral

# ----------------------------------------------------------------------------



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

    

# ----------------------------------------------------------------------------

# Auxilliary functions: integrand

# ----------------------------------------------------------------------------



def integrandNu(u,s,d,phi):

    return np.exp(-s/u)*np.power(1-u,phi-1) / np.power(u, d*phi+1)



def integrandUNuPlus(u,s,d,phi):

    return special.gammaincc(d*phi, s/u) *np.power(1-u,phi-1) / u



def nuPlus(s,d,phi):

    return np.power(s,d *phi-1) / special.gamma(d*phi) * integrate.quad(integrandNu, 0., 1. , args=(s,d,phi),points=(0.,1.))[0]



def UNuPlus(s,d,phi):

    return integrate.quad(integrandUNuPlus, 0., 1., args=(s,d,phi),points=(0.,1.))[0]



def integrandNorm(s,d):

    return invexp1(d*special.exp1(s)) * np.exp(-s)





def integrandNumerator(s,d,phi):

    return max(s*invexp1( UNuPlus(s,d,phi) ) * nuPlus(s,d,phi),0.)     




# ----------------------------------------------------------------------------

# Auxilliary (rescaled) functions: integrand

# ----------------------------------------------------------------------------


def integrandNuR(u,s,d,phi):

    return np.exp(-s/u) / np.power(u, d*phi+1)



def integrandUNuPlusR(u,s,d,phi):

    return special.gammaincc(d*phi, s/u)  / u



def nuPlusR(s,d,phi):

    return 1. / special.gamma(d*phi) * integrate.quad(integrandNuR, 0., 1., args=(s,d,phi), weight = 'alg', wvar=(0,phi-1))[0]



def UNuPlusR(s,d,phi):

    return integrate.quad(integrandUNuPlusR, 0., 1, args=(s,d,phi), weight = 'alg', wvar=(0,phi-1))[0]



def integrandNumeratorR(s,d,phi):

    return max(s*invexp1( UNuPlusR(s,d,phi) ) * nuPlusR(s,d,phi),0.) 



def integrandNumeratorRBis(s,d,phi):

    return max(np.power(s,d*phi)*invexp1( UNuPlusR(s,d,phi) ) * nuPlusR(s,d,phi),0.) 



# ----------------------------------------------------------------------------

# Computation of the integrals and the index

# ----------------------------------------------------------------------------



start = time()

# Compute for all values of z and d

nPhi = 30

nD = 4

phiArray = np.linspace(0.1,5.,nPhi)

dArray = np.array([2,3,4,5], dtype= int)



integralArray = np.zeros((nPhi,nD))

indexArray = np.zeros((nPhi,nD))




for j in range(nD):

    

    d = dArray[j]

    

    # Compute the normalizing integral

    normInt = integrate.quad(integrandNorm, 0., 500., args=(d),points=(0.))[0]

    

    

    for i in range(nPhi):

        phi = phiArray[i]

        # compute = integrate.quad(integrandNumerator, 1e-5, 100., args=(d,phi),points=(0.))

        compute = integrate.quad(integrandNumeratorRBis, 0., 500., args=(d,phi),points=(0.))

        # compute = integrate.quad(integrandNumeratorR, 0., 100., args=(d,phi), weight = 'alg', wvar=(d*phi-1,0.))

        integralArray[i,j] = compute[0]



        # Compute the index

        indexArray[i,j] = 1. - (d - integralArray[i,j])/(d - d*normInt)



end = time()



print("Elapsed time: "+str(round(end - start,2))+"s." )




# ----------------------------------------------------------------------------

# Writing the results in a file ready for a tikz reading

# ----------------------------------------------------------------------------





fileObject = open(nameFile, 'w')



# First line: name of the columns



fileObject.write("x ")

for j in range(nD):

    fileObject.write("y"+str(dArray[j]-1)+" ")

fileObject.write("\n")



# Next lines: value of the index



for i in range(nPhi):

    fileObject.write(str(phiArray[i])+" ")

    for j in range(nD):

        fileObject.write(str(indexArray[i,j])+" ")

    fileObject.write("\n")





fileObject.close()    