# -*- coding: utf-8 -*-

"""

Created on Wed Jun 23 10:25:06 2021


Compute the index for the additive model. Running the code produces the txt file which is then used to produce the figure of the article.

"""



# Name of the file to store the results

nameFile = "data_additive.txt"



# ----------------------------------------------------------------------------

# Imports

# ----------------------------------------------------------------------------



print("Module importation...")



from math import *



import scipy.integrate as integrate

import scipy.special as special

import numpy as np



import matplotlib.pyplot as plt



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


# Integrand of the expressions in Proposition 7.

def integrand(s,z,d):

    return (d*(1.-z)*np.exp(-s) + z*np.exp(-s/d))* invexp1(  d*(1-z)*special.exp1(s) + z*special.exp1(s/d) )





# ----------------------------------------------------------------------------

# Computation of the integrals and the index

# ----------------------------------------------------------------------------


# Value of the second moment of bar nu.
secondMoment = 1.




# Compute for all values of z and d

nZ = 20

nD = 3

zArray = np.linspace(0,1,nZ)

dArray = np.array([2,3,4], dtype= int)



integralArray = np.zeros((nZ,nD))

indexArray = np.zeros((nZ,nD))



for j in range(nD):

    for i in range(nZ):

        z = zArray[i]

        d = dArray[j]

        compute = integrate.quad(integrand, 0., 500., args=(z,d))

        integralArray[i,j] = compute[0]



        # Compute the index

        indexArray[i,j] = 1. - (d*secondMoment - integralArray[i,j])/(d*secondMoment - integralArray[0,j])





# ----------------------------------------------------------------------------

# Writing the results in a file ready for a tikz reading

# ----------------------------------------------------------------------------





fileObject = open(nameFile, 'w')



# First line: name of the columns



fileObject.write("x y1 ")

for j in range(nD):

    fileObject.write("y"+str(dArray[j])+" ")

fileObject.write("\n")



# Next lines: value of the index



for i in range(nZ):

    fileObject.write(str(zArray[i])+" "+str(zArray[i])+" ")

    for j in range(nD):

        fileObject.write(str(indexArray[i,j])+" ")

    fileObject.write("\n")





fileObject.close()    

    

    

    









    