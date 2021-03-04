import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

'''
I wanted to use "real" data for this but real data is a jerk and isn't usually linear
data = pd.read_csv('realestate.csv')
x = data['X2 house age'].values
y = data['Y house price of unit area'].values
'''
x = []
y = []

for i in range(400):
    x.append(i)
    y.append(5 + i * 1.25 * (100/random.randint(90,110))) #come back to this it is an affront to God

def Gradient_Descent (x, y, lr, its):
    theta_1 = np.random.randint(low = 1, high = 2)
    theta_0 = np.random.randint(low = 1, high = 2) #random starting points for our theta values
    m = x.shape[0]

    def avg_error(a,b): # J(theta0, theta1) this is what we're trying to minimize to get the "best fit"
        sum_mean = 0
        for i in range(m):
            sum_mean += a[i] - b[i]
        return sum_mean/m

    def descent(theta_0, theta_1, m): #this is our gradient descent function
        h_xi = theta_0 + theta_1*x #Our hypothesis equaton for linear regression
        temp_0 = theta_0 - lr * avg_error(h_xi, y) #Gradient Descent for theta0
        temp_1 = theta_1 - lr * avg_error(h_xi*x, y*x) #Gradient Descent for theta1
        return temp_0, temp_1

    fig = plt.figure() #builds our figure
    ax = fig.add_subplot(111) #1x1 grid, first subplot

    for i in range(its): #cycling through our number of iteratons
        theta_0, theta_1 = descent(theta_0, theta_1, m) # a single gradient descent pass
        print(theta_0 ,' ', theta_1) #text visualization of the values
        
    ''' all this stuff below can go into the for loop if you want to watch the regression in real time, but you're looking at it taking 10x
        the time. '''
    ax.clear() #clear whatever plot is present
    ax.plot(x,y, linestyle = 'None', marker = 'o') #define the point values of our data set
    ax.plot(x, theta_0 + theta_1*x) #define the current iteration of our hypothesis
    plt.show(block=True) # not this though, this has to stay here because VSC wants to kill everything I love when it finishes.

x = np.array(x)
y = np.array(y)

Gradient_Descent(x,y,0.000005,500)
