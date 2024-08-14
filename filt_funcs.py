#----------------------------------------------
# Tratamiento Digital de Señales Financieras
# Pablo Marchesi 
# Junio 2024
#----------------------------------------------

from scipy.signal import butter
import pandas as pd 
import numpy as np
from math import pi, cos, sin, sqrt, acos
from scipy.optimize import newton

# Siguiente potencia de dos 
def nextpow2(N):
    n = 1
    while n < N: n *= 2
    return n

# Metodo de Newton para obtener el perido de la SMA a partir de la frecuencia de corte
def N_sma(wc):

    wc = wc*pi
    func = lambda N: np.sin(wc*N/2) - (N/np.sqrt(2))*(np.sin(wc/2))
    deriv = lambda N: (wc/2)*np.cos(wc*N/2) - (1/np.sqrt(2))*np.sin(wc/2)
    N_0 = pi/wc  
    return int(np.round(newton(func, N_0, deriv)))

# Metodo de Newton para obtener el la frecuencia de corte a partir del periodo
def wc_sma(N, **kwargs):
    func = lambda w: sin(N*w/2) - N/sqrt(2) * sin(w/2)  
    deriv = lambda w: cos(N*w/2) * N/2 - N/sqrt(2) * cos(w/2) / 2  
    omega_0 = pi/N  
    return newton(func, omega_0, deriv, **kwargs)/pi

# Filtro media movil en funcion de la frecuencia de corte 
def sma(x, wc):
    
    N = N_sma(wc)
    y = pd.Series(x).rolling(N).mean()
    return y 

# Metodo de Newton para obtener la frecuencia de corte de la EMA a partir de alpha
def wc_ema(alpha):
    return (acos((alpha**2 + 2*alpha - 2) / (2*alpha - 2)))/pi

# Metodo de Newton para obtener el parametro alpha de la EMA a partir de la frecuencia de corte
def alpha_ema(wc):
    wc = wc*pi
    B = 2*(1-cos(wc)); C = 2*(cos(wc)-1)
    func = lambda alpha: alpha**2 + B*alpha + C 
    deriv = lambda alpha: 2*alpha + B
    alpha_0 = 0.5
    return round(newton(func, alpha_0, deriv),3)

# Filtro butterworth en funcion de la frecuencia de corte
def butterworth(x,wc):
    N = 1
    B,A = butter(N,wc)
    y = []

    for n in range(len(x)):
        if n == 0:
            y.append(x[0])
        else:
            y_n = B[0]*(x[n] + x[n-1]) - A[1]*y[n-1]
            y.append(y_n)
    return y

# Filtro supersmoother en funcion de la frecuencia de corte
def smooth(x,wc):

    a = np.exp(-np.sqrt(2)*np.pi*wc/2)
    b = 2*a*np.cos(np.sqrt(2)*np.pi*wc/2)

    c2 = b
    c3 = -a*a
    c1 = 1-c2-c3

    y = []

    for n in range(len(x)):
        if n == 0:
            y.append(x[0])
        else:
            y_n = (c1/2)*(x[n] + x[n-1]) + c2*y[n-1] + c3*y[n-2]
            y.append(y_n)
    return y

# Filtro media movil exponencial en funcion de la frecuencia de corte
def ema(x, wc):
    alpha = alpha_ema(wc)
    y = []

    for n in range(len(x)):
        if n == 0:
            y.append(x[0])
        else:
            y_n = alpha*x[n] + (1-alpha)*y[n-1]
            y.append(y_n)
    return y

# Metodo de Newton para obtener el perido de la DMA a partir de la frecuencia de corte
def N_dma(wc):
    wc = wc*pi
    func = lambda N: sin(wc*N/2) - (N/(2**(1/4)))*(sin(wc/2))
    deriv = lambda N: (wc/2)*cos(wc*N/2) - (1/(2**(1/4)))*sin(wc/2)
    N_0 = pi/wc  
    return int(np.round(newton(func, N_0, deriv)))

# Metodo de Newton para obtener la frecuencia de corte de la DMA a partir del periodo
def wc_dma(N):
    func = lambda wc: sin(N*wc/2) - (N/(2**(1/4))) * sin(wc/2)  
    deriv = lambda wc: cos(N*wc/2) * N/2 - (N/(2**(1/4))) * cos(wc/2) / 2  
    omega_0 = pi/N  
    return newton(func, omega_0, deriv)/pi

def dma(x, wc):
    N = N_dma(wc)
    h = (1/N)*np.ones(N,)
    B = np.convolve(h, h, mode='full')

    y = []
    for n in range(len(x)):
        if n < len(B):
            y.append(np.NAN)
        else:
            y_n = np.dot(B,x[n-len(B):n])
            y.append(y_n)

    return y