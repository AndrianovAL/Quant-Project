from scipy.stats import norm
import pylab
import pandas as pd
import numpy as np
import py_vollib.black as bk # http://vollib.org/documentation/python/1.0.2/apidoc/py_vollib.black.html
                             # http://vollib.org/documentation/python/0.1.5/apidoc/vollib.black_scholes.html
import py_vollib.black.greeks.analytical as greek # http://vollib.org/documentation/python/1.0.2/apidoc/py_vollib.black.greeks.html
import py_vollib
import py_vollib.black.implied_volatility as bk_iv

PLOT = True
#=====================================================================
# 1C
def black76(flag:str, F:float, K:float, t:float, r:float, sigma:float):
    '''Calculate the (discounted) Euro Futures Option Price

    flag (str): 'c' for Call, 'p' for Put
    F (float): underlying futures price
    K (float): strike price
    t (float): time to expiration in years
    r (float): the risk-free interest rate
    sigma (float): annualized volatility (standard deviation)
    '''
    d1 = (np.log(F/K) + (sigma**2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if   flag == 'c':
        return np.exp(-r*t) * (F*norm.cdf( d1) - K*norm.cdf( d2))
    elif flag == 'p':
        return np.exp(-r*t) * (F*norm.cdf(-d2) - F*norm.cdf(-d1))

def my_greeks(F, K, t, r, iv):
    d1 = (np.log(F/K) + iv**2 * t / 2) / (iv*np.sqrt(t))
    d2 = (np.log(F/K) - iv**2 * t / 2) / (iv*np.sqrt(t))
    assert round(d2, 10) == round(d1 - iv * np.sqrt(t), 10)

    gammaC = gammaP = np.exp(-r*t) / (F*iv*np.sqrt(t)) / np.sqrt(2*np.pi) * np.exp(-d1**2 / 2)
    vegaC = vegaP = F * np.exp(-r*t) * np.sqrt(t) / np.sqrt(2*np.pi) * np.exp(-d1**2 / 2)

    term1cp = F*iv * np.exp(-r*t) / (2*np.sqrt(2*np.pi*t)) * np.exp(-d1**2 / 2)  
    term2c = -r * F * np.exp(-r*t) * norm.cdf(d1)
    term3c =  r * K * np.exp(-r*t) * norm.cdf(d2)
    thetaC = -(term1cp + term2c + term3c)
    term2p = -r * F * np.exp(-r*t) * norm.cdf(-d1)
    term3p =  r * K * np.exp(-r*t) * norm.cdf(-d2)
    thetaP = -term1cp + term2p + term3p

    return gammaC, gammaP, vegaC/100, vegaP/100, thetaC/365., thetaP/365.

# def plot_PutCall(strikes, iv, name):
    pylab.figure(name)
    pylab.clf()
    pylab.grid()
    pylab.rcParams['lines.linewidth']=1
    pylab.rcParams['xtick.major.size']=7
    pylab.rcParams['ytick.major.size']=7
    pylab.rcParams['lines.markersize']=10
    pylab.rcParams['legend.numpoints']=1
    pylab.rcParams['xtick.labelsize']=16
    pylab.rcParams['ytick.labelsize']=16
    pylab.plot(strikes, iv, label='Implied Volatility')
    pylab.legend()

def plot_multi(nameFigure, Xs, Ys:tuple, names:tuple, nameX, nameY):
    pylab.figure(nameFigure)
    pylab.clf()
    pylab.grid()
    pylab.xlabel(nameX)
    pylab.ylabel(nameY)
    pylab.rcParams['lines.linewidth']=1
    pylab.rcParams['xtick.major.size']=7
    pylab.rcParams['ytick.major.size']=7
    pylab.rcParams['lines.markersize']=10
    pylab.rcParams['legend.numpoints']=1
    pylab.rcParams['xtick.labelsize']=16
    pylab.rcParams['ytick.labelsize']=16
    for i in range(len(Ys)):
        pylab.plot(Xs, Ys[i], label=names[i])
    pylab.legend()

df = pd.read_csv('data.csv')

K_index = 21
K = int(df[K_index:K_index+1]['Strike'])
C0 = float(df[K_index:K_index+1]['Call'])
P0 = float(df[K_index:K_index+1]['Put'])

F = 2900 # underlying asset (i.e. Dec20 ES Mini Futures) price
r = 0.0250
t = 99/252 #today is Aug1, there're 99 trading days until Dec20 and 252 trading days throughout the year

iv = bk.implied_volatility.implied_volatility_of_discounted_option_price(
    P0, F, K, r, t, flag='p')

risk_free_rates = np.arange(0, 1, 0.01)
f_call = lambda r: bk.black('c', F, K, t, r, iv)
f_put  = lambda r: bk.black('p', F, K, t, r, iv)
f_d1 = lambda r: (np.log(F/K) + iv**2 * t / 2) / (iv*np.sqrt(t))
f_d2 = lambda r: (np.log(F/K) - iv**2 * t / 2) / (iv*np.sqrt(t))

Ycall = np.array(list(map(f_call, risk_free_rates)))
Yput  = np.array(list(map(f_put, risk_free_rates)))
Yd1 = np.array(list(map(f_d1, risk_free_rates)))
Yd2 = np.array(list(map(f_d2, risk_free_rates)))
if PLOT:
    plot_multi('Call(r), Put(r)', risk_free_rates, (Ycall, Yput), ('Call', 'Put'), 'risk-free rate, r', 'Price')
    plot_multi('d1(r), d2(r)', risk_free_rates, (Yd1, Yd2), ('d1', 'd2'), 'risk-free rate, r', '')

#====================================================================
# 1D
def f_greeks(t):
    iv = bk.implied_volatility.implied_volatility_of_discounted_option_price(
    P0, F, K, r, t, flag='c')
    gammaC = greek.gamma('c', F, K, t, r, iv)
    # gammaP = greek.gamma('p', F, K, t, r, iv) # should be equal to gammaC
    vegaC  = greek.vega ('c', F, K, t, r, iv)
    # vegaP  = greek.vega ('p', F, K, t, r, iv) # should be equal to vegaP C
    thetaC = greek.theta('c', F, K, t, r, iv)
    thetaP = greek.theta('p', F, K, t, r, iv)
    deltaC = greek.delta('c', F, K, t, r, iv)
    deltaP = greek.delta('p', F, K, t, r, iv)
    return gammaC, vegaC, thetaC, thetaP, deltaC, deltaP

K_index = 17
K = int(df[K_index:K_index+1]['Strike'])
C0 = float(df[K_index:K_index+1]['Call'])
P0 = float(df[K_index:K_index+1]['Put'])
F = 2900 # underlying asset (i.e. Dec20 ES Mini Futures) price
r = 0.0250
times = np.arange(0.001, 1, 1/365)

Ys=np.array(list(map(f_greeks, times)))
YgammaC = Ys[:,0]
YvegaC  = Ys[:,1]
YthetaC = Ys[:,2]
YthetaP = Ys[:,3]
YdeltaC = Ys[:,4]
YdeltaP = Ys[:,5]
if PLOT:
    plot_multi('Gamma(t)', times*365, (YgammaC, ), ('Gamma', ), 'Time', 'Greeks')
    plot_multi('Vega(t)', times*365, (YvegaC, ), ('Vega', ), 'Time', 'Greeks')
    plot_multi('Theta_C(t)), Theta_P(t))', times*365, (YthetaC, YthetaP), ('Theta(C)', 'Theta(P)'), 'Time', 'Greeks')
