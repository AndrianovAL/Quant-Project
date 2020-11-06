from scipy.stats import norm
import pylab
import pandas as pd
import numpy as np
import py_vollib.black as bk # http://vollib.org/documentation/python/1.0.2/apidoc/py_vollib.black.html
                             # http://vollib.org/documentation/python/0.1.5/apidoc/vollib.black_scholes.html
import py_vollib.black.greeks.analytical as greek # http://vollib.org/documentation/python/1.0.2/apidoc/py_vollib.black.greeks.html
import py_vollib
import py_vollib.black.implied_volatility as bk_iv

#====================================================================
# 2A)
PLOT = True

def black76(flag:str, F:float, K:float, t:float, r:float, sigma:float):
    '''Calculate the (discounted) Computes Euro Futures Option Price

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

def plot_PutCall(strikes, iv, name):
    pylab.figure(name)
    pylab.clf()
    pylab.grid()
    # pylab.ylim(0.095,0.18)
    pylab.xlabel('Strike')
    pylab.ylabel('Implied Volatility')
    pylab.rcParams['lines.linewidth']=1
    pylab.rcParams['xtick.major.size']=7
    pylab.rcParams['ytick.major.size']=7
    pylab.rcParams['lines.markersize']=10
    pylab.rcParams['legend.numpoints']=1
    pylab.rcParams['xtick.labelsize']=16
    pylab.rcParams['ytick.labelsize']=16
    pylab.plot(strikes, iv, label='Implied Volatility')
    pylab.legend()

def plot2(nameFigure, strikes, ivPP, ivCC, nameP, nameC, nameX, nameY):
    pylab.figure(nameFigure)
    # pylab.xlim(0.5,1.5)
    pylab.clf()
    pylab.grid()
    # pylab.ylim(0.095,0.18)
    pylab.xlabel(nameX)
    pylab.ylabel(nameY)
    pylab.rcParams['lines.linewidth']=1
    pylab.rcParams['xtick.major.size']=7
    pylab.rcParams['ytick.major.size']=7
    pylab.rcParams['lines.markersize']=10
    pylab.rcParams['legend.numpoints']=1
    pylab.rcParams['xtick.labelsize']=16
    pylab.rcParams['ytick.labelsize']=16
    pylab.plot(strikes, ivPP, label=nameP)
    pylab.plot(strikes, ivCC, label=nameC)
    pylab.legend()

df = pd.read_csv('data.csv')
    
F = 2900 # underlying asset (i.e. Dec20 ES Mini Futures) price
r = 0.0250
t = 99/252 #today is Aug1, there're 99 trading days until Dec20 and 252 trading days throughout the year

# Out-of-the-Money means: call options with strikes > 2900, put options with strikes < 2900
strikes = list(df['Strike'][0:20])+list(df['Strike'][21:41])
iv = [None]*40 #0:19 for K=2500:2880; 20:39 for K=2920:3300
ivPP = [None]*40
ivCC = [None]*40
for i in range(20): #here strikes are < 2900, so we'll use put options
    K = df['Strike'][i]
    discounted_call_price = df['Call'][i]
    discounted_put_price = df['Put'][i]
    iv[i] = bk.implied_volatility.implied_volatility_of_discounted_option_price(discounted_put_price, F, K, r, t, flag='p')
    ivPP[i] = bk.implied_volatility.implied_volatility_of_discounted_option_price(discounted_put_price, F, K, r, t, flag='p')
    ivCC[i] = bk.implied_volatility.implied_volatility_of_discounted_option_price(discounted_call_price, F, K, r, t, flag='c')
for i in range(21, 41): #here strikes are > 2900, so we'll use call options
    K = df['Strike'][i]
    discounted_call_price = df['Call'][i]
    discounted_put_price = df['Put'][i]
    iv[i-1] = bk.implied_volatility.implied_volatility_of_discounted_option_price(discounted_call_price, F, K, r, t, flag='c')
    ivPP[i-1] = bk.implied_volatility.implied_volatility_of_discounted_option_price(discounted_put_price, F, K, r, t, flag='p')
    ivCC[i-1] = bk.implied_volatility.implied_volatility_of_discounted_option_price(discounted_call_price, F, K, r, t, flag='c')
if PLOT:
    plot_PutCall(strikes, iv, 'IV from OTM Puts and Calls')
    plot2('IV from all Calls and from all Puts', strikes, ivCC, ivPP, 'IV from all Calls', 'IV from all Puts', 'Strike', 'Implied Volatility')
    
    pylab.figure('Strike - Implied Volatility');
    pylab.grid()
    pylab.xlabel('Strike')
    pylab.ylabel('Implied Volatility')
    pylab.plot(strikes, iv, 'bo', label='Implied Volatility')
    model1 = pylab.polyfit(strikes, iv, 1)
    estIV1 = pylab.polyval(model1, strikes)
    model2 = pylab.polyfit(strikes, iv, 2)
    estIV2 = pylab.polyval(model2, strikes)
    pylab.plot(strikes, estIV1, 'g', label='Linear fit')
    pylab.plot(strikes, estIV2, 'r--', label='Quadratic fit')
    pylab.legend(loc='best')

print('max IV =', iv[np.argmax(iv)], 'is observed for Strike = ', df['Strike'][np.argmax(iv)])
print('min IV =', iv[np.argmin(iv)], 'is observed for Strike = ', df['Strike'][np.argmin(iv) + 1])

#====================================================================
# 2D): checking Put-Call Parity
put_call_parity =[None]*41
for i in range(41):
    put_call_parity[i] = df['Call'][i] + df['Strike'][i]*np.exp(-r*t) - df['Put'][i] - F * np.exp(-r*t)
    if round(put_call_parity[i], 1) != 0:
        print('For strike =', df['Strike'][i], 'put-call parity is violated:', round(put_call_parity[i], 1))

print('K=', df['Strike'][18],
      ': P=', df['Put'][18],
      '\t\tCalP=', round(bk.black('p', F, df['Strike'][18], t, r, sigma=iv[18]), 2),
      '\tC', df['Call'][18],
      '\tCalC=', round(bk.black('c', F, df['Strike'][18], t, r, sigma=iv[18]), 2), sep='')
print('K=', df['Strike'][40],
      ': P=', df['Put'][40],
      '\tCalP=', round(bk.black('p', F, df['Strike'][40], t, r, sigma=iv[39]), 2),
      '\tC=', df['Call'][40],
      '\tCalC=', round(bk.black('c', F, df['Strike'][40], t, r, sigma=iv[39]), 2), sep='')

#=====================================================================
# 2E), 2F)
# Computing Delta & Gamma of Call & Put (Gamma should be identical)
F=2900
deltaP = [None]*41
deltaC = [None]*41
gammaP = [None]*41
gammaC = [None]*41
iv2900c=bk.implied_volatility.implied_volatility_of_discounted_option_price(98.6, 2900, 2900, r, t, flag='c')
iv2900p=bk.implied_volatility.implied_volatility_of_discounted_option_price(98.6, 2900, 2900, r, t, flag='p')

for i in range(20):
    K=df['Strike'][i]
    deltaP[i] = greek.delta('p', F, K, t, r, iv[i])
    deltaC[i] = greek.delta('c', F, K, t, r, iv[i])
    gammaP[i] = greek.gamma('p', F, K, t, r, iv[i])
    gammaC[i] = greek.gamma('c', F, K, t, r, iv[i])
K = df['Strike'][20]
assert K == 2900
deltaP[20] = greek.delta('p', F, K, t, r, iv2900p)
deltaC[20] = greek.delta('c', F, K, t, r, iv2900c)
gammaP[20] = greek.gamma('p', F, K, t, r, iv2900p)
gammaC[20] = greek.gamma('c', F, K, t, r, iv2900c)
for i in range(21, 41):
    K=df['Strike'][i]
    deltaP[i] = greek.delta('p', F, K, t, r, iv[i-1])
    deltaC[i] = greek.delta('c', F, K, t, r, iv[i-1])
    gammaP[i] = greek.gamma('p', F, K, t, r, iv[i-1])
    gammaC[i] = greek.gamma('c', F, K, t, r, iv[i-1])
if PLOT:
    plot2('Deltas', list(df['Strike']), deltaP, deltaC, 'Delta(Put)', 'Delta(Call)', 'Strike', 'Delta')
    plot2('Gammas', list(df['Strike']), gammaP, gammaC, 'Gamma(Put)', 'Gamma(Call)', 'Strike', 'Gamma')

#====================================================================
# Creating Delta-Neutral portfolios
F = 2900
K = 2700
P2700 = 34.25
ivP2700 = bk.implied_volatility.implied_volatility_of_discounted_option_price(
    P2700, 2900, K, r, t, flag='p')
DeltaP2700 = greek.delta('p', F, K, t, r, ivP2700)
K = 3100
C3100 = 23.69
ivC3100 = bk.implied_volatility.implied_volatility_of_discounted_option_price(
    C3100, 2900, K, r, t, flag='c')
DeltaC3100 = greek.delta('c', F, K, t, r, ivC3100)
print('DeltaP2700 =', DeltaP2700, 'DeltaC3100 =', DeltaC3100)

#====================================================================
# 2H)
# choosing the best K through simple_sim
def coef_0delta(K_index):
    F0 = 2900
    K = int(df[K_index:K_index+1]['Strike'])
    C0 = float(df[K_index:K_index+1]['Call'])
    P0 = float(df[K_index:K_index+1]['Put'])
    ivP = bk.implied_volatility.implied_volatility_of_discounted_option_price(
    P0, F0, K, r, t, flag='p')
    DeltaP = greek.delta('p', F0, K, t, r, ivP)
    ivC = bk.implied_volatility.implied_volatility_of_discounted_option_price(
    C0, F0, K, r, t, flag='c')
    DeltaC = greek.delta('c', F0, K, t, r, ivC)
    coef = -DeltaC/DeltaP
    # print('Coef for Delta-Neutral: 1 Call and', coef, 'Puts')
    return coef

def sim_FT(v): #FIXME: can try to take into account changing iv as a function of the current FT
    r = 0.0250
    T = 15/252
    F0 = 2900
    FT = F0 * np.exp((r - 0.5 * v**2) * T + v * np.sqrt(T) * np.random.normal(0, 1))
    return FT

def pnl_sim(K_index):
    sim_num = 10**4
    coef = coef_0delta(K_index)
    # F0 = 2900
    K = int(df[K_index:K_index+1]['Strike'])
    C0 = float(df[K_index:K_index+1]['Call'])
    P0 = float(df[K_index:K_index+1]['Put'])

    # if K_index < 20:    sigma = iv[K_index]
    # elif K_index == 20: sigma = iv2900c
    # elif K_index >20:   sigma = iv[K_index-1]
    sigma = iv2900c

    sim_res_FT = np.zeros(sim_num)
    sim_res_CT = np.zeros(sim_num)
    sim_res_PT = np.zeros(sim_num)
    sim_res_PNL = np.zeros(sim_num)
    for i in range(sim_num):
        FT = sim_FT(sigma)
        sim_res_FT[i] = FT
        C1 = sim_res_CT[i] = max(FT - K, 0)
        P1 = sim_res_PT[i] = max(K - FT, 0)
        sim_res_PNL[i] = (C0 + coef*P0)*np.exp(r*t) - C1 - coef*P1 - (1+coef)*0.85
    PNL = np.mean(sim_res_PNL)
    # print('PNL(K={}) ='.format(K), PNL)
    # print('mean FT:', np.mean(sim_res_FT))
    return PNL
all_pnls = [None]*41
for K_index in range(41): all_pnls[K_index] = pnl_sim(K_index)
opt_K_index = np.argmax(all_pnls)
opt_pnl = all_pnls[opt_K_index]
print('The best K={}, PNL={}'.format(df['Strike'][opt_K_index], opt_pnl))

# Once we chose the best K we do this:
F0 = 2900
K_index = opt_K_index
options_in_contract = 100
K = int(df[K_index:K_index+1]['Strike'])
C0 = float(df[K_index:K_index+1]['Call'])
P0 = float(df[K_index:K_index+1]['Put'])

init_mrg_short_put  = P0 + max(0.15*F0 - max(0, F0 - K), 0.10*K) #https://www.interactivebrokers.com/en/index.php?f=26660&hm=us&ex=us&rgt=1&rsk=0&pm=1&rst=101004100808
coef = coef_0delta(K_index)
init_mrg_short_put *= coef
init_mrg_short_call = C0 + max(0.15*F0 - max(0, K - F0), 0.10*F0)
if init_mrg_short_put > init_mrg_short_call:
    margin = init_mrg_short_put + C0
    print(init_mrg_short_put + C0, 'margin =', margin)
elif init_mrg_short_call >= init_mrg_short_put:
    margin = init_mrg_short_call + coef*P0
    print(init_mrg_short_call + coef*P0, 'margin =', margin)

#the cost per contract to enter the position
print(init_mrg_short_put, init_mrg_short_call)
print('Portfolio = short 100 Calls and short {} Puts. '.format(100*round(coef,2)), 'In total we pay =', options_in_contract*(-C0 - coef*P0 + (1+coef)*0.85 + margin))

#==================================================================
# 2I)
def pnl(F, df, coef):
    r = 0.0250
    t = 15/252
    C0 = float(df['Call'])
    P0 = float(df['Put'])
    K = float(df['Strike'])
    C1 = max(F - K, 0)
    P1 = max(K - F, 0)
    # print('C0={}, P0={}, C1={}, P1={}, coef={}, PNL={}'.format(C0,P0,C1,P1,int(coef),round((C0 + coef*P0)*np.exp(r*t) - C1 - coef*P1 - (1+coef)*0.85),0))
    return (C0 + coef*P0)*np.exp(r*t) - C1 - coef*P1 - (1+coef)*0.85
F0 = 2900
K_index = opt_K_index# the strike chosen in (H)
coef = coef_0delta(K_index)
for F in (2500  , 2800  , 2900,  3000,    3300):
    print('PNL for F={}, K={}'.format(F, int(df[K_index:K_index+1]['Strike'])), 'is', round(100*pnl(F, df[K_index:K_index+1], coef), 2))
