from math import exp
import numpy as np
import pylab as plt

fwdNovDec = 0.0300
s0dec = 0.0260
div = 0.018

#=====================================================================
# 3D)
snp_oct1 = 2920 / exp((s0dec-div)*(80/365)) 
spy_oct1 = round(0.1*snp_oct1, 2)
print('SNP500 on Oct1 =', snp_oct1, '\nSPY on Oct1 =', spy_oct1)

#=====================================================================
# 3G)
s0nov = (s0dec * 80 -  fwdNovDec * 35) / 45
assert round(s0nov, 8) == round(0.0228888888888888, 8)
print('The current Nov 15, 2019 spot rate:', s0nov)

#=====================================================================
# 3I)
F_oct1_dec  = snp_oct1 * exp((s0nov-div)*(45/365)) * exp((fwdNovDec-div)*(35/365)) #2920
F_oct1_oct2 = snp_oct1 * exp((s0nov-div)*(1/365))

snpDailyReturn =  1.1**(1/365)-1
snpAnnual = (1+snpDailyReturn)**365
snp_oct2 = snp_oct1 * (1 + snpDailyReturn)
spy_oct2 = spy_oct1 * (1 + snpDailyReturn)
print('SPY on Oct2 =', round(spy_oct2, 2))

F_oct2_dec = snp_oct2 * exp((s0nov-div)*(44/365)) * exp((fwdNovDec - div) * (35/365))
assert 2920.7155838560448==2915.6378688*exp((s0nov-div)*44/365)*exp((fwdNovDec-div)*35/365)
f_oct2 = (F_oct2_dec - F_oct1_dec) * exp((s0nov-div)*(44/365)) * exp((fwdNovDec - div) * (35/365))
# print('price F_oct1_dec=', F_oct1_dec)
# print('price F_oct1_oct2=', F_oct1_oct2)
print('ES Mini price on Oct2 =', F_oct2_dec, '\tRounded to $0.25:',2920.75)
print('ES Mini value on Oct2 =', round(f_oct2,2))
# f_oct2 = (2920.75- F_oct1_dec) * exp((s0nov-div)*(44/365)) * exp((fwdNovDec - div) * (35/365))
# print('value f_oct2 (after rounding ES Minis=',f_oct2)

#=====================================================================
# 3K)
def f1_oct1_scenario1(snp_oct1, s0dec, div, k):
    F0 = 2920
    F1 = k*snp_oct1 * exp((s0dec - div)*49/365)
    f1 = (F1 - F0) / exp(s0dec*49/365)
    f1_direct = (k*2914.8844861447014*exp((0.0260-0.0180)*49/365)-2920)/exp(0.0260*49/365)
    assert f1 == f1_direct
    return f1_direct

def f1_oct1_scenario2(snp_oct1, s0nov, fwdRateNovDec, div, k):
    F0 = 2920
    F1 = k*snp_oct1 * exp((s0nov - div)*14/365)*exp((fwdRateNovDec-div)*35/365)
    f1 = (F1 - F0) / exp(s0nov*14/365) / exp(fwdRateNovDec*35/365)
    f1_direct = (k*2914.8844861447014*exp((0.0228888888888888-0.0180)*14/365)*exp((0.0300-0.0180)*35/365) - 2920)/exp(0.0228888888888888*14/365)/exp(0.0300*35/365)
    assert f1 == f1_direct
    return f1_direct

def f1_nov15_scenario1(S0, s0dec, div, k):
    F0 = S0 * exp((s0dec - div)*35/365)
    F1 = k*S0 * exp((s0dec - div)*5/365)
    f1 = (F1 - F0) / exp(s0dec*5/365)
    f1_direct = S0 * (k*exp((0.0260-0.0180)*5/365)-exp((0.0260-0.0180)*35/365)) / exp(0.0260*5/365)
    assert f1 == f1_direct
    return f1_direct

def f1_nov15_scenario2(S0, fwdRateNovDec, div, k):
    F0 = S0 * exp((fwdRateNovDec - div)*35/365)
    F1 = k*S0 * exp((fwdRateNovDec - div)*5/365)
    f1 = (F1 - F0) / exp(fwdRateNovDec*5/365)
    f1_direct = S0 * (k*exp((0.0300-0.0180)*5/365)-exp((0.0300-0.0180)*35/365)) / exp(0.0300*5/365)
    assert f1 == f1_direct
    return f1_direct

S0 = 1
plt.figure('Oct1_Scenario1')							#name of window;w/o this everything is plotted in 1 window
plt.xlim(0.5,1.5)
plt.clf()								#clear the frame
plt.grid()
plt.rcParams['lines.linewidth']=1					#line width
plt.rcParams['xtick.major.size']=7					#size of ticks on x-axis
plt.rcParams['ytick.major.size']=7					#size of ticks on y-axis
plt.rcParams['lines.markersize']=10					#size of markers
plt.rcParams['legend.numpoints']=1					#number of examples shown in legends
plt.rcParams['xtick.labelsize']=16					#size of numbers on x-axis
plt.rcParams['ytick.labelsize']=16					#size of numbers on y-axis

X = np.arange(0, 10, 0.01)
f = lambda k: f1_oct1_scenario1(snp_oct1, s0dec, div, k)
Y_f1_oct1_scenario1 = np.array(list(map(f, X)))

f = lambda k: f1_oct1_scenario2(snp_oct1, s0nov, fwdNovDec, div, k)
Y_f1_oct1_scenario2 = np.array(list(map(f, X)))

f = lambda k: f1_nov15_scenario1(S0, s0dec, div, k)
Y_f1_nov15_scenario1 = np.array(list(map(f, X)))

f = lambda k: f1_nov15_scenario2(S0, fwdNovDec, div, k)
Y_f1_nov15_scenario2 = np.array(list(map(f, X)))

plt.xlabel('x shows how much the price is going to change')
plt.ylabel('f1 - the value of the forward position on Nov 1')
plt.plot(Y_f1_oct1_scenario1, label='Enter on Oct1, Scenario 1')
plt.plot(Y_f1_oct1_scenario2, label='Enter on Oct1, Scenario 2')
plt.plot(Y_f1_nov15_scenario1, label='Enter on Nov15, Scenario 1')
plt.plot(Y_f1_nov15_scenario2, label='Enter on Nov15, Scenario 2')
plt.legend()

#====================================================================
# (K) Result:
print('f1_oct1_scenario1 < f1_oct1_scenario2:',
list(Y_f1_oct1_scenario1 < Y_f1_oct1_scenario2) == [True]*len(Y_f1_oct1_scenario1))
print('f1_nov15_scenario1 > f1_nov15_scenario2:',
list(Y_f1_nov15_scenario1 > Y_f1_nov15_scenario2) == [True]*len(Y_f1_oct1_scenario1))
