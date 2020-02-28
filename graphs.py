import pandas as pd
from time import strftime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


matplotlib.rcParams.update({'font.size': 12,
                            'font.weight': 'light'})
fontsize = 13

number_ewh = 50
n_set = 84

sensitivity = pd.read_csv('sensitivity_%d.csv' % n_set)
inputs = pd.read_csv("inputs1_%d.csv" % number_ewh)
outcomes = pd.read_csv('outcomes1_%d.csv' % number_ewh)

total_flex = np.zeros((2688, 14))
for i in range(1, 15):
    total_flex[:, i-1] = inputs['LV'].values + inputs['MV'].values + inputs['EWH flex %d' % (i*10*5)].values

total_noflex = inputs['LV'].values + inputs['MV'].values + inputs['EWH'].values
total_flex420 = inputs['LV'].values + inputs['MV'].values + inputs['EWH flex'].values
ewh_consumption = 4.2 * .25


def sun_surplus(demand, pv_production, n_set):  # function to get the value of ewh
    # that can be shift in a specific time frame
    surplus = int((n_set * pv_production - demand) / ewh_consumption)
    if surplus < 0:
        surplus = 0
    return surplus


v_sun_surplus = np.vectorize(sun_surplus)
surplus100 = v_sun_surplus(total_noflex, inputs['PV'], 20)
surplus150 = v_sun_surplus(total_noflex, inputs['PV'], 30)
surplus250 = v_sun_surplus(total_noflex, inputs['PV'], 50)
surplus350 = v_sun_surplus(total_noflex, inputs['PV'], 70)
surplus420 = v_sun_surplus(total_noflex, inputs['PV'], 84)

total_surplus = np.zeros(14)
total_surplus_flex = np.zeros(14)
for i in range(1, 15):
    total_surplus[i-1] = v_sun_surplus(total_noflex, inputs['PV'], i*10).sum()
    total_surplus_flex[i-1] = v_sun_surplus(total_flex[:, (i-1)], inputs['PV'], i*10).sum()
    print(total_surplus[i-1])
    print(' ---- %d' % (i-1))

print('LV stats')
print((inputs['LV']/.25).describe())
print('MV stats')
print((inputs['MV']/.25).describe())
print('Community stats')
print(pd.Series((inputs['MV'].values + inputs['LV'].values + inputs['EWH'].values)/.25).describe())

#  Community Parameter
total_consumption = inputs['MV'].values.sum() + inputs['LV'].values.sum() + inputs['EWH'].values.sum()
community_average = (inputs['MV'].values + inputs['LV'].values + inputs['EWH'].values).mean() / .25
community_peak = (inputs['MV'].values + inputs['LV'].values + inputs['EWH'].values).max() / .25
peak_location = (inputs['MV'].values + inputs['LV'].values + inputs['EWH'].values).argmax()
community_delta = (community_peak - community_average) / community_peak * 100
communityDR_average = (inputs['MV'].values + inputs['LV'].values + inputs['EWH flex'].values).mean() / .25
communityDR_peak = (inputs['MV'].values + inputs['LV'].values + inputs['EWH flex'].values).max() / .25
peak_locationDR = (inputs['MV'].values + inputs['LV'].values + inputs['EWH flex'].values).argmax()
communityDR_delta = (communityDR_peak - communityDR_average) / communityDR_peak * 100

#  MV parameters
MV_average = inputs['MV'].values.mean() / .25
MV_peak = inputs['MV'].values.max() / .25
MV_share = inputs['MV'].values.sum() / total_consumption
MV_delta = (MV_peak - MV_average) / MV_peak * 100
MV_total_consumption = inputs['MV'].values.sum()

# LV parameters
LV_average = inputs['LV'].values.mean() / .25
LV_peak = inputs['LV'].values.max() / .25
LV_share = inputs['LV'].values.sum() / total_consumption
LV_delta = (LV_peak - LV_average) / LV_peak * 100
LV_total_consumption = inputs['LV'].values.sum()

# EWH parameters
EWH_average = inputs['EWH'].values.mean() / .25
EWH_peak = inputs['EWH'].values.max() / .25
EWH_share = inputs['EWH'].values.sum() / total_consumption
EWH_delta = (EWH_peak - EWH_average) / EWH_peak
EWH_total_consumption = inputs['EWH'].values.sum()

print('community net peak %d' % (inputs['LV'].values + inputs['MV'].values + inputs['EWH flex'].values - inputs['PV'].values*n_set).min())
print('EWH av %f' % EWH_average)
print('EWH_peak %f' % EWH_peak)
print('EWH delta %f' % EWH_delta)
print('LV av %f' % LV_average)
print('LV peak %f' % LV_peak)
print('LV delta %f' % LV_delta)
print('MV average %f' % MV_average)
print('MV peak %f' % MV_peak)
print('MV delta %f' % MV_delta)

print('community peak %f' % community_peak)
print('community delta %f' % community_delta)
print('peak location %f' % peak_location)

print('communityDR peak %f' % communityDR_peak)
print('communityDR delta %f' % communityDR_delta)
print('peak location DR %f' % peak_locationDR)

print('-----------------------')
print('cost without DR %f' % outcomes['cost2'].min())
print('cost with DR %f' % outcomes['cost3'][outcomes['cost2'].idxmin()])
print('Optimal capacity without %f' % outcomes['cost2'].idxmin())
print('Optimal capacity with %f' % outcomes['cost3'].idxmin())

optimal = outcomes['cost2'].idxmin()
print('------------------------')
print('average delta SSR ewh %f' % np.average(outcomes['SSR3 ewh'].values[70:] - outcomes['SSR2 ewh'].values[70:]))
print('average delta SSR %f' % np.average(outcomes['SSR3'].values[70:] - outcomes['SSR2'].values[70:]))
print('delta SSR ewh in optimum postion %f' % (outcomes['SSR3 ewh'].values[optimal] - outcomes['SSR2 ewh'].values[optimal]))
print('SSR ewh without DR %f' % outcomes['SSR2 ewh'].values[optimal])
print('delta SSR in optimum point %f' % np.average(outcomes['SSR3'].values[optimal] - outcomes['SSR2'].values[optimal]))
print('average delta SCR %f' % np.average(outcomes['SCR3'].values[30:] - outcomes['SCR2'].values[30:]))
print('delta SCR  in optimum point %f' % np.average(outcomes['SCR3'].values[optimal] - outcomes['SCR2'].values[optimal]))

print('SSR without %f' % outcomes['SSR2'].values[optimal])
print('SSR with %f' % outcomes['SSR3'].values[optimal])

print('SCR without %f' % outcomes['SCR2'].values[optimal])
print('SCR with %f' % outcomes['SCR3'].values[optimal])

print('************************')
print('average delta costs %f' % np.average(outcomes['cost2'].values[70:] - outcomes['cost3'].values[70:]))

print('---------------------')
print('ratio between PV production and community consumption: %f' % (inputs['PV'].values.sum() * n_set /
                                                                     (MV_total_consumption +
                                                                      LV_total_consumption +
                                                                      EWH_total_consumption)))

'''''
CHARTS 
'''''
x_EWH = range(10, 201, 10)

x_PV = range(0, 150*5, 5)

# Net community load
day = 23
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot((inputs['MV'].values[day*96:(day*96+96)] +
         inputs['LV'].values[day*96:(day*96+96)] +
         inputs['EWH'].values[day*96:(day*96+96)] -
         inputs['PV'].values[day*96:(day*96+96)] * n_set)/.25, label='Scenario without DR')
ax1.plot((inputs['MV'].values[day*96:(day*96+96)] +
         inputs['LV'].values[day*96:(day*96+96)] +
         inputs['EWH flex'].values[day*96:(day*96+96)] -
         inputs['PV'].values[day*96:(day*96+96)] * n_set)/.25, '--', label='Scenario with DR')

ax1.set_xlim(xmin=0)
ax1.set_ylim(ymin=-300, ymax=+250)
ax1.axhline(linewidth=0.5, color='black')
ax1.legend(loc='lower left')
ax1.grid(which='both')

day = 27
fig2, ax2 = plt.subplots()
ax2.set_xlabel('Time of day %d' % day, fontsize=fontsize)
ax2.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax2.plot((inputs['MV'].values[day*96:(day*96+96)] +
         inputs['LV'].values[day*96:(day*96+96)] +
         inputs['EWH'].values[day*96:(day*96+96)] -
         inputs['PV'].values[day*96:(day*96+96)] * n_set)/.25, label='Scenario without DR')
ax2.plot((inputs['MV'].values[day*96:(day*96+96)] +
         inputs['LV'].values[day*96:(day*96+96)] +
         inputs['EWH flex'].values[day*96:(day*96+96)] -
         inputs['PV'].values[day*96:(day*96+96)] * n_set)/.25, '--', label='Scenario with DR')

ax2.set_xlim(xmin=0, xmax=95)
ax2.set_ylim(ymin=-300, ymax=250)
ax2.axhline(linewidth=0.5, color='black')
ax2.legend()
ax2.grid(which='major', axis='both')

plt.show()

# Sensitivity SCR and SSR
fig1, ax1 = plt.subplots()
ax1.plot(x_EWH, sensitivity['SCR2'], label='Scenario without DR')
ax1.plot(x_EWH, sensitivity['SCR3'], '--', label='Scenario with DR')
ax1.set_xlabel('Number of EWHs Installed', fontsize=fontsize)
ax1.set_ylabel('Self-Consumption Rate [%]', fontsize=fontsize)
ax1.set_xlim(xmin=0, xmax=200)
ax1.set_ylim(ymin=0, ymax=100)
ax1.grid()
ax1.legend()

fig2, ax2 = plt.subplots()
ax2.plot(x_EWH, sensitivity['SSR2'], label='Scenario without DR')
ax2.plot(x_EWH, sensitivity['SSR3'], '--', label='Scenario with DR')
ax2.set_xlabel('Number of EWHs Installed', fontsize=fontsize)
ax2.set_ylabel('Self-Sufficiency Rate [%]', fontsize=fontsize)
ax2.set_xlim(xmin=0, xmax=200)
ax2.set_ylim(ymin=0, ymax=100)
ax2.grid()
ax2.legend()
plt.show()

# Sensitivity cost
fig1, ax1 = plt.subplots()
ax1.plot(x_EWH, sensitivity['cost2'], label='Scenario without DR')
ax1.plot(x_EWH, sensitivity['cost3'], '--', label='Scenario with DR')
ax1.set_xlabel('Number of EWHs Installed', fontsize=fontsize)
ax1.set_ylabel('Electricity Cost [€/year]', fontsize=fontsize)
ax1.set_xlim(xmin=0, xmax=200)
ax1.set_ylim(ymin=0)
ax1.grid()
ax1.legend()

fig2, ax2 = plt.subplots()
ax2.plot(x_EWH, sensitivity['cost2'], label='Scenario without DR')
ax2.plot(x_EWH, sensitivity['cost3'], '--', label='Scenario with DR')
ax2.set_xlabel('Number of EWHs Installed', fontsize=fontsize)
ax2.set_ylabel('Electricity Cost [€/year]', fontsize=fontsize)
ax2.set_xlim(xmin=0, xmax=200)
ax2.set_ylim(ymin=7000)
ax2.grid()
ax2.legend()

plt.show()

# Surplus difference
fig, ax = plt.subplots()
ax.plot(range(50, 750, 50), total_surplus, label='Total surplus without DR')
ax.plot(range(50, 750, 50), total_surplus_flex, '--', label='Total surplus with DR')
ax.set_xlabel('PV Capacity Installed [kW]', fontsize=fontsize)
ax.set_ylabel('Number of surplus', fontsize=fontsize)
ax.set_xlim(xmin=0, xmax=700)
ax.set_ylim(ymin=0, ymax=50000)
ax.legend(loc='upper left')

fig1, ax1 = plt.subplots()
ax1.plot(range(50, 750, 50), (total_surplus - total_surplus_flex), 'k', label='Difference')
ax1.set_ylim(ymin=0, ymax=2500)
ax1.set_xlim(xmin=0, xmax=700)
ax1.legend(loc='upper left')

ax1.grid(which='both')
ax.grid(which='both')
plt.show()

# EWH changes vs surplus
day = 14
n_day = 2
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time frames from day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot(surplus150[day*96:(day*96+96*n_day)], 'y', label='Sun surplus')
ax1.plot((inputs['EWH'].values[day*96:(day*96+96*n_day)]) / ewh_consumption, 'k', label='Scenario without DR')
ax1.plot((inputs['EWH flex 150'].values[day*96:(day*96+96*n_day)]) / ewh_consumption,
         '--', label='Scenario with DR, 150 kW')

ax1.legend(loc='upper left')
fig1.tight_layout()

ax1.set_ylim(ymin=0)
ax1.set_xlim(xmin=0, xmax=96*n_day - 1)
ax1.grid(which='both')

fig2, ax1 = plt.subplots()
ax1.set_xlabel('Time frames from day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot(surplus250[day*96:(day*96+96*n_day)], 'y', label='Sun surplus')
ax1.plot((inputs['EWH'].values[day*96:(day*96+96*n_day)]) / ewh_consumption, 'k', label='Scenario without DR')
ax1.plot((inputs['EWH flex 250'].values[day*96:(day*96+96*n_day)]) / ewh_consumption,
         '--', label='Scenario with DR, 250 kW')

ax1.legend(loc='upper left')
fig2.tight_layout()

ax1.set_ylim(ymin=0)
ax1.set_xlim(xmin=0, xmax=96*n_day - 1)
ax1.grid(which='both')

fig3, ax1 = plt.subplots()
ax1.set_xlabel('Time frames from day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot(surplus350[day*96:(day*96+96*n_day)], 'y', label='Sun surplus')
ax1.plot((inputs['EWH'].values[day*96:(day*96+96*n_day)]) / ewh_consumption, 'k', label='Scenario without DR')
ax1.plot((inputs['EWH flex 350'].values[day*96:(day*96+96*n_day)]) / ewh_consumption,
         '--', label='Scenario with DR, 350 kW')

ax1.legend(loc='upper left')
fig3.tight_layout()

ax1.set_ylim(ymin=0)
ax1.set_xlim(xmin=0, xmax=96*n_day - 1)
ax1.grid(which='both')

fig4, ax1 = plt.subplots()
ax1.set_xlabel('Time frames from day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot(surplus420[day*96:(day*96+96*n_day)], 'y', label='Sun surplus')
ax1.plot((inputs['EWH'].values[day*96:(day*96+96*n_day)]) / ewh_consumption, 'k', label='Scenario without DR')
ax1.plot((inputs['EWH flex'].values[day*96:(day*96+96*n_day)]) / ewh_consumption,
         '--', label='Scenario with DR, 420 kW')

ax1.legend(loc='upper left')
fig4.tight_layout()

ax1.set_ylim(ymin=0)
ax1.set_xlim(xmin=0, xmax=96*n_day - 1)
ax1.grid(which='both')

fig5, ax1 = plt.subplots()
ax1.set_xlabel('Time frames from day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot(surplus100[day*96:(day*96+96*n_day)], 'y', label='Sun surplus')
ax1.plot((inputs['EWH'].values[day*96:(day*96+96*n_day)]) / ewh_consumption, 'k', label='Scenario without DR')
ax1.plot((inputs['EWH flex 100'].values[day*96:(day*96+96*n_day)]) / ewh_consumption,
         '--', label='Scenario with DR, 100 kW')

ax1.legend(loc='upper left')
fig5.tight_layout()

ax1.set_ylim(ymin=0)
ax1.set_xlim(xmin=0, xmax=96*n_day - 1)
ax1.grid(which='both')
plt.show()

# EWH Changes
day = 3
n_day = 2
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time frames from day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot((inputs['EWH'].values[day*96:(day*96+96*n_day)]) / ewh_consumption, 'k', label='Scenario without DR')
ax1.plot((inputs['EWH flex 150'].values[day*96:(day*96+96*n_day)]) / ewh_consumption, '--', label='Scenario with DR, 150 kW')
ax1.plot((inputs['EWH flex 250'].values[day*96:(day*96+96*n_day)]) / ewh_consumption, '--', label='Scenario with DR, 250 kW')
ax1.plot((inputs['EWH flex 350'].values[day*96:(day*96+96*n_day)]) / ewh_consumption, '--', label='Scenario with DR, 350 kW')

ax1.plot(np.nan, 'y', label='PV production')

ax2 = ax1.twinx()
ax2.set_ylabel('Normalized PV Production', fontsize=fontsize)
ax2.plot(inputs['PV'].values[day*96:(day*96+96*n_day)] / inputs['PV'].values[day*96:(day*96+96*n_day)].max(),
         'y', label='PV production')

ax1.legend(loc='upper left')
fig1.tight_layout()

ax1.set_ylim(ymin=0, ymax=100)
ax2.set_ylim(ymin=0, ymax=1)
ax1.set_xlim(xmin=0, xmax=96*n_day - 1)
ax2.set_xlim(xmin=0, xmax=96*n_day - 1)
ax1.grid(which='both')

plt.show()

# SSR & SCR
fig1, ax1 = plt.subplots()
ax1.set_xlabel('PV Capacity Installed [kW]', fontsize=fontsize)
ax1.set_ylabel('Self-Sufficiency Rate [%]', fontsize=fontsize)

ax1.plot(x_PV, outcomes['SSR2'].values, label='Scenario without DR')
ax1.plot(x_PV, outcomes['SSR3'].values, '--', label='Scenario with DR')
ax1.set_ylim(ymin=0, ymax=100)
ax1.set_xlim(xmin=0, xmax=745)

ax1.grid(which='both')
ax1.legend(loc='center left')

fig2, ax1 = plt.subplots()
ax1.set_xlabel('PV Capacity Installed [kW]', fontsize=fontsize)
ax1.set_ylabel('Self-Consumption Rate [%]', fontsize=fontsize)

ax1.plot(x_PV, outcomes['SCR2'].values, label='Scenario without DR')
ax1.plot(x_PV, outcomes['SCR3'].values, '--', label='Scenario with DR')
ax1.set_ylim(ymin=0, ymax=100)
ax1.set_xlim(xmin=0, xmax=745)

ax1.grid(which='both')
ax1.legend(loc='center left')
ax1.spines['top'].set_color('none')

fig3, ax1 = plt.subplots()
ax1.set_xlabel('PV Capacity Installed [kW]', fontsize=fontsize)
ax1.set_ylabel('Self-Sufficiency Rate difference [%]', fontsize=fontsize)
ax1.plot(x_PV, outcomes['SSR3'].values - outcomes['SSR2'].values, 'k')
ax1.set_ylim(ymin=0, ymax=10)
ax1.set_xlim(xmin=0, xmax=745)
ax1.grid(which='both')

fig4, ax1 = plt.subplots()
ax1.set_xlabel('PV Capacity Installed [kW]', fontsize=fontsize)
ax1.set_ylabel('Self-Consumption Rate difference [%]', fontsize=fontsize)
ax1.plot(x_PV, outcomes['SCR3'].values - outcomes['SCR2'].values, 'k')
ax1.set_ylim(ymin=0, ymax=10)
ax1.set_xlim(xmin=0, xmax=745)
ax1.grid(which='both')

plt.show()

# Community load
day = 23
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time frames of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot((inputs['MV'].values[day*96:(day*96+96)] +
         inputs['LV'].values[day*96:(day*96+96)] +
         inputs['EWH'].values[day*96:(day*96+96)])/.25, label='Scenario without DR')
ax1.plot((inputs['MV'].values[day*96:(day*96+96)] +
         inputs['LV'].values[day*96:(day*96+96)] +
         inputs['EWH flex'].values[day*96:(day*96+96)])/.25, '--', label='Scenario with DR')
ax1.plot(np.nan, 'y', label='PV production')
ax1.legend(loc='lower left', bbox_to_anchor=(-0.01, 0.97))
ax1.set_xlim(xmin=0, xmax=95)
ax1.set_ylim(ymin=0, ymax=250)

ax2 = ax1.twinx()
ax2.set_ylabel('Normalized PV Production', fontsize=fontsize)
ax2.plot(inputs['PV'].values[day*96:(day*96+96)] / inputs['PV'].values[day*96:(day*96+96)].max(),
         'y', label='PV production')

ax2.set_ylim(ymin=0, ymax=1)
fig1.tight_layout()
ax1.grid(which='both')

day = 27
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time frames of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot((inputs['MV'].values[day*96:(day*96+96)] +
         inputs['LV'].values[day*96:(day*96+96)] +
         inputs['EWH'].values[day*96:(day*96+96)])/.25, label='Scenario without DR')
ax1.plot((inputs['MV'].values[day*96:(day*96+96)] +
         inputs['LV'].values[day*96:(day*96+96)] +
         inputs['EWH flex'].values[day*96:(day*96+96)])/.25, '--', label='Scenario with DR')
ax1.plot(np.nan, 'y', label='PV production')
ax1.legend(loc='upper left', bbox_to_anchor=(-0.01, 0.97))
ax1.set_xlim(xmin=0, xmax=95)
ax1.set_ylim(ymin=0, ymax=250)

ax2 = ax1.twinx()
ax2.set_ylabel('Normalized PV Production', fontsize=fontsize)
ax2.plot(inputs['PV'].values[day*96:(day*96+96)] / inputs['PV'].values[day*96:(day*96+96)].max(),
         'y', label='PV production')

ax2.set_ylim(ymin=0, ymax=1)
fig1.tight_layout()
ax1.grid(which='both')

plt.show()

# SSR only EWH
fig1, ax1 = plt.subplots()
ax1.set_xlabel('PV Capacity Installed [kW]', fontsize=fontsize)
ax1.set_ylabel('Self-Sufficiency Rate for EWH [%]', fontsize=fontsize)

ax1.plot(x_PV, outcomes['SSR2 ewh'].values, label='Scenario without DR')
ax1.plot(x_PV, outcomes['SSR3 ewh'].values, '--', label='Scenario with DR')
ax1.set_ylim(ymin=0, ymax=100)
ax1.set_xlim(xmin=0, xmax=745)

ax1.grid(which='both')
ax1.legend(loc='upper left')
plt.show()

# SSR only EWH with gap between scenarios
fig1, ax1 = plt.subplots()
ax1.set_xlabel('PV Capacity Installed [kW]', fontsize=fontsize)
ax1.set_ylabel('Self-Sufficiency Rate for EWH [%]', fontsize=fontsize)

ax1.plot(x_PV, outcomes['SSR2 ewh'].values, label='Scenario without DR')
ax1.plot(x_PV, outcomes['SSR3 ewh'].values, '--', label='Scenario with DR')
ax1.plot(x_PV, outcomes['SSR3 ewh'].values - outcomes['SSR2 ewh'].values, color='magenta', label='Gap between Scenarios')
ax1.set_ylim(ymin=0, ymax=100)
ax1.set_xlim(xmin=0, xmax=745)

ax1.grid(which='both')
ax1.legend(loc='upper left')
plt.show()

# MV, LV, PV profiles of a set holiday
day = 13
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time frames of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot((inputs['MV'].values[day*96:(day*96+96)] +
          inputs['LV'].values[day*96:(day*96+96)] +
          inputs['EWH'].values[day*96:(day*96+96)]) / .25, 'k', label='Community Demand')
ax1.plot((inputs['MV'].values[day*96:(day*96+96)] +
          inputs['LV'].values[day*96:(day*96+96)] +
          inputs['EWH flex'].values[day*96:(day*96+96)]) / .25, '--', color='darkorange', label='Community Demand with DR')

ax1.plot(np.nan, 'y', label='PV production')

ax2 = ax1.twinx()
ax2.set_ylabel('Normalized PV Production', fontsize=fontsize)
ax2.plot(inputs['PV'].values[day*96:(day*96+96)] / inputs['PV'].values[day*96:(day*96+96)].max(),
         'y', label='PV production')

ax1.legend(loc='upper left', bbox_to_anchor=(-0.01, 0.97))
fig1.tight_layout()

ax1.set_ylim(ymin=0, ymax=310)
ax2.set_ylim(ymin=0, ymax=1)
ax1.set_xlim(xmin=0, xmax=95)
ax2.set_xlim(xmin=0, xmax=95)
ax1.grid(which='both')

plt.show()

# delta SSR and SCR
fig, ax = plt.subplots()
ax.plot(x_PV, outcomes['SSR3'].values - outcomes['SSR2'].values, label='SSR')
ax.plot(x_PV, outcomes['SCR3'].values - outcomes['SCR2'].values, label='SCR')
ax.set_xlabel('PV Capacity Installed [kW]', fontsize=fontsize)
ax.set_ylabel('Difference [%]', fontsize=fontsize)
ax.set_xlim(xmin=0, xmax=745)
ax.set_ylim(ymin=0, ymax=5)
ax.grid()
ax.legend()

plt.show()

# MV, LV, PV profiles of a set weekday
day = 10
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time frames of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Power [kW]', fontsize=fontsize)

ax1.plot((inputs['MV'].values[day*96:(day*96+96)] +
          inputs['LV'].values[day*96:(day*96+96)] +
          inputs['EWH'].values[day*96:(day*96+96)]) / .25, 'k', label='Community Demand')
ax1.plot((inputs['MV'].values[day*96:(day*96+96)] +
          inputs['LV'].values[day*96:(day*96+96)] +
          inputs['EWH flex'].values[day*96:(day*96+96)]) / .25, '--', color='darkorange', label='Community Demand with DR')

ax1.plot(inputs['PV'].values[day*96:(day*96+96)] / .25, 'y', label='PV production')

ax1.legend(loc='upper left', bbox_to_anchor=(-0.01, 0.97))
fig1.tight_layout()

ax1.set_ylim(ymin=0)
ax1.set_xlim(xmin=0, xmax=95)
ax1.grid(which='both')

# MV, LV, PV profiles of a set holiday
day = 13
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time frames of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot((inputs['MV'].values[day*96:(day*96+96)] +
          inputs['LV'].values[day*96:(day*96+96)] +
          inputs['EWH'].values[day*96:(day*96+96)]) / .25, 'k', label='Community Demand')
ax1.plot((inputs['MV'].values[day*96:(day*96+96)] +
          inputs['LV'].values[day*96:(day*96+96)] +
          inputs['EWH flex'].values[day*96:(day*96+96)]) / .25, '--', color='darkorange', label='Community Demand with DR')

ax1.plot(np.nan, 'y', label='PV production')

ax2 = ax1.twinx()
ax2.set_ylabel('Normalized PV Production', fontsize=fontsize)
ax2.plot(inputs['PV'].values[day*96:(day*96+96)] / inputs['PV'].values[day*96:(day*96+96)].max(),
         'y', label='PV production')

ax1.legend(loc='upper left', bbox_to_anchor=(-0.01, 0.97))
fig1.tight_layout()

ax1.set_ylim(ymin=0, ymax=310)
ax2.set_ylim(ymin=0, ymax=1)
ax1.set_xlim(xmin=0, xmax=95)
ax2.set_xlim(xmin=0, xmax=95)
ax1.grid(which='both')

plt.show()

# Pie Chart consumption share
pie_labels = 'MV profiles', 'LV profiles', 'EWH profiles'
consumption = [MV_total_consumption, LV_total_consumption, EWH_total_consumption]
explode = (0.0, 0.0, 0.10)

fig1, ax1 = plt.subplots()
ax1.pie(consumption, labels=pie_labels, autopct='%1.1f%%', explode=explode, startangle=90, shadow=False)
ax1.axis('equal')
ax1.legend(pie_labels, loc='upper left', bbox_to_anchor=(-0.12, 1.12))

plt.show()

# Cost graph - total
fig, ax1 = plt.subplots()
ax1.set_xlabel('PV Capacity Installed [kW]', fontsize=fontsize)
ax1.set_ylabel('Electricity Cost [€/year]', fontsize=fontsize)
ax1.plot(x_PV, outcomes['cost2'].values, label='Scenario without DR')
ax1.plot(x_PV, outcomes['cost3'].values, '--', label='Scenario with DR')

ax1.grid(which='both')
ax1.set_ylim(ymin=0, ymax=10000)
ax1.set_xlim(xmin=0, xmax=750)
ax1.legend(loc='center left')
fig.tight_layout()
plt.show()

# SSR & SCR
fig1, ax1 = plt.subplots()
ax1.set_xlabel('PV Capacity Installed [kW]', fontsize=fontsize)
ax1.set_ylabel('Self-Sufficiency Rate [%]', fontsize=fontsize)

ax1.plot(x_PV, outcomes['SSR2'].values, label='Scenario without DR')
ax1.plot(x_PV, outcomes['SSR3'].values, '--', label='Scenario with DR')
ax1.set_ylim(ymin=0, ymax=100)
ax1.set_xlim(xmin=0, xmax=745)

ax1.grid(which='both')
ax1.legend(loc='center left')

fig2, ax1 = plt.subplots()
ax1.set_xlabel('PV Capacity Installed [kW]', fontsize=fontsize)
ax1.set_ylabel('Self-Consumption Rate [%]', fontsize=fontsize)

ax1.plot(x_PV, outcomes['SCR2'].values, label='Scenario without DR')
ax1.plot(x_PV, outcomes['SCR3'].values, '--', label='Scenario with DR')
ax1.set_ylim(ymin=0, ymax=100)
ax1.set_xlim(xmin=0, xmax=745)

ax1.grid(which='both')
ax1.legend(loc='center left')
ax1.spines['top'].set_color('none')
plt.show()

# Cost - zoom
fig2, ax2 = plt.subplots()
ax2.set_ylabel('Electricity Cost [€/year]', fontsize=fontsize)
ax2.set_xlabel('PV Capacity Installed [kW]', fontsize=fontsize)
ax2.plot(x_PV, outcomes['cost2'].values, label='Scenario without DR')
ax2.plot(x_PV, outcomes['cost3'].values, '--', label='Scenario with DR')

ax2.grid(which='both')
ax2.set_ylim(ymin=7500, ymax=10000)
ax2.set_xlim(xmin=0, xmax=745)

ax2.legend(loc='upper right')
fig2.tight_layout()
plt.show()

# MV, LV, PV profiles of a set weekday
day = 3
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time frames of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot(inputs['MV'].values[day*96:(day*96+96)] / .25, label='MV profiles')
ax1.plot(inputs['LV'].values[day*96:(day*96+96)] / .25, label='LV profiles')
ax1.plot(inputs['EWH'].values[day*96:(day*96+96)] / .25, label='EWH profiles')
ax1.plot(np.nan, 'y', label='PV production')

ax2 = ax1.twinx()
ax2.set_ylabel('Normalized PV Production', fontsize=fontsize)
ax2.plot(inputs['PV'].values[day*96:(day*96+96)] / inputs['PV'].values[day*96:(day*96+96)].max(),
         'y', label='PV production')

ax1.legend(loc='upper left', bbox_to_anchor=(-0.01, 0.97))
fig1.tight_layout()

ax1.set_ylim(ymin=0, ymax=130)
ax2.set_ylim(ymin=0, ymax=1)
ax1.set_xlim(xmin=0, xmax=95)
ax2.set_xlim(xmin=0, xmax=95)
ax1.grid(which='both')

# MV, LV, PV profiles of a set holiday
day = 6
fig2, ax1 = plt.subplots()
ax1.set_xlabel('Time frames of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot(inputs['MV'].values[day*96:(day*96+96)] / .25, label='MV profiles')
ax1.plot(inputs['LV'].values[day*96:(day*96+96)] / .25, label='LV profiles')
ax1.plot(inputs['EWH'].values[day*96:(day*96+96)] / .25, label='EWH profiles')
ax1.plot(np.nan, 'y', label='PV production')

ax2 = ax1.twinx()
ax2.set_ylabel('Normalized PV Production', fontsize=fontsize)
ax2.plot(inputs['PV'].values[day*96:(day*96+96)] / inputs['PV'].values[day*96:(day*96+96)].max(),
         'y', label='PV production')

ax1.legend(loc='upper left', bbox_to_anchor=(-0.01, 0.97))
fig2.tight_layout()
ax1.set_ylim(ymin=0, ymax=130)
ax2.set_ylim(ymin=0, ymax=1)
ax1.set_xlim(xmin=0, xmax=95)
ax2.set_xlim(xmin=0, xmax=95)
ax1.grid(which='both')

plt.show()


