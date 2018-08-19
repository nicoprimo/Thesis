import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


matplotlib.rcParams.update({'font.size': 12,
                            'font.weight': 'light'})
fontsize = 13

number_ewh = 50

inputs = pd.read_csv("inputs_%d.csv" % number_ewh)
outcomes = pd.read_csv('outcomes_%d.csv' % number_ewh)
inputs['MV'] = np.zeros(inputs['MV'].shape)

print('LV stats')
print((inputs['LV']/.25).describe())
print('MV stats')
print((inputs['MV']/.25).describe())
print('Community stats')
print(pd.Series((inputs['MV'].values + inputs['LV'].values + inputs['EWH'].values)/.25).describe())

#  Community Parameter
total_consumption = inputs['LV'].values.sum() + inputs['EWH'].values.sum()
community_average = (inputs['LV'].values + inputs['EWH'].values).mean() / .25
community_peak = (inputs['LV'].values + inputs['EWH'].values).max() / .25
community_delta = (community_peak - community_average) / community_peak * 100
communityDR_average = (inputs['LV'].values + inputs['EWH flex'].values).mean() / .25
communityDR_peak = (inputs['LV'].values + inputs['EWH flex'].values).max() / .25
communityDR_delta = (communityDR_peak - communityDR_average) / communityDR_peak * 100


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


print('EWH av %f' % EWH_average)
print('EWH_peak %f' % EWH_peak)
print('EWH delta %f' % EWH_delta)
print('LV av %f' % LV_average)
print('LV peak %f' % LV_peak)
print('LV delta %f' % LV_delta)
print('community average %f' % community_average)
print('community peak %f' % community_peak)
print('community delta %f' % community_delta)
print('communityDR average %f' % communityDR_average)
print('communityDR peak %f' % communityDR_peak)
print('communityDR delta %f' % communityDR_delta)

'''''
CHARTS 
'''''
x_PV = range(0, 150*5, 5)

# MV, LV, PV profiles of a set weekday
day = 10
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time frames of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot((inputs['LV'].values[day*96:(day*96+96)] +
          inputs['EWH'].values[day*96:(day*96+96)]) / .25, color='green', label='Community Demand')

ax1.plot(np.nan, 'y', label='PV production')

ax2 = ax1.twinx()
ax2.set_ylabel('Normalized PV Production', fontsize=fontsize)
ax2.plot(inputs['PV'].values[day*96:(day*96+96)] / inputs['PV'].values[day*96:(day*96+96)].max(),
         'y', label='PV production')

ax1.legend(loc='upper left', bbox_to_anchor=(-0.01, 0.97))
fig1.tight_layout()

ax1.set_ylim(ymin=0)
ax2.set_ylim(ymin=0, ymax=1)
ax1.set_xlim(xmin=0, xmax=95)
ax2.set_xlim(xmin=0, xmax=95)
ax1.grid(which='both')

# MV, LV, PV profiles of a set holiday
day = 13
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time frames of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot((inputs['LV'].values[day*96:(day*96+96)] +
          inputs['EWH'].values[day*96:(day*96+96)]) / .25, color='orange', label='Community Demand')

ax1.plot(np.nan, 'y', label='PV production')

ax2 = ax1.twinx()
ax2.set_ylabel('Normalized PV Production', fontsize=fontsize)
ax2.plot(inputs['PV'].values[day*96:(day*96+96)] / inputs['PV'].values[day*96:(day*96+96)].max(),
         'y', label='PV production')

ax1.legend(loc='upper left', bbox_to_anchor=(-0.01, 0.97))
fig1.tight_layout()

ax1.set_ylim(ymin=0)
ax2.set_ylim(ymin=0, ymax=1)
ax1.set_xlim(xmin=0, xmax=95)
ax2.set_xlim(xmin=0, xmax=95)
ax1.grid(which='both')

plt.show()

# Cost graph - total
fig, ax1 = plt.subplots()
ax1.set_xlabel('PV Capacity Installed [kW]', fontsize=fontsize)
ax1.set_ylabel('Electricity Cost [€/year]', fontsize=fontsize)
ax1.plot(x_PV, outcomes['cost2'].values, label='Scenario without DR')
ax1.plot(x_PV, outcomes['cost3'].values, '--', label='Scenario with DR')

ax1.grid(which='both')
#  ax1.set_ylim(ymin=0, ymax=10000)
#  ax1.set_xlim(xmin=0, xmax=750)
ax1.legend(loc='center left')
fig.tight_layout()
plt.show()

# SSR & SCR
fig1, ax1 = plt.subplots()
ax1.set_xlabel('PV Capacity Installed [kW]', fontsize=fontsize)
ax1.set_ylabel('Self Sufficiency Rate [%]', fontsize=fontsize)

ax1.plot(x_PV, outcomes['SSR2'].values, label='Scenario without DR')
ax1.plot(x_PV, outcomes['SSR3'].values, '--', label='Scenario with DR')
ax1.set_ylim(ymin=0, ymax=100)
ax1.set_xlim(xmin=0, xmax=745)

ax1.grid(which='both')
ax1.legend(loc='center left')

fig2, ax1 = plt.subplots()
ax1.set_xlabel('PV Capacity Installed [kW]', fontsize=fontsize)
ax1.set_ylabel('Self Consumption Rate [%]', fontsize=fontsize)

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
#  ax2.set_ylim(ymin=7500, ymax=10000)
#  ax2.set_xlim(xmin=0, xmax=745)

ax2.legend(loc='upper right')
fig2.tight_layout()
plt.show()

# MV, LV, PV profiles of a set weekday
day = 3
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time frames of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

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


