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

total_surplus = np.zeros(14)
total_surplus_flex = np.zeros(14)
for i in range(1, 15):
    total_surplus[i-1] = v_sun_surplus(total_noflex, inputs['PV'], i*10).sum()
    total_surplus_flex[i-1] = v_sun_surplus(total_flex[:, (i-1)], inputs['PV'], i*10).sum()
    print(total_surplus[i-1])
    print(' ---- %d' % (i-1))

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

"CHARTS"
x_day = pd.date_range('1/1/2016 00:00', periods=96, freq='15min').strftime('%X')

# Community load
day = 23
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot(x_day, (inputs['MV'].values[day*96:(day*96+96)] +
         inputs['LV'].values[day*96:(day*96+96)] +
         inputs['EWH'].values[day*96:(day*96+96)])/.25, label='Scenario without DR')
ax1.plot(x_day, (inputs['MV'].values[day*96:(day*96+96)] +
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
ax1.xaxis.set_ticks(pd.date_range('1/1/2016 00:00', periods=6, freq='240min').strftime('%X'))
ax1.grid(which='both')

day = 27
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot(x_day, (inputs['MV'].values[day*96:(day*96+96)] +
         inputs['LV'].values[day*96:(day*96+96)] +
         inputs['EWH'].values[day*96:(day*96+96)])/.25, label='Scenario without DR')
ax1.plot(x_day, (inputs['MV'].values[day*96:(day*96+96)] +
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
ax1.xaxis.set_ticks(pd.date_range('1/1/2016 00:00', periods=6, freq='240min').strftime('%X'))
ax1.grid(which='both')

plt.show()

# MV, LV, PV profiles of a set weekday
day = 3
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot(x_day, inputs['MV'].values[day*96:(day*96+96)] / .25, label='MV profiles')
ax1.plot(x_day, inputs['LV'].values[day*96:(day*96+96)] / .25, label='LV profiles')
ax1.plot(x_day, inputs['EWH'].values[day*96:(day*96+96)] / .25, label='EWH profiles')
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
ax1.xaxis.set_ticks(pd.date_range('1/1/2016 00:00', periods=6, freq='240min').strftime('%X'))
ax1.grid(which='both')

# MV, LV, PV profiles of a set holiday
day = 6
fig2, ax1 = plt.subplots()
ax1.set_xlabel('Time of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot(x_day, inputs['MV'].values[day*96:(day*96+96)] / .25, label='MV profiles')
ax1.plot(x_day, inputs['LV'].values[day*96:(day*96+96)] / .25, label='LV profiles')
ax1.plot(x_day, inputs['EWH'].values[day*96:(day*96+96)] / .25, label='EWH profiles')
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
ax1.xaxis.set_ticks(pd.date_range('1/1/2016 00:00', periods=6, freq='240min').strftime('%X'))
ax1.grid(which='both')

plt.show()

# Net community load
day = 23
fig1, ax1 = plt.subplots()
ax1.set_xlabel('Time of day %d' % day, fontsize=fontsize)
ax1.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax1.plot(x_day, (inputs['MV'].values[day*96:(day*96+96)] +
         inputs['LV'].values[day*96:(day*96+96)] +
         inputs['EWH'].values[day*96:(day*96+96)] -
         inputs['PV'].values[day*96:(day*96+96)] * n_set)/.25, label='Scenario without DR')
ax1.plot(x_day, (inputs['MV'].values[day*96:(day*96+96)] +
         inputs['LV'].values[day*96:(day*96+96)] +
         inputs['EWH flex'].values[day*96:(day*96+96)] -
         inputs['PV'].values[day*96:(day*96+96)] * n_set)/.25, '--', label='Scenario with DR')

ax1.set_xlim(xmin=0, xmax=95)
ax1.set_ylim(ymin=-300, ymax=+250)
ax1.axhline(linewidth=0.5, color='black')
ax1.legend(loc='lower left')
ax1.xaxis.set_ticks(pd.date_range('1/1/2016 00:00', periods=6, freq='240min').strftime('%X'))
ax1.grid(which='both')

day = 27
fig2, ax2 = plt.subplots()
ax2.set_xlabel('Time of day %d' % day, fontsize=fontsize)
ax2.set_ylabel('Consumption Power [kW]', fontsize=fontsize)

ax2.plot(x_day, (inputs['MV'].values[day*96:(day*96+96)] +
         inputs['LV'].values[day*96:(day*96+96)] +
         inputs['EWH'].values[day*96:(day*96+96)] -
         inputs['PV'].values[day*96:(day*96+96)] * n_set)/.25, label='Scenario without DR')
ax2.plot(x_day, (inputs['MV'].values[day*96:(day*96+96)] +
         inputs['LV'].values[day*96:(day*96+96)] +
         inputs['EWH flex'].values[day*96:(day*96+96)] -
         inputs['PV'].values[day*96:(day*96+96)] * n_set)/.25, '--', label='Scenario with DR')

ax2.set_xlim(xmin=0, xmax=95)
ax2.set_ylim(ymin=-300, ymax=250)
ax2.axhline(linewidth=0.5, color='black')
ax2.legend()
ax2.xaxis.set_ticks(pd.date_range('1/1/2016 00:00', periods=6, freq='240min').strftime('%X'))
ax2.grid(which='major', axis='both')

plt.show()
