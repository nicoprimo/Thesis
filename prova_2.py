import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Variable
number_ewh = 50
LC = 151 / 52 * 4  # Price in euro/kW of PV installed per 1 week (1 year = 52 weeks) * 4 weeks (reference ones)
feed_in_tariff = 0.0377  # should be set as the average price in the stock market times 0.90 - 0.0377 from literature

gap = 100

# Read LV and MV aggregated demand // PV production // Price of electricity from the grid
# PV production
df_PV = pd.read_csv('pv_production.csv', index_col=0)
df_PV['time'] = pd.to_datetime(df_PV.index, dayfirst=True)
df_PV.index = df_PV['time']

# get he data for the reference weeks
df_PV_march = df_PV.loc['20160314':'201603210001']
df_PV_june = df_PV.loc['201606060001':'201606130001']
df_PV_september = df_PV.loc['201609120001':'201609190001']
df_PV_december = df_PV.loc['201612050001':'201612120001']

pv_production_march = df_PV_march['PV production']
pv_production_june = df_PV_june['PV production']
pv_production_september = df_PV_september['PV production']
pv_production_december = df_PV_december['PV production']

pv_production = pd.concat([pv_production_march,
                           pv_production_june,
                           pv_production_september,
                           pv_production_december],
                          ignore_index=True)

# Demand
df_LV = pd.read_csv('community_demand.csv', index_col=0)
df_MV = pd.read_csv('MV_demand.csv', index_col=0)
df_MV['time'] = pd.to_datetime(df_MV.index, dayfirst=True)
df_MV.index = df_MV['time']

df_MV_march = df_MV.loc['20160314':'201603210001']
df_MV_june = df_MV.loc['201606060001':'201606130001']
df_MV_september = df_MV.loc['201609120001':'201609190001']
df_MV_december = df_MV.loc['201612050001':'201612120001']

LV_consumption_winter = np.asarray(df_LV['final consumption winter'])
LV_consumption_summer = np.asarray(df_LV['final consumption winter'])  # need to create the summer consumption for LV
MV_consumption_march = np.asarray(df_MV_march['final consumption march'])
MV_consumption_june = np.asarray(df_MV_june['final consumption june'])
MV_consumption_september = np.asarray(df_MV_september['final consumption september'])
MV_consumption_december = np.asarray(df_MV_december['final consumption december'])


consumption_march = pd.Series(LV_consumption_winter + MV_consumption_march)
consumption_june = pd.Series(LV_consumption_summer + MV_consumption_june)
consumption_september = pd.Series(LV_consumption_summer + MV_consumption_september)
consumption_december = pd.Series(LV_consumption_winter + MV_consumption_december)

demand = pd.concat([consumption_march,
                    consumption_june,
                    consumption_september,
                    consumption_december],
                   ignore_index=True)

# Grid price
df_tariff_winter = pd.read_excel('tariff.xlsx', sheet_name='Winter_week')
df_tariff_summer = pd.read_excel('tariff.xlsx', sheet_name='Summer_week')

grid_price_winter = df_tariff_winter['Price']
period_price_winter = df_tariff_winter['Period']
grid_price_summer = df_tariff_summer['Price']
period_price_summer = df_tariff_summer['Period']

grid_price = pd.concat([grid_price_winter,
                        grid_price_summer,
                        grid_price_summer,
                        grid_price_winter], ignore_index=True)

period_price = pd.concat([period_price_winter,
                          period_price_summer,
                          period_price_summer,
                          period_price_winter],
                         ignore_index=True)


# Generate profiles on/of for EWH deciding morning/evening peak hours and peak gap


def assign_with_probability(p):
    return np.random.choice(a=[1, 0], p=[p, 1 - p])


# Vectorize functions
v_assign_with_probability = np.vectorize(assign_with_probability)


def ewh_profile(peak_gap, morning_peak, evening_peak):
    # Create a vector with very low probability
    index_minutes = pd.date_range(start='22/06/2018', periods=24*60, freq='T')
    probability_vector = np.full(24 * 60, fill_value=0.001)
    probability_vector[(morning_peak * 60) - peak_gap:(morning_peak * 60) + peak_gap] = \
        norm.pdf(np.linspace(norm.ppf(0.001),
                             norm.ppf(0.999), peak_gap * 2), scale=1)
    probability_vector[(evening_peak * 60) - peak_gap:(evening_peak * 60) + peak_gap] = \
        norm.pdf(np.linspace(norm.ppf(0.001),
                             norm.ppf(0.999), peak_gap * 2), scale=1)
    probability_vector = pd.Series(probability_vector, index=index_minutes)
    probability_vector = probability_vector.resample('15T').mean()
    # Create a probability vector for 4 weeks
    probability_vector_week = np.tile(probability_vector.values, 7 * 4)

    # Create vector ewh_on vector using probability_vector
    return v_assign_with_probability(probability_vector_week)


ewh_profiles = pd.DataFrame()
for i in range(number_ewh):
    ewh_profiles['%d' % i] = ewh_profile(300, 8, 19)
    for day in range(7*4):
        # Clean morning (from 0 till 47)
        n_drop = ewh_profiles.loc[day*96:(day*96+47), '%d' % i][ewh_profiles['%d' % i] > 0].count() - 1
        if n_drop > 0:
            drop_indices = np.random.choice(ewh_profiles.loc[day*96:(day*96+47), '%d' % i][ewh_profiles['%d' % i] > 0].index,
                                            n_drop, replace=False)
            ewh_profiles.loc[day*96:(day*96+47), '%d' % i] = ewh_profiles.loc[day*96:(day*96+47), '%d' % i].drop(drop_indices)
            ewh_profiles.loc[day*96:(day*96+47), '%d' % i].fillna(value=0, inplace=True)

        # Clean evening (from 48 till 95)
        n_drop = ewh_profiles.loc[(day*96+48):(day*96+95), '%d' % i][ewh_profiles['%d' % i] > 0].count() - 1
        if n_drop > 0:
            drop_indices = np.random.choice(ewh_profiles.loc[(day*96+48):(day*96+95), '%d' % i][ewh_profiles['%d' % i] > 0].index,
                                        n_drop, replace=False)
            ewh_profiles.loc[(day*96+48):(day*96+95), '%d' % i] = ewh_profiles.loc[(day*96+48):(day*96+95), '%d' % i].drop(drop_indices)
            ewh_profiles.loc[(day*96+48):(day*96+95), '%d' % i].fillna(value=0, inplace=True)

# Sum how many EWH are on per period
ewh_profiles['total'] = ewh_profiles.sum(1)

# Decide rated power, therefore single consumption per EWH on
rated_power = 4.5
ewh_consumption_single = rated_power * .25  # consumption per period in kWh

# create the df to operate optimization

df = pd.DataFrame({'demand': demand,
                   'flex': ewh_profiles['total'].values * ewh_consumption_single,
                   'pv production': pv_production,
                   'ewh status': ewh_profiles['total'].values,
                   'grid price': grid_price}, dtype=float)
ewh_profiles.drop('total', axis=1, inplace=True)

# assign Nan to all the time steps where there are no EWHs on - Easy later
ewh_profiles[ewh_profiles != 1] = np.nan
df = pd.concat([df, ewh_profiles], axis=1)
df['period price'] = period_price

# Definition of functions


def cost_period(demand, flex, pv_production, grid_price, n_set):
    if (demand + flex - n_set * pv_production) > 0:
        return (demand + flex - n_set * pv_production) * grid_price
    else:
        return (demand + flex - n_set * pv_production) * feed_in_tariff


def self_consumption(demand, flex, pv_production, n_set):
    if (demand + flex - n_set * pv_production) >= 0:
        return n_set * pv_production
    else:
        return demand + flex


def sun_surplus(demand, flex, pv_production, n_set):  # function to get the value of ewh
    # that can be shift in a specific time frame
    surplus = int((n_set * pv_production - demand + flex) / ewh_consumption_single)
    if surplus < 0:
        surplus = 0
    return surplus


# vectorize function to pass arrays instead of single values
v_cost_period = np.vectorize(cost_period)
v_sun_surplus = np.vectorize(sun_surplus)
v_self_consumption = np.vectorize(self_consumption)

cost2 = np.zeros(gap)  # to keep track of the price variation
SSR_2 = np.zeros(gap)
SSR_ewh_2 = np.zeros(gap)
SCR_2 = np.zeros(gap)
SCR_ewh_2 = np.zeros(gap)

for n_set2 in range(gap):  # See the price variations up to 100 PV systems
    df['cost2'] = v_cost_period(df['demand'].values, df['flex'].values,
                                df['pv production'].values, df['grid price'].values, n_set2)
    cost2[n_set2] = df['cost2'].sum() + n_set2 * 5 * LC
    SSR_2[n_set2] = (v_self_consumption(df['demand'].values, df['flex'].values,
                     df['pv production'].values, n_set2).sum() / (df['demand'].sum() + df['flex'].sum())) * 100
    SSR_ewh_2[n_set2] = (v_self_consumption(0, df['flex'].values,
                         df['pv production'].values, n_set2).sum() / (df['flex'].sum())) * 100
    SCR_2[n_set2] = (v_self_consumption(df['demand'].values, df['flex'].values,
                     df['pv production'].values, n_set2).sum() / (df['pv production'] * n_set2).sum()) * 100
    SCR_ewh_2[n_set2] = (v_self_consumption(0, df['flex'].values, df['pv production'].values, n_set2).sum() / (
                                     df['pv production'] * n_set2).sum()) * 100


# delta = 20

cost3 = np.zeros(gap)
SSR_3 = np.zeros(gap)
SSR_ewh_3 = np.zeros(gap)
SCR_3 = np.zeros(gap)
SCR_ewh_3 = np.zeros(gap)

cost3_min = cost2.min()
for n_set3 in range(gap):  # See the price variations up to 100 PV systems
    # add flexibility part // Align as much as possible ewh status and sun surplus //
    print('**** PV number %d ****' % n_set3)
    df['sun surplus'] = v_sun_surplus(df['demand'].values, df['flex'].values, df['pv production'].values, n_set3)

    new_ewh_status = np.zeros(shape=(96, 28))
    for day in range(28):
        print('DAY %d' % day)
        # Get the values of the day
        df_day = df[(day * 96):((day + 1) * 96)]
        df_day.reset_index(inplace=True, drop=True)

        period_price_day = df_day['period price'].values
        sun_surplus_day = df_day['sun surplus'].values
        ewh_status_day = df_day['ewh status'].values
        check_status = ewh_status_day.sum()
        ewh_n_status_day = ewh_profiles[(day * 96):((day + 1) * 96)]
        ewh_n_status_day.reset_index(inplace=True, drop=True)

        # Get the profiles "ponta"
        ewh_ponta_day = ewh_n_status_day[df_day['period price'] == 'p'].where(ewh_n_status_day > 0)
        if not ewh_ponta_day.dropna(axis=0, how='all').empty:
            ponta_starting_index = int(ewh_ponta_day.dropna(axis=0, how='all').index[0])

        # Get the profiles "cheia"
        ewh_cheia_day = ewh_n_status_day[df_day['period price'] == 'c'].where(ewh_n_status_day > 0)
        if not ewh_cheia_day.dropna(axis=0, how='all').empty:
            cheia_starting_index = int(ewh_cheia_day.dropna(axis=0, how='all').index[0])

        # Get the profiles "vazio"
        ewh_vazio_day = ewh_n_status_day[df_day['period price'] == 'v'].where(ewh_n_status_day > 0)
        if not ewh_vazio_day.dropna(axis=0, how='all').empty:
            vazio_starting_index = int(ewh_vazio_day.dropna(axis=0, how='all').index[0])

        # Get the profiles "super vazio"
        ewh_supervazio_day = ewh_n_status_day[df_day['period price'] == 'sv'].where(ewh_n_status_day > 0)
        if not ewh_supervazio_day.dropna(axis=0, how='all').empty:
            supervazio_starting_index = int(ewh_supervazio_day.dropna(axis=0, how='all').index[0])

        for time_step in range(96):
            # Fill up sun surplus
            skip = 0
            while sun_surplus_day[time_step] > 0 and (not ewh_ponta_day.dropna(axis=0, how='all').empty or
                                                      not ewh_cheia_day.dropna(axis=0, how='all').empty or
                                                      not ewh_vazio_day.dropna(axis=0, how='all').empty or
                                                      not ewh_supervazio_day.dropna(axis=0, how='all').empty)\
                    and skip != 1:

                start_surplus = sun_surplus_day[time_step]
                # Check if there is any ponta profile before time_step
                if not ewh_ponta_day.dropna(axis=0, how='all').empty and ponta_starting_index < time_step:
                    # Get how many EWH profiles are available
                    profile_available = ewh_ponta_day.dropna(axis=0, how='all').head(1).dropna(axis=1).shape[1]
                    # Check if there is the need to shift all the profiles
                    if profile_available > sun_surplus_day[time_step]:
                        profile_available = int(sun_surplus_day[time_step])
                    shifts = profile_available
                    # Shift the profiles available to the sun surplus
                    for i in range(profile_available):
                        ewh_available = int(ewh_ponta_day.dropna(axis=0, how='all').head(1).dropna(axis=1).columns[0])
                        # Check if the profile can be shifted
                        check_profile = ewh_n_status_day.loc[(ponta_starting_index + 1):(time_step - 1),
                                                                 '%d' % ewh_available]

                        if check_profile.dropna().empty:
                            ewh_status_day[time_step] += 1
                            ewh_status_day[ponta_starting_index] -= 1

                            ewh_ponta_day.loc[ponta_starting_index, '%d' % ewh_available] = np.nan

                        else:
                            shifts -= 1

                    # Update index of first values available and drop the 1st row of profiles used
                    sun_surplus_day[time_step] -= shifts
                    if sun_surplus_day[time_step] < 0:
                        print('** ponta **')
                        print('-.- time step %d -.-' % time_step)
                    ewh_ponta_day.dropna(how='all', axis=0, inplace=True)
                    if not ewh_ponta_day.dropna(axis=0, how='all').empty:
                        ponta_starting_index = int(ewh_ponta_day.dropna(how='all', axis=0).index[0])

                # Check for cheia profiles
                if sun_surplus_day[time_step] > 0 \
                        and not ewh_cheia_day.dropna(axis=0, how='all').empty and cheia_starting_index < time_step:
                    # Get how many EWH profiles are available
                    profile_available = ewh_cheia_day.dropna(axis=0, how='all').head(1).dropna(axis=1).shape[1]
                    # Check if there is the need to shift all the profiles
                    if profile_available > sun_surplus_day[time_step]:
                        profile_available = sun_surplus_day[time_step]
                    shifts = profile_available
                    # Shift the profiles available to the sun surplus
                    for i in range(profile_available):
                        ewh_available = int(ewh_cheia_day.dropna(axis=0, how='all').head(1).dropna(axis=1).columns[0])
                        # Check if the profile can be shifted
                        check_profile = ewh_n_status_day.loc[(cheia_starting_index + 1):(time_step - 1),
                                            '%d' % ewh_available]

                        if check_profile.dropna().empty:
                            ewh_status_day[time_step] += 1
                            ewh_status_day[cheia_starting_index] -= 1

                            ewh_cheia_day.loc[cheia_starting_index, '%d' % ewh_available] = np.nan

                        else:
                            shifts -= 1

                    # Update index of first values available and drop the 1st row of profiles used
                    sun_surplus_day[time_step] -= shifts
                    if sun_surplus_day[time_step] < 0:
                        print('** cheia **')
                        print('-.- time step %d -.-' % time_step)
                    ewh_cheia_day.dropna(how='all', axis=0, inplace=True)
                    if not ewh_cheia_day.empty:
                        cheia_starting_index = int(ewh_cheia_day.dropna(how='all', axis=0).index[0])

                # Check for vazio profiles
                if sun_surplus_day[time_step] > 0 \
                        and not ewh_vazio_day.dropna(axis=0, how='all').empty and vazio_starting_index < time_step:
                    # Get how many EWH profiles are available
                    profile_available = ewh_vazio_day.dropna(axis=0, how='all').head(1).dropna(axis=1).shape[1]
                    # Check if there is the need to shift all the profiles
                    if profile_available > sun_surplus_day[time_step]:
                        profile_available = sun_surplus_day[time_step]
                    shifts = profile_available
                    # Shift the profiles available to the sun surplus
                    for i in range(profile_available):
                        ewh_available = int(ewh_vazio_day.dropna(axis=0, how='all').head(1).dropna(axis=1).columns[0])
                        # Check profile
                        check_profile = ewh_n_status_day.loc[(vazio_starting_index + 1):(time_step - 1),
                                                                '%d' % ewh_available]

                        if check_profile.dropna().empty:
                            ewh_status_day[time_step] += 1
                            ewh_status_day[vazio_starting_index] -= 1

                            ewh_vazio_day.loc[vazio_starting_index, '%d' % ewh_available] = np.nan

                        else:
                            shifts -= 1

                    # Update index of first values available and drop the 1st row of profiles used
                    sun_surplus_day[time_step] -= profile_available
                    if sun_surplus_day[time_step] < 0:
                        print('** vazio **')
                        print('-.- time step %d -.-' % time_step)
                    ewh_vazio_day.dropna(how='all', axis=0, inplace=True)
                    if not ewh_vazio_day.empty:
                        vazio_starting_index = int(ewh_vazio_day.dropna(how='all', axis=0).index[0])

                # Check for supervazio profiles
                if sun_surplus_day[time_step] > 0 \
                        and not ewh_supervazio_day.dropna(axis=0, how='all').empty and supervazio_starting_index < time_step:
                    # Get how many EWH profiles are available
                    profile_available = ewh_supervazio_day.dropna(axis=0, how='all').head(1).dropna(axis=1).shape[1]
                    # Check if there is the need to shift all the profiles
                    if profile_available > sun_surplus_day[time_step]:
                        profile_available = sun_surplus_day[time_step]
                    shifts = profile_available
                    # Shift the profiles available to the sun surplus
                    for i in range(profile_available):
                        ewh_available = int(ewh_supervazio_day.dropna(axis=0, how='all').head(1).dropna(axis=1).columns[0])
                        # Check if the profile can be shifted
                        check_profile = ewh_n_status_day.loc[(supervazio_starting_index + 1):(time_step - 1),
                                                                '%d' % ewh_available]

                        if check_profile.dropna().empty:
                            ewh_status_day[time_step] += 1
                            ewh_status_day[supervazio_starting_index] -= 1

                            ewh_supervazio_day.loc[supervazio_starting_index, '%d' % ewh_available] = np.nan

                        else:
                            shifts -= 1

                    # Update index of first values available and drop the 1st row of profiles used
                    sun_surplus_day[time_step] -= shifts
                    if sun_surplus_day[time_step] < 0:
                        print('** supervazio **')
                        print('-.- time step %d -.-' % time_step)
                    ewh_supervazio_day.dropna(how='all', axis=0, inplace=True)

                    if not ewh_supervazio_day.dropna(axis=0, how='all').empty:
                        supervazio_starting_index = int(ewh_supervazio_day.dropna(how='all', axis=0).index[0])

                # Check if there are changes in sun surplus - if not stop while cycle
                if start_surplus == sun_surplus_day[time_step]:
                    skip = 1

        new_ewh_status[:, day] = ewh_status_day[:]
        if (ewh_status_day.sum() - check_status) < 0:
            print('*.* Low *.*')
        elif (ewh_status_day.sum() - check_status) > 0:
            print('*.* High *.*')

    new_ewh_status_df = new_ewh_status[:, 0]
    for i in range(1, 28):
        new_ewh_status_df = np.append(arr=new_ewh_status_df, values=new_ewh_status[:, i])
    df['new ewh status 1'] = pd.Series(new_ewh_status_df)

    # recalculate flex vector
    df['new flex 1'] = df['new ewh status 1'].values * ewh_consumption_single
    # Get the cost for the scenario
    df['cost3'] = v_cost_period(df['demand'].values, df['new flex 1'].values,
                                df['pv production'].values, df["grid price"].values, n_set3)
    SSR_3[n_set3] = (v_self_consumption(df['demand'].values, df['new flex 1'].values,
                                        df['pv production'].values, n_set3).sum() /
                     (df['demand'].sum() + df['new flex 1'].sum())) * 100
    SSR_ewh_3[n_set3] = (v_self_consumption(0, df['new flex 1'].values, df['pv production'].values, n_set3).sum() /
                         (df['new flex 1'].sum())) * 100
    SCR_3[n_set3] = (v_self_consumption(df['demand'].values, df['new flex 1'].values,
                                        df['pv production'].values, n_set3).sum() /
                     (df['pv production'] * n_set3).sum()) * 100
    SCR_ewh_3[n_set3] = (v_self_consumption(0, df['new flex 1'].values, df['pv production'].values, n_set3).sum() /
                         (df['pv production'] * n_set3).sum()) * 100

    cost3[n_set3] = df['cost3'].sum() + n_set3 * 5 * LC

    if n_set3 == cost2.argmin():
        df['new flex'] = df['new flex 1']
        df['new ewh status'] = df['new ewh status 1']
        cost3_min = cost3[n_set3]
    elif cost3[n_set3] < cost3_min:
        df['new flex'] = df['new flex 1']
        df['new ewh status'] = df['new ewh status 1']
        cost3_min = cost3[n_set3]

# Print results
print('* Set up *')
print('FIT = %.2f' % feed_in_tariff + '€/kWh')
print('LC = %d' % (LC*52/4) + '€/kW/year')
print('number of EWHs = %d' % number_ewh)
print('* Yearly Costs *')
print('without %.2f' % min(cost2))
print('with %.2f' % min(cost3))
print('* Optimal PV installation *')
print('without %d' % (cost2.argmin() * 5) + 'kW')
print('with %d' % (cost3.argmin() * 5) + 'kW')
print('* Self Consumption comparison *')
print('SCR')
print('without flex %.2f' % SCR_2[cost2.argmin()])
print('with flex %.2f' % SCR_3[cost2.argmin()])
print('SSR')
print('without flex %.2f' % SSR_2[cost2.argmin()])
print('with flex %.2f' % SSR_3[cost2.argmin()])
print('SCR only EWH')
print('without flex %.2f' % SCR_ewh_2[cost2.argmin()])
print('with flex %.2f' % SCR_ewh_3[cost2.argmin()])
print('SSR only EWH')
print('without flex %.2f' % SSR_ewh_2[cost2.argmin()])
print('with flex %.2f' % SSR_ewh_3[cost2.argmin()])

df['sun surplus'] = v_sun_surplus(df['demand'].values, df['flex'].values, df['pv production'].values, cost3.argmin())

plt.plot(df['pv production'] * (cost3.argmin()), 'r')
plt.plot(df['demand'] + df['new flex'], '--')
plt.grid()

plt.show()

plt.plot(df['ewh status'], 'g')
plt.plot(df['new ewh status'], '--k')
plt.plot(df['sun surplus'], 'r')

plt.show()

# Scenario 2 vs Scenario 3 - overall
fig, ax1 = plt.subplots()
ax1.set_xlabel('number of 5 kW PV systems')
ax1.set_ylabel('Price spent per year (€)')
ax1.plot(cost2, 'g', label='cost2')
ax1.plot(cost3, '--b', label='cost3')
ax1.legend(loc=2)


ax2 = ax1.twinx()
ax2.set_ylabel('Self Sufficiency Rates (%) ')
ax2.plot(SSR_2, 'y', label='SSR2')
ax2.plot(SSR_3, '-.k', label='SSR3')
ax2.legend(loc=6)
fig.tight_layout()
plt.grid()
plt.show()
# -----------------------------------
fig, ax1 = plt.subplots()
ax1.set_xlabel('Number of 5 kW PV systems')
ax1.set_ylabel('Price spent per year (€)')
ax1.plot(cost2, 'g', label='cost2')
ax1.plot(cost3, '--b', label='cost3')
ax1.legend(loc=2)


ax2 = ax1.twinx()
ax2.set_ylabel('Self Consumption Rates (%) ')
ax2.plot(SCR_2, 'y', label='SCR2')
ax2.plot(SCR_3, '-.k', label='SCR3')
ax2.legend(loc=6)
fig.tight_layout()
plt.grid()
plt.show()

# Scenario 2 and 3 EWH comparison
fig, ax1 = plt.subplots()
ax1.set_xlabel('Number of 5 kW PV systems')
ax1.set_ylabel('Price spent per year (€)')
ax1.plot(cost2, 'g', label='cost2')
ax1.plot(cost3, '--b', label='cost3')
ax1.legend(loc=2)

ax2 = ax1.twinx()
ax2.set_ylabel('Self Sufficiency Ratio (%)')
ax2.plot(SSR_ewh_2, 'y', label='SSR2 only EWH')
ax2.plot(SSR_ewh_3, '-.k', label='SSR3 only EWH')
ax2.legend(loc=6)
fig.tight_layout()
plt.grid()
plt.show()
# ----------------------------
fig, ax1 = plt.subplots()
ax1.set_xlabel('Number of 5 kW PV systems')
ax1.set_ylabel('Price spent per year (€)')
ax1.plot(cost2, 'g', label='cost2')
ax1.plot(cost3, '--b', label='cost3')
ax1.legend(loc=2)

ax2 = ax1.twinx()
ax2.set_ylabel('Self Consumption Ratio (%)')
ax2.plot(SCR_ewh_2, 'y', label='SCR2 only EWHs')
ax2.plot(SCR_ewh_3, '-.k', label='SCR3 only EWHs')
ax2.legend(loc=6)
fig.tight_layout()
plt.grid()
plt.show()
