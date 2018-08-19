import numpy as np
import pandas as pd
from scipy.stats import norm


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


number_ewh = 190
# Define consumption unit
P_nominal = 4.2   # Nominal power in kW
ewh_consumption = P_nominal * .25

#  Create MultiIndex DataFrame for EWH profiles
iterable = [np.array(range(number_ewh)),
            np.array(range(2688))]
index = pd.MultiIndex.from_product(iterable)
ewh_profiles = pd.DataFrame(index=index,
                            columns=['shower', 'consumption'])

for i in range(number_ewh):
    ewh_profiles.loc[i, 'shower'] = ewh_profile(300, 8, 19)

    for day in range(7*4):
        # Clean morning (from 0 till 47)
        n_drop = ewh_profiles.loc[i, 'shower'][day*96:(day*96+48)][ewh_profiles.loc[i, 'shower'] > 0].count() - 1

        if n_drop > 0:
            drop_indices = np.random.choice(ewh_profiles.loc[i, 'shower'][day*96:(day*96+48)]
                                            [ewh_profiles.loc[i, 'shower'] > 0].index, n_drop, replace=False)
            ewh_profiles.loc[i, 'shower'][drop_indices] = 0

        consumption_time = ewh_profiles.loc[i, 'shower'][day*96:(day*96+48)][ewh_profiles.loc[i, 'shower'] > 0]

        ewh_profiles.loc[i, 'consumption'][consumption_time.index + 1] = ewh_consumption
        ewh_profiles.loc[i, 'consumption'][consumption_time.index + 2] = ewh_consumption

        # Clean evening (from 48 till 95)
        n_drop = ewh_profiles.loc[i, 'shower'][(day*96+48):(day*96+96)][ewh_profiles.loc[i, 'shower'] > 0].count() - 1
        if n_drop > 0:
            drop_indices = np.random.choice(ewh_profiles.loc[i, 'shower'][(day*96+48):(day*96+96)]
                                            [ewh_profiles.loc[i, 'shower'] > 0].index, n_drop, replace=False)
            ewh_profiles.loc[i, 'shower'][drop_indices] = 0

        consumption_time = ewh_profiles.loc[i, 'shower'][(day*96+48):(day*96+96)][ewh_profiles.loc[i, 'shower'] > 0]

        if consumption_time.index >= 2685:
            ewh_profiles.loc[i, 'consumption'][2686] = ewh_consumption
            ewh_profiles.loc[i, 'consumption'][2867] = ewh_consumption

            ewh_profiles.loc[i, 'shower'][2685:2688] = 0
            ewh_profiles.loc[i, 'shower'][2685] = 1
        else:
            ewh_profiles.loc[i, 'consumption'][consumption_time.index + 1] = ewh_consumption
            ewh_profiles.loc[i, 'consumption'][consumption_time.index + 2] = ewh_consumption

    ewh_profiles.loc[i, 'shower'].replace(0, np.nan, inplace=True)
    ewh_profiles.loc[i, 'consumption'].fillna(value=0, inplace=True)

ewh_profiles.to_csv('ewh_%d.csv' % number_ewh)
