import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


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


number_ewh = 50
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

plt.plot(ewh_profiles.sum(1))
plt.show()
