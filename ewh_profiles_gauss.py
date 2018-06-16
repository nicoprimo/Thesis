from math import exp, pi, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gauss(x):
    return exp(-x ** 2 / 2) / sqrt(2 * pi)


def assign_with_probability(p):
    return np.random.choice(a=[1, 0], p=[p, 1 - p])


# Vectorize functions
v_gauss = np.vectorize(gauss)
v_assign_with_probability = np.vectorize(assign_with_probability)


def ewh_profile(peak_gap, morning_peak, evening_peak):
    # Create a vector with very low probability
    probability_vector = np.full(96, fill_value=0.01)
    probability_vector[(morning_peak * 4) - peak_gap:(morning_peak * 4) + peak_gap] = \
        v_gauss(np.array(range(-peak_gap, peak_gap)))
    probability_vector[(evening_peak * 4) - peak_gap:(evening_peak * 4) + peak_gap] = \
        v_gauss(np.array(range(-peak_gap, peak_gap)))
    # Create a probability vector for 4 weeks
    probability_vector_week = np.tile(probability_vector, 7*4)
    # Create vector ewh_on vector using probability_vector
    return v_assign_with_probability(probability_vector_week)


# This last step could be improved!
number_ewh = 50
ewh_profiles = pd.DataFrame()
for i in range(number_ewh):
    ewh_profiles['%d' % i] = ewh_profile(10, 8, 20)


ewh_profiles['total'] = ewh_profiles.sum(1)
ewh_profiles.to_csv('flex profiles.csv')
print(ewh_profiles['total'].max())
print(ewh_profiles['total'].argmax())
plt.plot(ewh_profiles['total'])
plt.show()
