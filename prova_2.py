import pandas as pd
import numpy as np

number_ewh = 50
feed_in_tariff = 0.0377

inputs = pd.read_csv("inputs_%d.csv" % number_ewh)
outcomes = pd.read_csv('outcomes_%d.csv' % number_ewh)
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

inputs['grid price'] = grid_price


def cost_period(flex, pv_production, grid_price):
    if (flex - pv_production) > 0:
        return (flex - pv_production) * grid_price
    else:
        return 0


v_cost_period = np.vectorize(cost_period)

cost_ewh_without = v_cost_period(inputs['EWH'].values, inputs['PV'].values, inputs['grid price'].values)
cost_ewh_with = v_cost_period(inputs['EWH flex'].values, inputs['PV'].values, inputs['grid price'].values)

print('Cost EWH with DR %f' % cost_ewh_with.sum())
print('Cost EWH without DR %f' % cost_ewh_without.sum())
print('Delta %f' % ((cost_ewh_without.sum() - cost_ewh_with.sum())/cost_ewh_without.sum()))
