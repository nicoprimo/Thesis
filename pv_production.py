import pandas as pd
import matplotlib.pyplot as plt
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.location import Location

# Location data for Martim Longo
latitude = 37.442
longitude = -7.771
tz = 'UTC'
altitude = 249
location = Location(latitude, longitude, tz=tz, altitude=altitude)

# Weather data from Meteonorm, 1 min resolution
df1 = pd.read_excel('Marim_Longo-min.xlsx')
df1['times'] = pd.date_range('1/1/2016', periods=525600, freq='min')
df1.index = df1['times']
df1.index.name = 'Time'

df = df1.loc[df1['dy'] <= 7]    # get the portion of data required

times = df['times']
d_weather = {'temp_air': df['Ta'],
             'wind_speed': df['FF'],
             'ghi': df['G_Gh'],
             'dni': df['G_Bn'],
             'dhi': df['G_Dh']}
weather_data = pd.DataFrame(d_weather)

# PV data
pv_db = pvlib.pvsystem.retrieve_sam('SandiaMod')
inverter_db = pvlib.pvsystem.retrieve_sam('cecinverter')

pv_data = pv_db['Hanwha_HSL60P6_PA_4_250T__2013_']
inverter = inverter_db['iPower__SHO_5_2__240V__240V__CEC_2018_']

# Production
system = PVSystem(surface_tilt=37, surface_azimuth=180, albedo=0.2,
                  surface_type=None, module_parameters=pv_data,
                  modules_per_string=10, strings_per_inverter=2,
                  inverter_parameters=inverter,
                  racking_model='open_rack_cell_glassback')
mc = ModelChain(system, location, orientation_strategy=None,
                clearsky_model='ineichen', transposition_model='perez',
                solar_position_method='nrel_numpy',
                airmass_model='kastenyoung1989',
                dc_model='sapm', ac_model=None, aoi_model='physical',
                spectral_model='no_loss', temp_model='sapm',
                losses_model='no_loss', name=None)

mc.run_model(times, None, weather_data)
power_ac = pd.Series(mc.ac)
power_ac.fillna(value=0, inplace=True)

# Summing up the energy produced each 15 min
energy_produced_min = power_ac / 60
energy_produced_15min = energy_produced_min.resample('15min').mean()

plt.plot(energy_produced_15min)
plt.grid()
plt.show()
