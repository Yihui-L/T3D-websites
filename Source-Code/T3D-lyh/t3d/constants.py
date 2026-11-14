from math import sqrt

from scipy.constants import elementary_charge as ELEMENTARY_CHARGE
from scipy.constants import speed_of_light as SPEED_OF_LIGHT
from scipy.constants import proton_mass
from scipy.constants import electron_volt
from scipy.constants import physical_constants

_KILO = 1e3
_KILOGRAMS_TO_GRAMS = _KILO

ATOMIC_MASS_CONSTANT = physical_constants["atomic mass constant"][0]  # in kg
M_PROTON_CGS = proton_mass * _KILOGRAMS_TO_GRAMS
EV_JOULES = electron_volt

# reference thermal speed in m/s: sqrt(1 keV / m)
# m is proton mass
REFERENCE_THERMAL_SPEED = sqrt(_KILO * EV_JOULES / proton_mass)

# reference gyroradius, in meters, for a 1 keV-energy proton in a 1 T field
# (m v) / q B
# m is a proton mass in kg
# v is the reference thermal speed at 1 keV, in m/s
# q is the elementary charge, in C
# B is 1 Tesla
REFERENCE_GYRORADIUS = sqrt(_KILO * EV_JOULES * proton_mass) / ELEMENTARY_CHARGE
