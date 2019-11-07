[![Documentation Status](https://readthedocs.org/projects/urban-driving-simulator/badge/?version=v2)](https://urban-driving-simulator.readthedocs.io/en/v2/?badge=v2) [![Build Status](https://travis-ci.org/BerkeleyAutomation/Urban_Driving_Simulator.svg?branch=v2)](https://travis-ci.org/BerkeleyAutomation/Urban_Driving_Simulator) [![Coverage Status](https://coveralls.io/repos/github/BerkeleyAutomation/Urban_Driving_Simulator/badge.svg)](https://coveralls.io/github/BerkeleyAutomation/Urban_Driving_Simulator)

# FLUIDS 2.0

The core FLUIDS simulator.

To install from source, 
```
pip3 install -e .
```

For examples see `examples/fluids_test.py`.

## Gym Environment

The FLUIDS Gym environments provide a familiar interface to the FLUIDS simulator.

To install from source,
```
git submodule update --init --recursive
pip3 install -e gym_fluids
```

## Testing FLUIDS

```
make test
```
Travis testing will sometimes fail due to a xvfb issue with pygame. If you see a pygame error related to the bitdepth and alpha channel try restarting the build.
