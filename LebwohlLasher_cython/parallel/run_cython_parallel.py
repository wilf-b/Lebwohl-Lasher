import sys
from LebwohlLasher_cython_parallel import main

ITERATIONS = 250
SIZE = 50
TEMPERATURE = 0.5
PLOTFLAG = 0

main("LebwohlLasher_cython_parallel", ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)