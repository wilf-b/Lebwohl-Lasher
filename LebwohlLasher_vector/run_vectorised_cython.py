import sys
from LebwohlLasher_vectorised_cython import main

ITERATIONS = 250
SIZE = 50
TEMPERATURE = 0.5
PLOTFLAG = 0

main("LebwohlLasher_vectorised_cython", ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)