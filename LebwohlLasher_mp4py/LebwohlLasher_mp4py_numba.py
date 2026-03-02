# This code is not finished as there is an issue with the ghost cells causing a variable size of the tensor
# this make calcualting the eigenvalues of the matrix hard to do dynamically.  

"""
Mpi4py implementation of Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

mpirun -np <nDOMAINS> python LebwohlLasher_mp4py.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

WB 26.02.26
"""

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpi4py import MPI
from numba import njit

# ============================================================
# MPI setup


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dims = MPI.Compute_dims(size, 2)
cart = comm.Create_cart(dims, periods=[True, True], reorder=True)

up, down = cart.Shift(0, 1)
left, right = cart.Shift(1, 1)

# ============================================================
def initdat_local(nx, ny):
    '''
    this needs modification
    '''
    arr = np.zeros((nx+2, ny+2))
    arr[1:-1,1:-1] = np.random.random((nx,ny))*2*np.pi
    return arr

# ============================================================

def halo_exchange(arr):
    """
    Perform nearest-neighbour halo exchange for a 2D domain-decomposed lattice.

    Description:
        Each MPI process owns a sub-block of the global lattice,
        including an extra layer of ghost (halo) cells around the
        boundary. These ghost cells store boundary values from
        neighbouring MPI ranks.

        This function exchanges boundary rows/columns with the
        four Cartesian neighbours (up, down, left, right) using
        MPI Sendrecv calls.

        After completion:
            - The top ghost row contains data from the upper neighbour
            - The bottom ghost row contains data from the lower neighbour
            - The left ghost column contains data from the left neighbour
            - The right ghost column contains data from the right neighbour

    Parameters:
        local (ndarray):
            2D NumPy array of shape (nx+2, ny+2),
            where interior points are in:
                local[1:-1, 1:-1]
            and ghost cells occupy the outer layer.

    Returns:
        None
            The function modifies the local array in place.
    """
    send = arr[1,1:-1].copy()
    recv = np.empty_like(send)
    cart.Sendrecv(send, up, recvbuf=recv, source=down)
    arr[-1,1:-1] = recv

    send = arr[-2,1:-1].copy()
    recv = np.empty_like(send)
    cart.Sendrecv(send, down, recvbuf=recv, source=up)
    arr[0,1:-1] = recv

    send = arr[1:-1,1].copy()
    recv = np.empty_like(send)
    cart.Sendrecv(send, left, recvbuf=recv, source=right)
    arr[1:-1,-1] = recv

    send = arr[1:-1,-2].copy()
    recv = np.empty_like(send)
    cart.Sendrecv(send, right, recvbuf=recv, source=left)
    arr[1:-1,0] = recv

# ============================================================
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """

    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
# ============================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "mpi-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
# ============================================================
def one_energy(arr,ix,iy):

    en = 0.0

    # do not deal with coord of the neighbors as these are dealt with in

    en += 0.5*(1-3*np.cos(arr[ix,iy]-arr[ix+1,iy])**2)
    en += 0.5*(1-3*np.cos(arr[ix,iy]-arr[ix-1,iy])**2)
    en += 0.5*(1-3*np.cos(arr[ix,iy]-arr[ix,iy+1])**2)
    en += 0.5*(1-3*np.cos(arr[ix,iy]-arr[ix,iy-1])**2)

    return en

# ============================================================

def all_energy(arr):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      note all f's have been modified to not use nmax as this is not needed
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
	Returns:
	  enall (float) = reduced energy of lattice.
    """
    # calls halo exchange to get the neighbors 
    halo_exchange(arr)

    nx = arr.shape[0]-2
    ny = arr.shape[1]-2

    enall = 0.0
    # has to iterate one further owing to ghost cells
    for i in range(1,nx+1):
        for j in range(1,ny+1):
            enall += one_energy(arr,i,j)

    return comm.allreduce(enall, op=MPI.SUM)

# ============================================================
@njit
def get_order(arr):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    nx = arr.shape[0]-2
    ny = arr.shape[1]-2
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((
        np.cos(arr[1:-1,1:-1]),
        np.sin(arr[1:-1,1:-1]),
        np.zeros((nx,ny))
    )).reshape(3,nx*ny)

    Qab = np.zeros((3,3))

    for v in lab.T:
        Qab += 3*np.outer(v,v)-np.eye(3)

    Qab /= (2*nx*ny)

    Qab = comm.allreduce(Qab, op=MPI.SUM)
    # this is causing issues as the tensor is not of a defined shape owing top the ghost cells 
    # as ;attice is (nx+2,ny+2)
    eigenvalues,eigvec = np.linalg.eigvals(Qab)

    return eigenvalues.max()

# ============================================================
@njit
def MC_step(arr,Ts):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
    #
    # Pre-compute some random numbers.  This is faster than
    # using lots of individual calls.  "scale" sets the width
    # of the distribution for the angle changes - increases
    # with temperature.
    halo_exchange(arr)

    nx = arr.shape[0]-2
    ny = arr.shape[1]-2

    scale = 0.1 + Ts

    accept = 0

    for i in range(1,nx+1):
        for j in range(1,ny+1):

            old = arr[i,j]

            en0 = one_energy(arr,i,j)

            arr[i,j] += np.random.normal(scale=scale)

            en1 = one_energy(arr,i,j)

            if en1 <= en0:
                accept += 1
            else:
                boltz = np.exp(-(en1-en0)/Ts)

                if boltz >= np.random.rand():
                    accept += 1
                else:
                    arr[i,j] = old

    total_accept = comm.allreduce(accept, op=MPI.SUM)

    total_sites = nx*ny*size

    return total_accept/total_sites

# ============================================================
def gather_lattice(local,nmax):

    nx = local.shape[0]-2
    ny = local.shape[1]-2

    sendbuf = local[1:-1,1:-1].copy()

    recvbuf = None

    if rank == 0:
        recvbuf = np.empty((nmax,nmax))

    comm.Gather(sendbuf, recvbuf, root=0)

    return recvbuf

# ============================================================
def main(program, nsteps, nmax, temp, pflag):

    nx_local = nmax//dims[0]
    ny_local = nmax//dims[1]

    lattice = initdat_local(nx_local,ny_local)

    full0 = gather_lattice(lattice,nmax)
    plotdat(full0,pflag,nmax)

    energy = np.zeros(nsteps+1)
    ratio = np.zeros(nsteps+1)
    order = np.zeros(nsteps+1)

    energy[0] = all_energy(lattice)
    ratio[0] = 0.5
    order[0] = get_order(lattice)

    comm.Barrier()

    initial = time.time()

    for it in range(1,nsteps+1):

        ratio[it] = MC_step(lattice,temp)
        energy[it] = all_energy(lattice)
        order[it] = get_order(lattice)

    comm.Barrier()

    final = time.time()

    runtime = final-initial

    full = gather_lattice(lattice,nmax)

    if rank==0:

        print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(
            program,nmax,nsteps,temp,order[nsteps-1],runtime))

    savedat(full,nsteps,temp,runtime,ratio,energy,order,nmax)

    plotdat(full,pflag,nmax)

# ============================================================

if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================
