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
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
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
    # direction 0 (rows): negative disp is "north", positive disp is "south"
    nbr_n, nbr_s = cart.Shift(0, 1)  # returns (source, dest)

    src_n, dst_s = cart.Shift(0, 1)
    src_w, dst_e = cart.Shift(1, 1)

    # Exchange rows (north/south)
    send_n = arr[1, 1:-1].copy()
    recv_s = np.empty_like(send_n)
    cart.Sendrecv(sendbuf=send_n, dest=src_n, recvbuf=recv_s, source=dst_s)
    arr[-1, 1:-1] = recv_s  # south ghost gets data coming from south neighbour

    send_s = arr[-2, 1:-1].copy()
    recv_n = np.empty_like(send_s)
    cart.Sendrecv(sendbuf=send_s, dest=dst_s, recvbuf=recv_n, source=src_n)
    arr[0, 1:-1] = recv_n   # north ghost gets data coming from north neighbour

    # Exchange cols (west/east)
    send_w = arr[1:-1, 1].copy()
    recv_e = np.empty_like(send_w)
    cart.Sendrecv(sendbuf=send_w, dest=src_w, recvbuf=recv_e, source=dst_e)
    arr[1:-1, -1] = recv_e  # east ghost

    send_e = arr[1:-1, -2].copy()
    recv_w = np.empty_like(send_e)
    cart.Sendrecv(sendbuf=send_e, dest=dst_e, recvbuf=recv_w, source=src_w)
    arr[1:-1, 0] = recv_w   # west ghost
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
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
    en = 0.0

    # do not deal with coord of the neighbors as these are dealt with in
    ang = arr[ix,iy]-arr[ix+1,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix-1,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iy+1]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iy-1]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
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
    for i in range(1,nx+1):
        for j in range(1,ny+1):
            enall += one_energy(arr,i,j)

    return comm.allreduce(enall, op=MPI.SUM)

# ============================================================
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
    # get lattice size (excluding ghost cells)
    nx = arr.shape[0]-2
    ny = arr.shape[1]-2
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #

    # build 3D unit vectors (cos, sin, 0) for interior spins
    labels = np.vstack((
        np.cos(arr[1:-1,1:-1]),
        np.sin(arr[1:-1,1:-1]),
        np.zeros((nx,ny))
    )).reshape(3,nx*ny)

    Qab = np.zeros((3,3))

    for v in labels.T:
        Qab += 3*np.outer(v,v)-np.eye(3)

    Qab /= (2*nx*ny)
    # sum qab across all ranks 
    Qab = comm.allreduce(Qab, op=MPI.SUM)

    eigenvalues = np.linalg.eigvals(Qab)

    return eigenvalues.max()

# ============================================================
def MC_step(arr, Ts):
    """
    Arguments:
      arr (float(nx+2,ny+2)) = local lattice data including ghost cells;
      Ts (float) = reduced temperature (range 0 to 2).
    Description:
      Function to perform one MC step (MCS), consisting of an average
      of 1 attempted change per local lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Returns the global acceptance ratio
      for information (fraction of attempted changes that are successful).
    Returns:
      (float) = global acceptance ratio for current MCS.
    """
    #
    # Update ghost cells once at the start of the sweep so neighbour
    # values are consistent for boundary sites.
    #
    halo_exchange(arr)

    nx = arr.shape[0] - 2
    ny = arr.shape[1] - 2

    #
    # Pre-compute some random numbers.  This is faster than
    # using lots of individual calls.  "scale" sets the width
    # of the distribution for the angle changes - increases
    # with temperature.
    #
    scale = 0.1 + Ts
    accept = 0

    # choose random interior sites (local coordinates 1..nx, 1..ny)
    xran = np.random.randint(1, high=nx+1, size=(nx, ny))
    yran = np.random.randint(1, high=ny+1, size=(nx, ny))

    # proposed angle changes
    aran = np.random.normal(scale=scale, size=(nx, ny))

    for i in range(nx):
        for j in range(ny):

            ix = xran[i, j]
            iy = yran[i, j]
            ang = aran[i, j]

            en0 = one_energy(arr, ix, iy)

            arr[ix, iy] += ang

            en1 = one_energy(arr, ix, iy)

            if en1 <= en0:
                accept += 1
            else:
                # Now apply the Monte Carlo test - compare
                # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = np.exp(-(en1 - en0) / Ts)

                if boltz >= np.random.uniform(0.0, 1.0):
                    accept += 1
                else:
                    arr[ix, iy] -= ang

    # reduce acceptance count across all ranks
    total_accept = comm.allreduce(accept, op=MPI.SUM)

    # total attempted moves globally (one attempt per local site)
    total_sites = (nx * ny) * size

    return total_accept / total_sites
# ============================================================
def gather_lattice(local, nmax):
    """
    Arguments:
      local (float(nx+2,ny+2)) = local lattice block including ghost cells;
      nmax (int) = side length of the global square lattice.
    Description:
      Function to gather the interior lattice blocks from all MPI ranks
      onto the root process (rank 0).  Ghost cells are excluded.
      The gathered lattice is used for plotting and output.
    Returns:
      recvbuf (float(nmax,nmax)) = full lattice on rank 0;
      None on all other ranks.
    """

    # determine interior lattice size (exclude ghost cells)
    nx = local.shape[0] - 2
    ny = local.shape[1] - 2

    # extract interior block to send to root
    sendbuf = local[1:-1, 1:-1].copy()

    recvbuf = None

    # root allocates array for full lattice
    if rank == 0:
        recvbuf = np.empty((nmax, nmax))

    # gather all local blocks onto root process
    comm.Gather(sendbuf, recvbuf, root=0)

    return recvbuf

# ============================================================
def main(program, nsteps, nmax, temp, pflag):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    nx_local = nmax//dims[0]
    ny_local = nmax//dims[1]
    # Create and initialise lattice
    lattice = initdat_local(nx_local,ny_local)
    # Plot initial frame of lattice
    plotdat(lattice,pflag,nmax)
    # Create arrays to store energy, acceptance ratio and order parameter
    full0 = gather_lattice(lattice,nmax)
    plotdat(full0,pflag,nmax)

    energy = np.zeros(nsteps+1,dtype=float)
    ratio = np.zeros(nsteps+1,dtype=float)
    order = np.zeros(nsteps+1,dtype=float)
    # Set initial values in arrays
    energy[0] = all_energy(lattice)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice)

    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp)
        energy[it] = all_energy(lattice)
        order[it] = get_order(lattice)
    final = time.time()
    runtime = final-initial
 
    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
    # Plot final frame of lattice and generate output file
    savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
    plotdat(lattice,pflag,nmax)

#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
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
