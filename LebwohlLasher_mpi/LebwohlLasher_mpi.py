
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
nranks = comm.Get_size()

dims = MPI.Compute_dims(nranks, 2)
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
    # array includes a 1-cell halo (ghost layer) on all sides.
    arr = np.zeros((ny + 2, nx + 2), dtype=float)
    arr[1:-1, 1:-1] = np.random.random((ny, nx)) * 2.0 * np.pi
    return arr

# ============================================================
def halo_exchange(arr):
    """
    Arguments:
        local (ndarray):
            2D NumPy array of shape (ny+2, nx+2),
            where interior points are in:
                local[1:-1, 1:-1]
            and ghost cells occupy the outer layer.

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
            - etc
    Returns:
        None
            The procedure modifies the local array in place.
    """
    # rows: send first interior row up, receive bottom ghost from down
    send = arr[1, 1:-1].copy()
    recv = np.empty_like(send)
    cart.Sendrecv(sendbuf=send, dest=up, recvbuf=recv, source=down)
    arr[-1, 1:-1] = recv
    
    # rows: send last interior row down, receive top ghost from up
    send = arr[-2, 1:-1].copy()
    recv = np.empty_like(send)
    cart.Sendrecv(sendbuf=send, dest=down, recvbuf=recv, source=up)
    arr[0, 1:-1] = recv                 

    # cols: send first interior col left, receive right ghost from right
    send = arr[1:-1, 1].copy()
    recv = np.empty_like(send)
    cart.Sendrecv(sendbuf=send, dest=left, recvbuf=recv, source=right)
    arr[1:-1, -1] = recv                

    # cols: send last interior col right, receive left ghost from left
    send = arr[1:-1, -2].copy()
    recv = np.empty_like(send)
    cart.Sendrecv(sendbuf=send, dest=right, recvbuf=recv, source=left)
    arr[1:-1, 0] = recv                 

# ============================================================
def plotdat(arr, pflag, nmax):
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

    if pflag == 0:
        return

    # arr is the full global lattice (no ghosts), stored as arr[y, x]
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)

    cols = np.zeros((nmax, nmax), dtype=float)
    if pflag == 1:  # colour the arrows according to energy
        mpl.rc("image", cmap="rainbow")
        for iy in range(nmax):
            for ix in range(nmax):
                cols[iy, ix] = one_energy_global(arr, ix, iy, nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag == 2:  # colour the arrows according to angle
        mpl.rc("image", cmap="hsv")
        cols = arr % np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc("image", cmap="gist_gray")
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0, pivot="middle", headwidth=1, scale=1.1 * nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols, norm=norm, **quiveropts)
    ax.set_aspect("equal")
    plt.show()

# ============================================================
def savedat(arr, nsteps, Ts, runtime, ratio, energy, order, nmax):
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
    filename = "mpi_nb-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename, "w")
    # Write a header with run parameters
    print("#=====================================================", file=FileOut)
    print("# File created:        {:s}".format(current_datetime), file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax, nmax), file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps), file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts), file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime), file=FileOut)
    print("#=====================================================", file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:", file=FileOut)
    print("#=====================================================", file=FileOut)
    # Write the columns of data
    for i in range(nsteps + 1):
        print(
            "   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i, ratio[i], energy[i], order[i]),
            file=FileOut,
        )
    FileOut.close()

# ============================================================

# a division in one_energy to local/global has happened.
# this is needed as the two exsist using different memory models- local works on 


def one_energy_global(arr, ix, iy, nmax):
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

    # periodic neighbours in global lattice
    xp = (ix + 1) % nmax
    xm = (ix - 1) % nmax
    yp = (iy + 1) % nmax
    ym = (iy - 1) % nmax

    ang = arr[iy, ix] - arr[iy, xp]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[iy, ix] - arr[iy, xm]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[iy, ix] - arr[yp, ix]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[iy, ix] - arr[ym, ix]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)

    return en

# ============================================================
def one_energy_local(arr, ix, iy):
    """
    Arguments:
      arr (float(ny+2,nx+2)) = local array including ghost cells;
      ix (int) = local x lattice coordinate of cell (interior: 1..nx);
      iy (int) = local y lattice coordinate of cell (interior: 1..ny).
    Description:
      Function that computes the energy of a single local cell of the
      lattice taking into account periodic boundaries via the ghost cells.
      Working with reduced energy (U/epsilon), equivalent to setting epsilon=1.
    Returns:
      en (float) = reduced energy of cell.
    """
    en = 0.0

    # Neighbours are available directly (including across subdomain boundaries)
    ang = arr[iy, ix] - arr[iy, ix + 1]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[iy, ix] - arr[iy, ix - 1]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[iy, ix] - arr[iy + 1, ix]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)
    ang = arr[iy, ix] - arr[iy - 1, ix]
    en += 0.5 * (1.0 - 3.0 * np.cos(ang) ** 2)

    return en

# ============================================================

def all_energy(arr):
    """
    Arguments:
      arr (float(ny+2,nx+2)) = local array including ghost cells;
      note all f's have been modified to not use nmax as this is not needed
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
    Returns:
      enall (float) = reduced energy of lattice.
    """
    # calls halo exchange to get the neighbors
    halo_exchange(arr)

    ny = arr.shape[0] - 2
    nx = arr.shape[1] - 2

    en_local = 0.0
    for iy in range(1, ny + 1):
        for ix in range(1, nx + 1):
            en_local += one_energy_local(arr, ix, iy)

    return comm.allreduce(en_local, op=MPI.SUM)

# ============================================================

def get_order(arr, nmax):
    """
    Arguments:
      arr (float(ny+2,nx+2)) = local array including ghost cells;
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
    Returns:
      max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    # get local lattice size (excluding ghost cells)
    ny = arr.shape[0] - 2
    nx = arr.shape[1] - 2
    #
    # Generate a 3D unit vector for each cell (x,y) and
    # accumulate Qab locally, then reduce.
    #
    spins = arr[1:-1, 1:-1]
    lab = np.vstack(
        (
            np.cos(spins),
            np.sin(spins),
            np.zeros((ny, nx)),
        )
    ).reshape(3, nx * ny)

    Qab_local = np.zeros((3, 3), dtype=float)
    for v in lab.T:
        Qab_local += 3.0 * np.outer(v, v) - np.eye(3)

    # sum Qab across all ranks (then normalise once, globally)
    Qab = comm.allreduce(Qab_local, op=MPI.SUM)
    Qab /= (2.0 * nmax * nmax)
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

    ny = arr.shape[0] - 2
    nx = arr.shape[1] - 2

    #
    # Pre-compute some random numbers.  This is faster than
    # using lots of individual calls.  "scale" sets the width
    # of the distribution for the angle changes - increases
    # with temperature.
    scale = 0.1 + Ts
    accept = 0
    xran = np.random.randint(1, high=nx + 1, size=(ny, nx))
    yran = np.random.randint(1, high=ny + 1, size=(ny, nx))
    aran = np.random.normal(scale=scale, size=(ny, nx))

    for j in range(ny):
        for i in range(nx):
            ix = xran[j, i]
            iy = yran[j, i]
            ang = aran[j, i]
            en0 = one_energy_local(arr, ix, iy)
            arr[iy, ix] += ang
            en1 = one_energy_local(arr, ix, iy)
            if en1 <= en0:
                accept += 1
            else:
            # Now apply the Monte Carlo test - compare
            # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = np.exp(-(en1 - en0) / Ts)

                if boltz >= np.random.uniform(0.0, 1.0):
                    accept += 1
                else:
                    arr[iy, ix] -= ang

    # reduce acceptance count across all ranks
    total_accept = comm.allreduce(accept, op=MPI.SUM)

    # total attempted moves globally (one attempt per local site)
    total_sites = (nx * ny) * nranks

    return total_accept / total_sites

# ============================================================
def gather_lattice(local, nmax, nx_local, ny_local):
    """
    Arguments:
      local (float(ny_local+2,nx_local+2)) = local lattice including ghost cells;
      nmax (int) = side length of the global square lattice.
    Description:
      Function to gather the interior lattice blocks from all MPI ranks
      onto the root process (rank 0).  Ghost cells are excluded.
      The gathered lattice is used for plotting and output.
    Returns:
      recvbuf (float(nmax,nmax)) = full lattice on rank 0;
      None on all other ranks.
    """

    # extract interior block to send to root
    sendbuf = local[1:-1, 1:-1].copy()

    recvbuf = None
    if rank == 0:
        recvbuf = np.empty((nmax, nmax), dtype=float)

        # Place rank 0's own block
        cy, cx = cart.Get_coords(0)  # coords are (row=y, col=x)
        y0 = cy * ny_local
        x0 = cx * nx_local
        recvbuf[y0 : y0 + ny_local, x0 : x0 + nx_local] = sendbuf

        # Receive blocks from other ranks and place them by Cartesian coords
        for r in range(1, nranks):
            block = np.empty((ny_local, nx_local), dtype=float)
            comm.Recv(block, source=r, tag=77)

            cy, cx = cart.Get_coords(r)
            y0 = cy * ny_local
            x0 = cx * nx_local
            recvbuf[y0 : y0 + ny_local, x0 : x0 + nx_local] = block
    else:
        comm.Send(sendbuf, dest=0, tag=77)

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
    # Ensure the global lattice can be evenly decomposed
    if (nmax % dims[0] != 0) or (nmax % dims[1] != 0):
        if rank == 0:
            print(
                "ERROR: SIZE must be divisible by process grid dims. "
                f"Got SIZE={nmax}, dims={dims}."
            )
        comm.Abort(2)

    # Local sizes: axis0 is rows (y), axis1 is cols (x).
    ny_local = nmax // dims[0]
    nx_local = nmax // dims[1]

    # Create and initialise lattice
    lattice = initdat_local(nx_local, ny_local)

    # Plot initial frame of lattice (rank 0 only, gathered global lattice)
    if rank == 0:
        full0 = gather_lattice(lattice, nmax, nx_local, ny_local)
        plotdat(full0, pflag, nmax)
    else:
        gather_lattice(lattice, nmax, nx_local, ny_local)  # participate

    # Create arrays to store energy, acceptance ratio and order parameter (rank 0 owns)
    energy = np.zeros(nsteps + 1, dtype=float) if rank == 0 else None
    ratio = np.zeros(nsteps + 1, dtype=float) if rank == 0 else None
    order = np.zeros(nsteps + 1, dtype=float) if rank == 0 else None

    # Set initial values in arrays
    e0 = all_energy(lattice)
    o0 = get_order(lattice, nmax)

    if rank == 0:
        energy[0] = e0
        ratio[0] = 0.5  # ideal value
        order[0] = o0

    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1, nsteps + 1):
        r = MC_step(lattice, temp)
        e = all_energy(lattice)
        o = get_order(lattice, nmax)

        if rank == 0:
            ratio[it] = r
            energy[it] = e
            order[it] = o

    final = time.time()
    runtime = final - initial

    # Final outputs (rank 0 only)
    if rank == 0:
        print(
            "{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(
                program, nmax, nsteps, temp, order[nsteps], runtime
            )
        )

        fullf = gather_lattice(lattice, nmax, nx_local, ny_local)
        savedat(fullf, nsteps, temp, runtime, ratio, energy, order, nmax)
        plotdat(fullf, pflag, nmax)
    else:
        gather_lattice(lattice, nmax, nx_local, ny_local)  # participate

#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == "__main__":
    # Make random streams differ across ranks
    np.random.seed((int(time.time()) + 1* rank) % (2**32))

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