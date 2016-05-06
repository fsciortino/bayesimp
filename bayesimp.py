from __future__ import division

from signal import signal, SIGPIPE, SIG_DFL
#Ignore SIG_PIPE and don't throw exceptions on it... (http://docs.python.org/library/signal.html)
signal(SIGPIPE, SIG_DFL)

import sys
# Override the system eqtools:
sys.path.insert(0, "/home/markchil/codes/efit/development/EqTools")

import os
from distutils.dir_util import copy_tree
import subprocess
import scipy
import scipy.io
import scipy.interpolate
import scipy.optimize
import numpy.random
import MDSplus
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.gridspec as mplgs
import matplotlib.widgets as mplw
from mpl_toolkits.axes_grid1 import make_axes_locatable
import profiletools
import profiletools.gui
import gptools
import eqtools
import Tkinter as tk
import re
import glob
import copy
import cPickle as pkl
import collections
import pexpect
import time as time_
import shutil
import tempfile
import multiprocessing
import emcee
from emcee.interruptible_pool import InterruptiblePool
import smtplib
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.utils import COMMASPACE, formatdate
import warnings
import TRIPPy
import TRIPPy.XTOMO
from gptools.splines import spev
import itertools
try:
    import PyGMO
    _have_PyGMO = True
except ImportError:
    warnings.warn("Could not import PyGMO!", RuntimeWarning)
    _have_PyGMO = False
import periodictable
import lines
import nlopt
import pymysql as MS
import traceback
from connect import get_connection

# Store the PID of the main thread:
MAIN_PID = os.getpid()

# List of directories for use in main thread:
MAIN_THREAD_DIRS = [None]

# Lock object to handle threads with PyGMO:
DIR_POOL_LOCK = multiprocessing.Lock()

# Regex used to split lists up. This will let the list be delimted by any
# non-numeric characters, where the decimal point and minus sign are NOT
# considered numeric.
LIST_REGEX = r'([0-9]+)[^0-9]*'

# Region of interest for Ca lines on XEUS (in nanometers):
XEUS_ROI = (1.8, 2.2)

# Locations of Ca 17+ lines (in nanometers, taken from pue#ca17.dat):
CA_17_LINES = (
    1.8683, 30.2400, 1.9775, 1.8727, 34.4828, 1.9632, 2.0289, 1.4080, 2.0122,
    1.4091, 1.4736, 1.2642, 1.9790, 1.4652, 1.4852, 1.2647, 1.3181, 1.4763,
    5.4271, 1.3112, 1.3228, 5.7753, 1.4739, 5.4428, 3.7731, 5.7653, 5.8442,
    5.6682, 1.3157, 3.9510, 5.6332, 3.7770, 5.8015, 3.9457, 3.8990, 1.3182,
    3.9404, 5.8320, 3.8812, 3.9210, 5.8370, 11.8177, 12.4743, 5.6734, 3.9644,
    12.6866, 12.4557, 11.8555, 5.7780, 12.2669, 3.9627, 3.9002, 12.6020, 12.6091,
    3.9517, 12.2001, 5.8190, 12.6265, 12.4970, 12.4883, 3.9585, 12.2793, 12.4807,
    12.5836, 12.5252, 12.5256, 12.5007, 107.5003, 12.5127, 124.3039, 260.0374,
    301.5136, 229.1056, 512.7942, 286.7219, 595.2381, 321.8228, 545.2265,
    682.1748, 1070.0909, 1338.6881, 766.2248, 1505.1174, 3374.9577, 4644.6817,
    6583.2783, 9090.9089, 7380.0736, 14430.0141
)

# Locations of Ca 16+ lines (in nanometers):
CA_16_LINES = (19.2858,)

# Combined Ca 16+, 17+ lines (in nanometers):
CA_LINES = scipy.asarray(CA_17_LINES + CA_16_LINES)

# Threshold for size of HiReX-SR errorbar to reject:
HIREX_THRESH = 0.05

# Template for the Ca.atomdat file:
CA_ATOMDAT_TEMPLATE = """cv   main ion brems       SXR     spectral brems      thermal CX    
             0             0            0                 0

cv  diagnostic lines
            1        

cc begin atomicData
acd:acd85_ca.dat           recombination
scd:scd85_ca.dat           ionisation
prb:prb00_ca.dat           continuum radiation 
plt:plt00_ca.dat           line radiation

cc end atomic Data   


********************Diagnostic Lines******************** 

cd     excitation     recombination   charge exchange
          1                 0               0

cd   # of lines
    {num_lines:d}

cd   charge of ion      wavelength(A)     half width of window(A)       file extension
{line_spec}
"""

# Template for each line of the line_spec in Ca.atomdat:
LINE_SPEC_TEMPLATE = "    {charge:d}    {wavelength:.3f}    {halfwidth:.3f}    'pue'\n"

# Global variable to keep track of the currently-running IDL session. Handled
# with global variables so that it works with threading and provides
# persistence of the IDL session.
IDL_SESSION = None

# Global variable to keep track of the working directory for the current thread.
WORKING_DIR = None

# Global variable to keep track of the master directory to copy from.
MASTER_DIR = None

# Global variable to keep track of number of calls to STRAHL.
NUM_STRAHL_CALLS = 0

# Translate status codes into human-readable strings:
OPT_STATUS = {
    1: "success",
    2: "stopval reached",
    3: "ftol reached",
    4: "xtol reached",
    5: "unconverged: maxeval reached",
    6: "unconverged: maxtime reached",
    -1: "failure",
    -2: "failure: invalid args",
    -3: "failure: out of memory",
    -4: "failure: roundoff limited",
    -5: "failure: forced stop",
    -100: "unconverged: no work done",
    0: "unconverged: in progress",
    10: "bayesimp failure"
}

def test_lock(*args, **kwargs):
    """Test the thread-safety of putting the lock in the module namespace.
    
    Looks like it works!
    """
    with DIR_POOL_LOCK:
        time_.sleep(5)
        print("done.")
    
    return 0

def get_idl_session():
    """Launch an IDL session and store it in the global variable IDL_SESSION.
    
    The global variable is used to provide persistence and support for
    multiprocessing.
    
    If there is no active IDL session when called, it launches IDL, compiles
    execute_strahl.pro and restores run_data to the interactive workspace. It
    also checks to see if view_data.sav exists and, if so, loads that.
    """
    global IDL_SESSION
    if IDL_SESSION is None:
        IDL_SESSION = pexpect.spawn('idl')
        # Use the following line for debugging:
        # IDL_SESSION.logfile = sys.stdout
        IDL_SESSION.expect('IDL> ')
        IDL_SESSION.sendline('.compile execute_strahl')
        IDL_SESSION.expect('IDL> ')
        IDL_SESSION.sendline('run_data = load_run_data()')
        IDL_SESSION.expect('IDL> ')
        IDL_SESSION.sendline('if file_test("view_data.sav") then restore, "view_data.sav"')
        IDL_SESSION.expect('IDL> ')
    return IDL_SESSION

def acquire_working_dir():
    """Get the first available working directory. If none is available, create one.
    """
    # Some clever trickery is needed to deal with execution in the main thread
    # for connected topologies, since PaGMO seems to spawn way too many at once
    # and there are concurrent access issues. The solution adopted is to use
    # MAIN_THREAD_DIRS as a sentinel. If it is empty, the main directory is in
    # use. If it has one element, the main directory can be used. So, to acquire
    # the directory you pop (an atomic operation) and to release it you append
    # (also an atomic operation).
    global WORKING_DIR
    global MASTER_DIR
    
    if os.getpid() == MAIN_PID:
        # This will hammer away indefinitely until the directory is acquired,
        # trying every 10ms.
        while True:
            try:
                status = MAIN_THREAD_DIRS.pop()
            except IndexError:
                time_.sleep(1e-2)
            else:
                WORKING_DIR = MASTER_DIR
                os.chdir(WORKING_DIR)
                return
    else:
        global DIR_POOL_LOCK
        
        with DIR_POOL_LOCK:
            with open(os.path.join(MASTER_DIR, 'working_dirs.txt'), 'r') as f:
                lines = f.read().splitlines(True)
            # Handle case where all directories are already taken:
            if len(lines) == 0:
                WORKING_DIR = tempfile.mkdtemp(prefix='bayesimp')
                os.chdir(WORKING_DIR)
                copy_tree(MASTER_DIR, WORKING_DIR)
            else:
                WORKING_DIR = lines[0][:-1]
                os.chdir(WORKING_DIR)
                with open(os.path.join(MASTER_DIR, 'working_dirs.txt'), 'w') as f:
                    f.writelines(lines[1:])

def release_working_dir():
    """Release the current working directory, adding its name back onto the list of available ones.
    """
    global WORKING_DIR
    global MASTER_DIR
    
    # Some clever trickery is needed to deal with execution in the main thread
    # for connected topologies, since PaGMO seems to spawn way too many at once
    # and there are concurrent access issues. The solution adopted is to use
    # MAIN_THREAD_DIRS as a sentinel. If it is empty, the main directory is in
    # use. If it has one element, the main directory can be used. So, to acquire
    # the directory you pop (an atomic operation) and to release it you append
    # (also an atomic operation).
    if os.getpid() == MAIN_PID:
        # Append is atomic, so I can safely just push this back on:
        MAIN_THREAD_DIRS.append(None)
    else:
        global DIR_POOL_LOCK
        
        with DIR_POOL_LOCK:
            with open(os.path.join(MASTER_DIR, 'working_dirs.txt'), 'a') as f:
                f.write(WORKING_DIR + '\n')
        
        os.chdir(MASTER_DIR)

def setup_working_dir(*args, **kwargs):
    """Setup a temporary working directory, and store its name in WORKING_DIR.
    """
    global WORKING_DIR
    global MASTER_DIR
    global DIR_POOL_LOCK
    # global IDL_SESSION
    global NUM_STRAHL_CALLS
    
    assert WORKING_DIR is None
    
    MASTER_DIR = os.getcwd()
    WORKING_DIR = tempfile.mkdtemp(prefix='bayesimp')
    NUM_STRAHL_CALLS = 0
    
    print("Setting up %s..." % (WORKING_DIR,))
    
    os.chdir(WORKING_DIR)
    copy_tree(MASTER_DIR, WORKING_DIR)
    
    with DIR_POOL_LOCK:
        with open(os.path.join(MASTER_DIR, 'working_dirs.txt'), 'a') as f:
            f.write(WORKING_DIR + '\n')
    
    # Also open an IDL session:
    # print("Launching IDL session...")
    # assert IDL_SESSION is None
    # idl = get_idl_session()
    
    print("Ready to work!")

def cleanup_working_dir(*args, **kwargs):
    """Remove the WORKING_DIR. This should be called in each worker when closing out a pool.
    """
    global WORKING_DIR
    global MASTER_DIR
    
    if WORKING_DIR is not None:
        print("Cleaning up %s..." % (WORKING_DIR,))
        # Also quit the active IDL session:
        # get_idl_session().sendline('exit')
        
        os.chdir(MASTER_DIR)
        shutil.rmtree(WORKING_DIR)
        WORKING_DIR = None

def finalize_pool(pool):
    """Have each worker in a pool clean up its working directory.
    """
    # Use a factor of 4 to ensure each worker gets called, since I don't
    # know a better way to ensure this.
    dum = [1] * (4 * pool._processes)
    pool.map(cleanup_working_dir, dum)

def make_pool(num_proc=None):
    """Create and return a pool.
    """
    global MASTER_DIR
    global DIR_POOL_LOCK
    
    MASTER_DIR = os.getcwd()
    
    # Blank out the list of available directories:
    with DIR_POOL_LOCK:
        f = open(os.path.join(MASTER_DIR, 'working_dirs.txt'), 'w')
        f.close()
    if num_proc is None:
        num_proc = multiprocessing.cpu_count()
    # Close out the IDL session before creating the pools, since this
    # seems to cause some issues.
    global IDL_SESSION
    if IDL_SESSION is not None:
        IDL_SESSION.sendline('exit')
    IDL_SESSION = None
    pool = InterruptiblePool(processes=num_proc, initializer=setup_working_dir)
    
    return pool

class Run(object):
    """Class to load and run bayesimp/STRAHL.
    
    Must be run from the directory containing bayesimp!
    
    If the directory strahl_<SHOT>_<VERSION> does not exist, creates it.
    
    Most of the parameters are required, but are used with keywords to make
    setup simpler.
    
    Parameters
    ----------
    shot : int
        The shot number to analyze.
    version : int
        The version of the analysis to perform. Default is 0.
    time_1 : float
        The start time of the simulation.
    time_2 : float
        The end time of the simulation.
    injections : list of :py:class:`Injection`
        Skeleton objects describing the injections. These will be filled in as
        part of building the object.
    tht : int
        The THT to use. Default is 0.
    line : int
        The HiReX-SR line to use. Default is 6. OTHER LINES ARE NOT SUPPORTED
        YET!
    Te_args : list of str
        List of command-line arguments to pass to gpfit when fitting the Te
        profile.
    ne_args : list of str
        List of command-line arguments to pass to gpfit when fitting the ne
        profile.
    debug_plots : int
        Set to 0 to suppress superfluous plots. Set to 1 to enable some plots.
        Set to 2 to enable all plots. Default is 0.
    num_eig_D : int
        The number of eigenvalues/free parameters to use for the D profile.
    num_eig_V : int
        The number of eigenvalues/free parameters to use for the V profile.
    D_hyperprior : :py:class:`gptools.JointPrior` instance
        The hyperprior to use for the hyperparameters of the D profile. Note
        that this will override the existing hyperprior of `k_D`.
    V_hyperprior : :py:class:`gptools.JointPrior` instance
        The hyperprior to use for the hyperparameters of the V profile. Note
        that this will override the existing hyperprior of `k_V`.
    k_D : :py:class:`gptools.Kernel` instance
        The covariance kernel to use for the D profile. Default is to use a
        squared exponential (SE) kernel.
    k_V : :py:class:`gptools.Kernel` instance
        The covariance kernel to use for the V profile. Default is to use a
        squared exponential (SE) kernel.
    mu_D : :py:class:`gptools.MeanFunction`
        The mean function to use for the D profile. Default is to use a constant
        mean function for which the value is selected during the inference.
    clusters : bool
        Whether or not the flat "cluster" region should be included in the
        source model.
    roa_grid : array of float
        r/a grid to evaluate the ne, Te profiles on.
    roa_grid_DV : array of float
        r/a grid to evaluate the D, V profiles on.
    source_prior : :py:class:`gptools.JointPrior` instance
        The prior distribution for the parameters of the model source function.
    nt_source : int
        The number of time steps to evaluate the source function at.
    source_file : str
        If present, this is a path to a properly-formatted source file to use
        instead of the source model. This overrides the other options related
        to sources (though note that source_prior still acts as the prior for
        the temporal shift applied to the data, and hence should be univariate).
    method : {'GP', 'spline', 'linterp'}
        The method to use when evaluating the D, V profiles.
        
            * If 'GP', a Gaussian process prior will be used for the V profile
              and the logarithm of the D profile. `num_eig_D` and `num_eig_V`
              will be the number of terms kept in a truncated Karhunen-Loeve
              expansion of the GP prior.
            * If 'spline', B-splines will be used for the D and V profiles.
              `num_eig_D` and `num_eig_V` will then be the number of free
              coefficients in the respective splines. Because there is a slope
              constraint on D and a value constraint on V, this is one fewer
              than the actual number of spline coefficients. The number of knots
              is then num_eig - k + 2.
            * If 'linterp', piecewise linear functions will be used for the D
              and V profiles. `num_eig_D` and `num_eig_V` will then be the
              number of free values which are then linearly-interpolated.
              Because there is a slope constraint on D and a value constraint on
              V, the number of knots is num_eig + 1.
        
    knotgrid_D : array of float
        The (fixed) knots to use for the D profile.
    knotgrid_V : array of float
        The (fixed) knots to use for the V profile.
    free_knots : bool
        If True, the (internal) knot locations will be taken to be free
        parameters included in the inference. There will always be a knot at 0
        and a knot at 1. Default is to use fixed knots.
    spline_k_D : int
        The spline order to use for the D profile with method = 'spline'.
        Default is 3 (cubic spline).
    spline_k_V : int
        The spline order to use for the V profile with method = 'spline'.
        Default is 3 (cubic spline).
    include_loweus : bool
        If True, the data from the LoWEUS spectrometer will be included in the
        likelihood. Otherwise, LoWEUS will only be evaluated to compare the
        brightness. Note that you should set this True for shots which have no
        LoWEUS data.
    use_scaling : bool
        If True, a scale factor applied to each diagnostic signal will be
        included as a free parameter. This can help deal with issues in the
        normalization. Note that the signals are still normalized, so this
        factor should end up being very close to unity for each signal. Default
        is False (only normalize).
    sort_knots : bool
        If True, the knots will be sorted when splitting the params. Default is
        False (don't sort knots, unsorted knots are treated as infeasible cases).
    """
    def __init__(
            self,
            shot=0,
            version=0,
            time_1=0.0,
            time_2=0.0,
            injections=[],
            tht=0,
            line=6,
            Te_args=[],
            ne_args=[],
            debug_plots=0,
            num_eig_D=5,
            num_eig_V=5,
            D_hyperprior=None,
            V_hyperprior=None,
            k_D=None,
            k_V=None,
            mu_D=None,
            clusters=False,
            roa_grid=scipy.linspace(0, 1.2, 100),
            roa_grid_DV=scipy.linspace(0, 1.05, 100),
            source_prior=None,
            nt_source=200,
            source_file=None,
            method='GP',
            knotgrid_D=None,
            knotgrid_V=None,
            free_knots=False,
            spline_k_D=3,
            spline_k_V=3,
            include_loweus=False,
            use_scaling=False,
            sort_knots=False
        ):
        
        global MASTER_DIR
        
        # TODO: Add some command line flags to override/redo as needed.
        # TODO: Make it recognize when the settings have changed and update the
        # the right portions accordingly!
        
        self._ll_normalization = None
        self._ar_ll_normalization = None
        
        self.shot = shot
        self.version = version
        self.time_1 = time_1
        self.time_2 = time_2
        self.injections = injections
        self.tht = tht
        self.line = line
        self.Te_args = Te_args
        self.ne_args = ne_args
        self.debug_plots = debug_plots
        
        self.include_loweus = include_loweus
        
        self.free_knots = free_knots
        self.sort_knots = sort_knots
        
        self.use_scaling = use_scaling
        
        if method == 'spline':
            self.spline_k_D = spline_k_D
            self.spline_k_V = spline_k_V
        
        self.method = method
        if not self.free_knots:
            if self.method == 'spline':
                if knotgrid_D is None:
                    knotgrid_D = scipy.linspace(0, 1.2, num_eig_D - self.spline_k_D + 2)
                if knotgrid_V is None:
                    knotgrid_V = scipy.linspace(0, 1.2, num_eig_V - self.spline_k_D + 2)
            elif self.method == 'linterp':
                if knotgrid_D is None:
                    knotgrid_D = scipy.linspace(0, 1.2, num_eig_D + 1)
                if knotgrid_V is None:
                    knotgrid_V = scipy.linspace(0, 1.2, num_eig_V + 1)
            
            self.knotgrid_D = knotgrid_D
            self.knotgrid_V = knotgrid_V
        
        self.num_eig_D = num_eig_D
        self.num_eig_V = num_eig_V
        if self.method == 'GP':
            if k_D is None:
                k_D = gptools.SquaredExponentialKernel()
            if k_V is None:
                k_V = gptools.SquaredExponentialKernel()
            if D_hyperprior is None:
                D_hyperprior = k_D.hyperprior
            else:
                k_D.hyperprior = D_hyperprior
            if V_hyperprior is None:
                V_hyperprior = k_V.hyperprior
            else:
                k_V.hyperprior = V_hyperprior
            self.D_hyperprior = D_hyperprior
            self.V_hyperprior = V_hyperprior
            self.k_D = k_D
            self.mu_D = mu_D
            self.k_V = k_V
        
        self.source_file = source_file
        if source_file is None:
            self.clusters = clusters
            self.nt_source = nt_source
        self.roa_grid = roa_grid
        self.roa_grid_DV = roa_grid_DV
        # Convert the psinorm grids to r/a:
        self.efit_tree = eqtools.CModEFITTree(self.shot)
        self.psinorm_grid = self.efit_tree.roa2psinorm(
            self.roa_grid,
            (self.time_1 + self.time_2) / 2.0
        )
        # In case there is a NaN:
        self.psinorm_grid[0] = 0.0
        
        self.psinorm_grid_DV = self.efit_tree.roa2psinorm(
            self.roa_grid_DV,
            (self.time_1 + self.time_2) / 2.0
        )
        # In case there is a NaN:
        self.psinorm_grid_DV[0] = 0.0
        
        if source_prior is None:
            if source_file is not None:
                # Offsets for HiReX, XEUS/LoWEUS and XTOMO.
                source_prior = gptools.NormalJointPrior([0, 0, 0], [2e-3, 2e-3, 2e-3])
            # TODO: This code is stale and needs to be updated!
            elif clusters:
                source_prior = (
                    gptools.UniformJointPrior([(-2e-3, 2e-3)]) *
                    gptools.GammaJointPrior(
                        [1.0] * 6,
                        [0.1e-3, 1, 1.0e-3, 3, 20e-3, 1e-3]
                    )
                )
            else:  # no source_file, no clusters:
                source_prior = (
                    gptools.UniformJointPrior([(-2e-3, 2e-3)]) *
                    gptools.GammaJointPrior(
                        [1.0] * 4,
                        [0.1e-3, 1, 1.0e-3, 3]
                    )
                )
        self.source_prior = source_prior
        
        # If a STRAHL directory doesn't exist yet, create one and set it up:
        current_dir = os.getcwd()
        strahl_dir = os.path.join(current_dir, self.working_dir)
        if not (os.path.isdir(strahl_dir) and os.path.exists(strahl_dir)):
            self.setup_files()
        else:
            print("STRAHL directory %s is already in place." % (strahl_dir,))
            os.chdir(strahl_dir)
        
        MASTER_DIR = os.getcwd()
        
        # Create the strahl.control file:
        self.write_control()
        
        # Load run data into Python, save the processed data for later use:
        print("Loading run data...")
        try:
            with open('run_data.pkl', 'rb') as f:
                self.run_data = pkl.load(f)
            print("Loaded run data from run_data.pkl.")
        except IOError:
            self.run_data = RunData(self)
            with open('run_data.pkl', 'wb') as f:
                pkl.dump(self.run_data, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        # Compute and store the view data:
        if not os.path.isfile('view_data.pkl') or not os.path.isfile('xtomo_view_data.pkl'):
            print("Finding view data...")
            self.compute_view_data()
        else:
            with open('view_data.pkl', 'rb') as f:
                self.weights = pkl.load(f)
            with open('ar_view_data.pkl', 'rb') as f:
                self.ar_weights = pkl.load(f)
            with open('xtomo_view_data.pkl', 'rb') as f:
                self.xtomo_weights = pkl.load(f)
            print("Loaded view data from view_data.pkl.")
        
        self._PEC = None
        self.load_PEC()
        
        self._Ar_PEC = None
        self.load_Ar_PEC()
        
        self.atomdat = read_atomdat('Ca.atomdat')
        self.Ar_atomdat = read_atomdat('Ar.atomdat')
        
        self.atdata = lines.read_atdata()
        self.sindat = lines.read_sindat()
        with open('Be_50_um_PEC.pkl', 'rb') as f:
            self.filter_trans = pkl.load(f)
    
    def eval_DV(self, params, plot=False, lc=None, label=None):
        """Evaluate the D, V profiles for the given parameters.
        
        Parameters
        ----------
        params : array of float
            The parameters to evaluate at.
        plot : bool, optional
            If True, a plot of D and V will be produced. Default is False.
        """
        eig_D, eig_V, knots_D, knots_V, hp_D, hp_mu_D, hp_V, param_scaling, param_source = self.split_params(params)
        
        if self.method == 'GP':
            D = scipy.exp(
                eval_profile(
                    self.roa_grid_DV,
                    self.k_D,
                    eig_D,
                    1,
                    params=scipy.concatenate((hp_D, hp_mu_D)),
                    mu=self.mu_D
                )
            )
            V = eval_profile(self.roa_grid_DV, self.k_V, eig_V, 0, params=hp_V)
        elif self.method == 'spline':
            if self.free_knots:
                knotgrid_D = scipy.concatenate(([self.roa_grid_DV.min()], knots_D, [self.roa_grid_DV.max()]))
                knotgrid_V = scipy.concatenate(([self.roa_grid_DV.min()], knots_V, [self.roa_grid_DV.max()]))
            else:
                knotgrid_D = self.knotgrid_D
                knotgrid_V = self.knotgrid_V
            D = spev(
                knotgrid_D,
                scipy.insert(eig_D, 0, eig_D[0]),
                self.spline_k_D,
                self.roa_grid_DV
            )
            # Hackishly attempt to prevent numerical issues with STRAHL:
            D[0] = D[1]
            V = spev(
                knotgrid_V,
                scipy.insert(eig_V, 0, 0.0),
                self.spline_k_V,
                self.roa_grid_DV
            )
        elif self.method == 'linterp':
            if self.free_knots:
                knotgrid_D = scipy.concatenate(([self.roa_grid_DV.min()], knots_D, [self.roa_grid_DV.max()]))
                knotgrid_V = scipy.concatenate(([self.roa_grid_DV.min()], knots_V, [self.roa_grid_DV.max()]))
            else:
                knotgrid_D = self.knotgrid_D
                knotgrid_V = self.knotgrid_V
            D = scipy.interpolate.InterpolatedUnivariateSpline(
                knotgrid_D,
                scipy.insert(eig_D, 0, eig_D[0]),
                k=1
            )(self.roa_grid_DV)
            V = scipy.interpolate.InterpolatedUnivariateSpline(
                knotgrid_V,
                scipy.insert(eig_V, 0, 0.0),
                k=1
            )(self.roa_grid_DV)
        else:
            raise ValueError("Illegal method '%s'!" % (self.method,))
        
        if plot:
            f = plt.figure()
            a_D = f.add_subplot(3, 1, 1)
            a_V = f.add_subplot(3, 1, 2, sharex=a_D)
            a_VD = f.add_subplot(3, 1, 3, sharex=a_D)
            a_D.set_xlabel('$r/a$')
            a_V.set_xlabel('$r/a$')
            a_VD.set_xlabel('$r/a$')
            a_D.set_ylabel('$D$ [m$^2$/s]')
            a_V.set_ylabel('$V$ [m/s]')
            a_VD.set_ylabel('$V/D$ [1/m]')
            
            a_D.plot(self.roa_grid_DV, D, color=lc, label=label)
            a_V.plot(self.roa_grid_DV, V, color=lc, label=label)
            a_VD.plot(self.roa_grid_DV, V / D, color=lc, label=label)
        
        return (D, V)
    
    def propagate_u(self, u, cov, nsamp=1000, debug_plots=False):
        r"""Propagate the uncertainties `cov` through :py:meth:`eval_DV`.
        
        Parameters
        ----------
        u : array of float, (`num_params`,)
            The parameters to evaluate at, mapped to :math:`[0, 1]` using the
            CDF.
        cov : array of float, (`num_params`, `num_params`)
            The covariance matrix (i.e., the inverse Hessian returned by the
            optimizer in typical applications) to use.
        nsamp : int, optional
            The number of Monte Carlo samples to take. Default is 1000.
        """
        u_samples = numpy.random.multivariate_normal(u, cov, size=nsamp)
        u_samples[u_samples > 1.0] = 1.0
        u_samples[u_samples < 0.0] = 0.0
        p = self.get_prior()
        D = scipy.zeros((nsamp, len(self.roa_grid_DV)))
        V = scipy.zeros((nsamp, len(self.roa_grid_DV)))
        for i, uv in enumerate(u_samples):
            D[i, :], V[i, :] = self.eval_DV(p.sample_u(uv))
        mu_D = scipy.mean(D, axis=0)
        mu_V = scipy.mean(V, axis=0)
        std_D = scipy.std(D, axis=0, ddof=1)
        std_V = scipy.std(V, axis=0, ddof=1)
        
        if debug_plots:
            f = plt.figure()
            a_D = f.add_subplot(2, 1, 1)
            a_V = f.add_subplot(2, 1, 2)
            D_test, V_test = self.eval_DV(p.sample_u(u))
            a_D.plot(self.roa_grid_DV, D_test, 'r')
            a_V.plot(self.roa_grid_DV, V_test, 'r')
            gptools.univariate_envelope_plot(
                self.roa_grid_DV,
                mu_D,
                std_D,
                ax=a_D,
                color='b'
            )
            gptools.univariate_envelope_plot(
                self.roa_grid_DV,
                mu_V,
                std_V,
                ax=a_V,
                color='b'
            )
            a_D.set_ylabel("$D$ [m$^2$/s]")
            a_V.set_ylabel("$V$ [m/s]")
            a_V.set_xlabel("$r/a$")
        
        return mu_D, std_D, mu_V, std_V
    
    def split_params(self, params):
        """Split the given param vector into its constituent parts.
        """
        params = scipy.asarray(params, dtype=float)
        
        # Try to avoid some stupid issues:
        params[params == scipy.inf] = 1e-100 * sys.float_info.max
        params[params == -scipy.inf] = -1e-100 * sys.float_info.max
        
        # Split up the params:
        eig_D = params[:self.num_eig_D]
        eig_V = params[self.num_eig_D:self.num_eig_D + self.num_eig_V]
        
        if self.method == 'GP':
            eig_D = scipy.atleast_2d(eig_D).T
            eig_V = scipy.atleast_2d(eig_V).T
            hp_D = params[
                self.num_eig_D + self.num_eig_V:
                self.num_eig_D + self.num_eig_V + self.k_D.num_free_params
            ]
            hp_mu_D = params[
                self.num_eig_D + self.num_eig_V + self.k_D.num_free_params:
                self.num_eig_D + self.num_eig_V + self.k_D.num_free_params +
                    self.mu_D.num_free_params
            ]
            hp_V = params[
                self.num_eig_D + self.num_eig_V + self.k_D.num_free_params +
                    self.mu_D.num_free_params:
                self.num_eig_D + self.num_eig_V + self.k_D.num_free_params +
                    self.mu_D.num_free_params + self.k_V.num_free_params
            ]
            if self.use_scaling:
                param_scaling = params[
                    self.num_eig_D + self.num_eig_V + self.k_D.num_free_params +
                        self.mu_D.num_free_params + self.k_V.num_free_params:
                    self.num_eig_D + self.num_eig_V + self.k_D.num_free_params +
                        self.mu_D.num_free_params + self.k_V.num_free_params +
                        1 + self.run_data.vuv_signals_norm_combined.shape[0] +
                        len(
                            [k for k in self.run_data.xtomo_sig.keys() if self.run_data.xtomo_sig[k] is not None]
                        )
                ]
                param_source = params[
                    self.num_eig_D + self.num_eig_V + self.k_D.num_free_params +
                        self.mu_D.num_free_params + self.k_V.num_free_params +
                        1 + self.run_data.vuv_signals_norm_combined.shape[0] +
                        len(
                            [k for k in self.run_data.xtomo_sig.keys() if self.run_data.xtomo_sig[k] is not None]
                        ):
                ]
            else:
                param_scaling = []
                param_source = params[
                    self.num_eig_D + self.num_eig_V + self.k_D.num_free_params +
                        self.mu_D.num_free_params + self.k_V.num_free_params:
                ]
        else:
            hp_D = []
            hp_mu_D = []
            hp_V = []
            if self.free_knots:
                if self.method == 'spline':
                    knots_D = params[
                        self.num_eig_D + self.num_eig_V:
                        self.num_eig_D + self.num_eig_V + self.num_eig_D -
                            self.spline_k_D
                    ]
                    knots_V = params[
                        self.num_eig_D + self.num_eig_V + self.num_eig_D -
                            self.spline_k_D:
                        self.num_eig_D + self.num_eig_V + self.num_eig_D -
                            self.spline_k_D + self.num_eig_V - self.spline_k_V
                    ]
                    if self.use_scaling:
                        param_scaling = params[
                            self.num_eig_D + self.num_eig_V + self.num_eig_D -
                                self.spline_k_D + self.num_eig_V - self.spline_k_V:
                            self.num_eig_D + self.num_eig_V + self.num_eig_D -
                                self.spline_k_D + self.num_eig_V - self.spline_k_V +
                                1 + self.run_data.vuv_signals_norm_combined.shape[0] +
                                len(
                                    [k for k in self.run_data.xtomo_sig.keys() if self.run_data.xtomo_sig[k] is not None]
                                )
                        ]
                        param_source = params[
                            self.num_eig_D + self.num_eig_V + self.num_eig_D -
                                self.spline_k_D + self.num_eig_V - self.spline_k_V +
                                1 + self.run_data.vuv_signals_norm_combined.shape[0] +
                                len(
                                    [k for k in self.run_data.xtomo_sig.keys() if self.run_data.xtomo_sig[k] is not None]
                                ):
                        ]
                    else:
                        param_scaling = []
                        param_source = params[
                            self.num_eig_D + self.num_eig_V + self.num_eig_D -
                                self.spline_k_D + self.num_eig_V - self.spline_k_V:
                        ]
                elif self.method == 'linterp':
                    knots_D = params[
                        self.num_eig_D + self.num_eig_V:
                        self.num_eig_D + self.num_eig_V + self.num_eig_D - 1
                    ]
                    knots_V = params[
                        self.num_eig_D + self.num_eig_V + self.num_eig_D - 1:
                        self.num_eig_D + self.num_eig_V + self.num_eig_D - 1 +
                            self.num_eig_V - 1
                    ]
                    if self.use_scaling:
                        param_scaling = params[
                            self.num_eig_D + self.num_eig_V + self.num_eig_D - 1 +
                                self.num_eig_V - 1:
                            self.num_eig_D + self.num_eig_V + self.num_eig_D - 1 +
                                self.num_eig_V - 1 +
                                1 + self.run_data.vuv_signals_norm_combined.shape[0] +
                                len(
                                    [k for k in self.run_data.xtomo_sig.keys() if self.run_data.xtomo_sig[k] is not None]
                                )
                        ]
                        param_source = params[
                            self.num_eig_D + self.num_eig_V + self.num_eig_D - 1 +
                                self.num_eig_V - 1 +
                                1 + self.run_data.vuv_signals_norm_combined.shape[0] +
                                len(
                                    [k for k in self.run_data.xtomo_sig.keys() if self.run_data.xtomo_sig[k] is not None]
                                ):
                        ]
                    else:
                        param_scaling = []
                        param_source = params[
                            self.num_eig_D + self.num_eig_V + self.num_eig_D - 1 +
                                self.num_eig_V - 1:
                        ]
                if self.sort_knots:
                    knots_D = scipy.sort(knots_D)
                    knots_V = scipy.sort(knots_V)
            else:
                knots_D = []
                knots_V = []
                if self.use_scaling:
                    param_scaling = params[
                        self.num_eig_D + self.num_eig_V:
                        self.num_eig_D + self.num_eig_V +
                            1 + self.run_data.vuv_signals_norm_combined.shape[0] +
                            len(
                                [k for k in self.run_data.xtomo_sig.keys() if self.run_data.xtomo_sig[k] is not None]
                            )
                    ]
                    param_source = params[
                        self.num_eig_D + self.num_eig_V +
                            1 + self.run_data.vuv_signals_norm_combined.shape[0] +
                            len(
                                [k for k in self.run_data.xtomo_sig.keys() if self.run_data.xtomo_sig[k] is not None]
                            ):
                    ]
                else:
                    param_scaling = []
                    param_source = params[self.num_eig_D + self.num_eig_V:]
        
        # Fudge the source times and scaling factors since this seems to go
        # crazy:
        param_scaling[param_scaling > 1e3] = 1e3
        param_scaling[param_scaling < 1e-3] = 1e-3
        param_source[param_source > 1e3] = 1e3
        param_source[param_source < -1e3] = -1e3
        
        return (eig_D, eig_V, knots_D, knots_V, hp_D, hp_mu_D, hp_V, param_scaling, param_source)
    
    def get_prior(self):
        """Return the prior distribution.
        
        This is a :py:class:`gptools.JointPrior` instance -- when called as a
        function, it returns the log-probability.
        """
        D_lb = 0.0
        D_ub = 30.0  # was 50
        # D_ub = 50.0  # was 50
        V_lb = -200.0
        V_ub = 200.0
        V_lb_outer = -200.0
        # V_lb_outer = -1000.0  # was -1000
        # V_lb_outer = -500.0  # was -500
        
        if self.method == 'GP':
            prior = (
                gptools.NormalJointPrior(
                    [0.0] * (self.num_eig_D + self.num_eig_V),
                    [1.0] * (self.num_eig_D + self.num_eig_V)
                ) *
                self.k_D.hyperprior *
                self.mu_D.hyperprior *
                self.k_V.hyperprior
            )
        else:
            prior = gptools.UniformJointPrior(
                [(D_lb, D_ub)] * self.num_eig_D +
                [(V_lb, V_ub)] * (self.num_eig_V - 1) +
                [(V_lb_outer, V_ub)]
            )
        if self.free_knots:
            if self.method == 'spline':
                k_D = self.spline_k_D
                k_V = self.spline_k_V
            elif self.method == 'linterp':
                k_D = 1
                k_V = 1
            if self.sort_knots:
                prior = prior * gptools.UniformJointPrior(
                    [(self.roa_grid_DV.min(), self.roa_grid_DV.max())] * (self.num_eig_D - k_D + self.num_eig_V - k_V)
                )
            else:
                prior = prior * (
                    gptools.SortedUniformJointPrior(
                        self.num_eig_D - k_D,
                        self.roa_grid_DV.min(),
                        self.roa_grid_DV.max()
                    ) *
                    gptools.SortedUniformJointPrior(
                        self.num_eig_V - k_V,
                        self.roa_grid_DV.min(),
                        self.roa_grid_DV.max()
                    )
                )
        
        if self.use_scaling:
            prior = prior * gptools.GammaJointPriorAlt(
                [1.0] * (
                    1 + self.run_data.vuv_signals_norm_combined.shape[0] +
                    len(
                        [k for k in self.run_data.xtomo_sig.keys() if self.run_data.xtomo_sig[k] is not None]
                    )
                ),
                [0.1] * (
                    1 + self.run_data.vuv_signals_norm_combined.shape[0] +
                    len(
                        [k for k in self.run_data.xtomo_sig.keys() if self.run_data.xtomo_sig[k] is not None]
                    )
                )
            )
        
        prior = prior * self.source_prior
        
        return prior
    
    def DV2cs_den(
            self,
            params,
            explicit_D=None,
            explicit_D_grid=None,
            explicit_V=None,
            explicit_V_grid=None,
            steady_ar=None,
            debug_plots=False,
            no_write=False,
            no_strahl=False,
            compute_view_data=False
        ):
        """Calls STRAHL with the given parameters and returns the charge state densities.
        
        If evaluation of the profiles from the given parameters fails, returns a
        single NaN. This failure can either take the form of :py:meth:`eval_DV`
        raising a :py:class:`ValueError`, or there being an Inf or NaN in the
        resulting profiles.
        
        Returns a single int of the return code if the call to STRAHL fails.
        
        Returns NaN if STRAHL fails due to max iterations.
        
        If everything works, returns a tuple of the following:
        
        * `cs_den`: Charge state densities. Array of float with shape
          (`n_time`, `n_cs`, `n_space`)
        * `sqrtpsinorm`: Square root of psinorm grid used for the results. Array
          of float with shape (`n_space`,)
        * `time`: Time grid used for the results. Array of float with shape
          (`n_time`,)
        * `ne`: Electron density profile used by STRAHL. Array of float with
          shape (`n_time`, `n_space`).
        * `Te`: Electron temperature profile used by STRAHL. Array of float with
          shape (`n_time`, `n_space`).
        
        Parameters
        ----------
        params : array of float
            The parameters to use when evaluating the model. The order is:
            * eig_D: The eigenvalues to use when evaluating the D profile.
            * eig_V: The eigenvalues to use when evaluating the V profile.
            * param_D: The hyperparameters to use for the D profile.
            * param_mu_D: The hyperparameters to use for the mean function of
              the D profile.
            * param_V: The hyperparameters to use for the V profile.
            * knots_D: The knots of the D profile.
            * knots_V: The knots of the V profile.
            * scaling: The scaling factors for each diagnostic.
            * param_source: The parameters to use for the model source function.
        explicit_D : array of float, optional
            Explicit values of D to use. Overrides the profile which would have
            been obtained from the parameters in `params` (but the scalings/etc.
            from `params` are still used).
        explicit_D_grid : array of float, optional
            Grid of sqrtpsinorm which `explicit_D` is given on.
        explicit_V : array of float, optional
            Explicit values of V to use. Overrides the profile which would have
            been obtained from the parameters in `params` (but the scalings/etc.
            from `params` are still used).
        explicit_V_grid : array of float, optional
            Grid of sqrtpsinorm which `explicit_V` is given on.
        steady_ar : float, optional
            If present, will compute the steady-state (constant-source) Ar
            profiles for the given source instead of the time-evolving Ca
            profiles. Default is None.
        debug_plots : bool, optional
            If True, plots of the various steps will be generated. Default is
            False (do not produce plots).
        no_write : bool, optional
            If True, the STRAHL control files are not written. Default is False.
        no_strahl : bool, optional
            If True, STRAHL is not actually called (and the existing results
            file is evaluated). Used for debugging. Default is False.
        compute_view_data : bool, optional
            Set this to True to only compute the view_data.sav file. (Returns
            the sqrtpsinorm grid STRAHL uses.)
        """
        global NUM_STRAHL_CALLS
        
        eig_D, eig_V, knots_D, knots_V, hp_D, hp_mu_D, hp_V, param_scaling, param_source = self.split_params(params)
        
        if (explicit_D is None) or (explicit_V is None):
            try:
                D, V = self.eval_DV(params, plot=debug_plots)
            except ValueError:
                print("Failure evaluating profiles!")
                print(params)
                return scipy.nan
        
        # Get the correct grids, handle explicit D and V:
        if explicit_D is not None:
            D = explicit_D
            D_grid = explicit_D_grid
        else:
            D_grid = scipy.sqrt(self.psinorm_grid_DV)
        if explicit_V is not None:
            V = explicit_V
            V_grid = explicit_V_grid
        else:
            V_grid = scipy.sqrt(self.psinorm_grid_DV)
        
        # Check for bad values in D, V profiles:
        if scipy.isinf(D).any() or scipy.isnan(D).any():
            print("inf in D!")
            print(params)
            return scipy.nan
        if scipy.isinf(V).any() or scipy.isnan(V).any():
            print("inf in V!")
            print(params)
            return scipy.nan
        
        # Evaluate ne, Te:
        ne = self.run_data.ne_res['mean_val']
        Te = self.run_data.Te_res['mean_val']
        # HACK to get rid of negative values in ne, Te:
        ne[ne < 0.0] = 0.0
        Te[Te < 0.0] = 0.0
        
        # Now write the param and pp files, if required:
        if not no_write:
            # Need to override the start/end times of steady_ar is not None:
            if steady_ar is None:
                time_2_override = None
            else:
                time_2_override = self.time_1 + 0.2
            self.write_control(time_2_override=time_2_override)
            self.write_pp(
                scipy.sqrt(self.psinorm_grid),
                ne,
                Te,
                self.time_2 if steady_ar is None else time_2_override
            )
            self.write_param(
                D_grid,
                V_grid,
                D,
                V,
                # compute_NC=compute_NC,
                const_source=steady_ar,
                element='Ca' if steady_ar is None else 'Ar',
                time_2_override=time_2_override
            )
        
        # Now call STRAHL:
        try:
            if no_strahl:
                out = 'STRAHL not run!'
            else:
                command = ['./strahl', 'a', 'n']
                # The "n" disables STRAHL's calculation of radiation.
                NUM_STRAHL_CALLS += 1
                out = subprocess.check_output(command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print("STRAHL exited with error code %d." % (e.returncode))
            return e.returncode
        
        # Process the results:
        f = scipy.io.netcdf.netcdf_file('result/strahl_result.dat', 'r')
        sqrtpsinorm = scipy.asarray(f.variables['rho_poloidal_grid'][:], dtype=float)
        if compute_view_data:
            return sqrtpsinorm
        time = scipy.asarray(f.variables['time'][:], dtype=float)
        
        # Check to make sure it ran through:
        if time[-1] <= self.time_2 - 0.1 * (self.time_2 - self.time_1):
            print(time[-1])
            print(len(time))
            print("STRAHL failed (max iterations)!")
            print(params)
            return scipy.nan
        
        # cs_den has shape (n_time, n_cs, n_space)
        cs_den = scipy.asarray(f.variables['impurity_density'][:], dtype=float)
        
        # These are needed for subsequent calculations:
        ne = scipy.asarray(f.variables['electron_density'][:], dtype=float)
        Te = scipy.asarray(f.variables['electron_temperature'][:], dtype=float)
        
        if debug_plots:
            # Plot the charge state densities:
            slider_plot(
                sqrtpsinorm,
                time,
                scipy.rollaxis(cs_den.T, 1),
                xlabel=r'$\sqrt{\psi_n}$',
                ylabel=r'$t$ [s]',
                zlabel=r'$n$ [cm$^{-3}$]',
                labels=[str(i) for i in range(0, cs_den.shape[1])]
            )
        
        return cs_den, sqrtpsinorm, time, ne, Te
    
    def cs_den2dlines(self, params, cs_den, sqrtpsinorm, time, ne, Te, steady_ar=None, debug_plots=False):
        """Predicts the local emissivities that would arise from the given charge state densities.
        
        Parameters
        ----------
        params : array of float
            The parameters to use.
        cs_den : array of float, (`n_time`, `n_cs`, `n_space`)
            The charge state densities as computed by STRAHL.
        sqrtpsinorm : array of float, (`n_space`,)
            The square root of psinorm grid which `cs_den` is given on.
        time : array of float, (`n_time`,)
            The time grid which `cs_den` is given on.
        ne : array of float, (`n_time`, `n_space`)
            The electron density profile used by STRAHL.
        Te : array of float, (`n_time`, `n_space`)
            The electron temperature profile used by STRAHL.
        steady_ar : float, optional
            If None, compute for calcium. If a float, compute for argon.
        debug_plots : bool, optional
            If True, plots of the various steps will be generated. Default is
            False (do not produce plots).
        """
        eig_D, eig_V, knots_D, knots_V, hp_D, hp_mu_D, hp_V, param_scaling, param_source = self.split_params(params)
        
        atomdat = self.atomdat if steady_ar is None else self.Ar_atomdat
        # Put the SXR signal as the final entry in dlines.
        n_lines = len(atomdat[0]) + 1 if steady_ar is None else 1
        dlines = scipy.zeros((len(time), n_lines, len(sqrtpsinorm)))
        
        if steady_ar is None:
            for i, chg, cw, hw in zip(
                    range(0, len(atomdat[0])),
                    atomdat[0],
                    atomdat[1],
                    atomdat[2]
                ):
                dlines[:, i, :] = compute_emiss(
                    self.PEC[chg],
                    cw,
                    hw,
                    ne,
                    cs_den[:, chg, :],
                    Te
                )
            # Compute the emissivity seen through the core XTOMO filters:
            dlines[:, -1, :] = lines.compute_SXR(
                cs_den,
                ne,
                Te,
                self.atdata,
                self.sindat,
                self.filter_trans,
                self.PEC
            )
        else:
            # We need to add up the contributions to the z-line. These are
            # stored in the PEC dict in the charge of the state the line is
            # populated from.
            # Excitation:
            dlines[:, 0, :] = compute_emiss(
                self.Ar_PEC[16],
                4.0,
                0.1,
                ne,
                cs_den[:, 16, :],
                Te,
                no_ne=True
            )
            # Ionization:
            dlines[:, 0, :] += compute_emiss(
                self.Ar_PEC[15],
                4.0,
                0.1,
                ne,
                cs_den[:, 15, :],
                Te,
                no_ne=True
            )
            # Recombination:
            dlines[:, 0, :] += compute_emiss(
                self.Ar_PEC[17],
                4.0,
                0.1,
                ne,
                cs_den[:, 17, :],
                Te,
                no_ne=True
            )
        
        if debug_plots:
            # Plot the emissivity profiles:
            slider_plot(
                sqrtpsinorm,
                time,
                scipy.rollaxis(dlines.T, 1),
                xlabel=r'$\sqrt{\psi_n}$',
                ylabel=r'$t$ [s]',
                zlabel=r'$\epsilon$ [W/cm$^3$]',
                labels=[str(i) for i in range(0, dlines.shape[1])]
            )
        
        return dlines
    
    def dlines2sig(
            self,
            params,
            dlines,
            time,
            steady_ar=None,
            debug_plots=False,
            big_f=None,
            a_H=None,
            a_V=None,
            f_xtomo=None,
            a_xtomo=None,
            ar_f=None,
            ar_a=None,
            label='_nolegend_',
            lc='g',
            alpha=1.0
        ):
        """Computes the diagnostic signals corresponding to the given local emissivities.
        
        Returns a tuple of (`sbright`, `vbright`, `xtomobright`), where
            - `sbright` is the HiReX-SR brightnesses.
            - `vbright` is the VUV spectrometer brightness (with XEUS first).
            - `xtomobright` is a dictionary of XTOMO brightnesses.
        
        In all cases, the shapes are (`n_time`, `n_chan`).
        
        Parameters
        ----------
        params : array of float
            The parameters to use.
        dlines : array of float, (`n_time`, `n_lines`, `n_space`)
            The spatial profiles of local emissivities.
        time : array of float, (`n_time`,)
            The time grid which `dlines` is given on.
        steady_ar : float, optional
            If None, compute for calcium. If a float, compute for argon.
        debug_plots : bool, optional
            If True, plots of the various steps will be generated. Default is
            False (do not produce plots).
        big_f : Figure instance, optional
            The figure containing plots of each data source. If not present, a
            new figure is created by calling :py:meth:`RunData.plot_data`.
        a_H : list of Axes instances, optional
            The axes from `big_f` which contain the HiReX-SR data.
        a_V : list of Axes instances, optional
            The axes from `big_f` which contain the VUV spectrometer data.
        f_xtomo : dict of Figure instances, optional
            The figures containing plots of each XTOMO system. If not present,
            new figures are created by calling :py:meth:`RunData.plot_xtomo`.
        a_xtomo : dict of lists of Axes instances, optional
            The axes from `f_xtomo`.
        ar_f : Figure instance, optional
            The figure containing the argon data. If not present, a new figure
            is created by calling :py:meth:`RunData.plot_ar`.
        ar_a : Axes instance, optional
            The axes from `ar_f` to plot the argon profile on.
        label : str, optional
            Label to use for lines added to existing figures. Default is
            '_nolegend_'.
        lc : str, optional
            Line color to use for lines added to existing figures. Default is
            None.
        alpha : float, optional
            Transparency to use for lines added to existing figures. Default is
            1.0 (solid).
        """
        eig_D, eig_V, knots_D, knots_V, hp_D, hp_mu_D, hp_V, param_scaling, param_source = self.split_params(params)
        
        time = time - self.time_1
        time_s = time + param_source[0]
        time_v = time + param_source[1]
        time_xtomo = time + param_source[2]
        
        if steady_ar is None:
            # Compute the HiReX-SR brightness:
            sbright = self.weights[:self.run_data.hirex_pos.shape[0]].dot(
                dlines[:, self.run_data.hirex_line_idx, :].T
            ).T
            
            # Normalize the HiReX-SR signal:
            sbright = sbright / sbright.max()
            
            # Compute the XEUS brightness:
            xbright = []
            for i in self.run_data.xeus_line_idxs:
                xbright.append(
                    self.weights[self.run_data.hirex_pos.shape[0]].dot(
                        dlines[:, i, :].T
                    )
                )
            xbright = scipy.vstack(xbright).T
            
            # Compute the LoWEUS brightness:
            if self.run_data.loweus_line_idxs:
                lbright = []
                for i in self.run_data.loweus_line_idxs:
                    lbright.append(
                        self.weights[self.run_data.hirex_pos.shape[0] + 1].dot(
                            dlines[:, i, :].T
                        )
                    )
                lbright = scipy.vstack(lbright).T
            
            # Normalize the VUV signals:
            for k in xrange(0, xbright.shape[1]):
                xbright[:, k] = xbright[:, k] / xbright[:, k].max()
            if self.run_data.loweus_line_idxs:
                for k in xrange(0, lbright.shape[1]):
                    lbright[:, k] = lbright[:, k] / lbright[:, k].max()
            
            # Group VUV signals for output:
            if self.run_data.loweus_line_idxs:
                vbright = scipy.hstack((xbright, lbright))
            else:
                vbright = xbright
            
            # Compute the XTOMO brightness:
            xtomobright = {}
            for s in self.run_data.xtomo_signal_norm_combined.keys():
                if self.run_data.xtomo_signal_norm_combined[s] is not None:
                    xtomobright[s] = self.xtomo_weights[s].dot(
                        dlines[:, -1, :].T
                    ).T
                    # Normalize the XTOMO signal:
                    xtomobright[s] = xtomobright[s] / xtomobright[s].max()
            
            # Apply scalings:
            if self.use_scaling:
                sbright = sbright * param_scaling[0]
                for k in xrange(0, len(self.run_data.xeus_line_idxs) + len(self.run_data.loweus_line_idxs)):
                    vbright[:, k] *= param_scaling[k + 1]
                i = k + 2
                for k in self.run_data.xtomo_sig.keys():
                    if self.run_data.xtomo_sig[k] is not None:
                        xtomobright[k] *= param_scaling[i]
                        i += 1
        else:
            sbright = self.ar_weights.dot(dlines[:, 0, :].T).T
            # Normalize the HiReX-SR signal:
            sbright = sbright / sbright.max()
        
        # Big plots:
        if debug_plots or big_f is not None:
            # HiReX-SR and VUV plots:
            if big_f is None:
                if steady_ar is None:
                    big_f, a_H, a_V = self.run_data.plot_data()
                else:
                    # Make the figure to hold the time-series data:
                    big_f = plt.figure()
                    ncol = 6
                    nrow = int(scipy.ceil(1.0 * self.run_data.ar_signal.shape[1] / ncol))
                    gs = mplgs.GridSpec(nrow, ncol)
                    a_H = []
                    i_col = 0
                    i_row = 0
                    for k in xrange(0, self.run_data.ar_signal.shape[1]):
                        a_H.append(big_f.add_subplot(gs[i_row, i_col]))
                        i_col += 1
                        if i_col >= ncol:
                            i_col = 0
                            i_row += 1
                    
                    for k in xrange(0, len(a_H)):
                        a_H[k].set_xlabel('$t$ [s]')
                        a_H[k].set_ylabel('$b$ [AU]')
                        a_H[k].set_title('HiReX-SR chord %d' % (k,))
                        # a_H[k].set_ylim(bottom=0)
            for a, k in zip(a_H, range(0, len(a_H))):
                l = a.plot(time_s, sbright[:, k], color=lc, alpha=alpha)
            if steady_ar is None:
                for a, k in zip(a_V, range(0, len(a_V))):
                    l = a.plot(time_v, vbright[:, k], label=label, color=lc, alpha=alpha)
            big_f.canvas.draw()
            
            # XTOMO plots:
            if steady_ar is None:
                # Create the figure if necessary:
                if f_xtomo is None:
                    f_xtomo = {}
                    a_xtomo = {}
                    for k in self.run_data.xtomo_sig.keys():
                        if self.run_data.xtomo_sig[k] is not None:
                            f_xtomo[k], a_xtomo[k] = self.run_data.plot_xtomo(k, norm=True)
                for k, f in f_xtomo.iteritems():
                    for i in range(0, len(a_xtomo[k])):
                        a_xtomo[k][i].plot(time_xtomo, xtomobright[k][:, i], label=label, color=lc, alpha=alpha)
        
        return sbright, vbright, xtomobright
    
    def sig2diffs(self, params, sbright, vbright, xtomobright, time, steady_ar=None, no_diff=False):
        """Computes the individual diagnostic differences corresponding to the given signals.
        
        Parameters
        ----------
        params : array of float
            The parameters to use.
        sbright : array of float, (`n_time`, `n_chords`)
            The predicted HiReX-SR signals on the STRAHL timebase.
        vbright : array of float, (`n_time`, `n_lines`)
            The predicted VUV spectrometer signals (with XEUS first, LoWEUS
            second) on the STRAHL timebase.
        xtomobright : dict of arrays of float (`n_time`, `n_chords`)
            The predicted XTOMO signals on the STRAHL timebase. The keys of the
            dict should be the system indices as ints, the values should be
            arrays of float with shape (`n_time`, `n_chords`).
        time : array of float, (`n_time`,)
            The time grid which `dlines` is given on.
        steady_ar : float, optional
            If None, compute for calcium. If a float, compute for argon.
        no_diff : bool, optional
            If True, the difference with the experimental data is not taken --
            just the signals interpolated onto the diagnostic timebase are
            returned. Default is False (return differences).
        """
        eig_D, eig_V, knots_D, knots_V, hp_D, hp_mu_D, hp_V, param_scaling, param_source = self.split_params(params)
        
        time = time - self.time_1
        time_s = time + param_source[0]
        time_v = time + param_source[1]
        time_xtomo = time + param_source[2]
        
        # Interpolate sbright onto the HiReX-SR timebase:
        sbright_interp = scipy.zeros(
            (len(self.run_data.hirex_time_combined), sbright.shape[1])
        )
        # Use postinj to zero out before the injection:
        postinj = self.run_data.hirex_time_combined >= param_source[0]
        for k in xrange(0, sbright.shape[1]):
            sbright_interp[postinj, k] = scipy.interpolate.InterpolatedUnivariateSpline(
                time_s,
                sbright[:, k]
            )(self.run_data.hirex_time_combined[postinj])
        
        if steady_ar is None:
            # Interpolate vbright onto the VUV timebase(s):
            vbright_interp = scipy.zeros(
                (self.run_data.vuv_times_combined.shape[1], vbright.shape[1])
            )
            for k in xrange(0, vbright.shape[1]):
                postinj = self.run_data.vuv_times_combined[k, :] >= param_source[1]
                vbright_interp[postinj, k] = scipy.interpolate.InterpolatedUnivariateSpline(
                    time_v,
                    vbright[:, k]
                )(self.run_data.vuv_times_combined[k, postinj])
            
            # Interpolate xtomobright onto the XTOMO timebase(s):
            xtomobright_interp = {}
            for s, b in xtomobright.iteritems():
                xtomobright_interp[s] = scipy.zeros(
                    (len(self.run_data.xtomo_times_combined[s]), b.shape[1])
                )
                for k in xrange(0, b.shape[1]):
                    # NOTE: I am using a linear spline to speed this up, since
                    # there are so many points.
                    postinj = self.run_data.xtomo_times_combined[s] >= param_source[2]
                    xtomobright_interp[s][postinj, k] = scipy.interpolate.InterpolatedUnivariateSpline(
                        time_xtomo,
                        b[:, k],
                        k=1
                    )(self.run_data.xtomo_times_combined[s][postinj])
        
        # Convert to differences:
        # Weighting must be accomplished in diffs2ln_prob.
        if steady_ar is None:
            if no_diff:
                return sbright_interp, vbright_interp, xtomobright_interp
            else:
                sbright_diff = sbright_interp - self.run_data.hirex_signal_norm_combined
                vbright_diff = vbright_interp - self.run_data.vuv_signals_norm_combined.T
                xtomobright_diff = {}
                for s, b in xtomobright_interp.iteritems():
                    xtomobright_diff[s] = b - self.run_data.xtomo_signal_norm_combined[s].T
                return sbright_diff, vbright_diff, xtomobright_diff
        else:
            ar_mask = (self.run_data.ar_time >= self.time_1) & (self.run_data.ar_time <= self.time_2)
            ar_signal = self.run_data.ar_signal[ar_mask, :]
            ar_sim = scipy.tile(sbright[-1, :], (ar_signal.shape[0], 1))
            sbright_diff = ar_sim - ar_signal
            
            return sbright_diff, None, None
    
    def diffs2ln_prob(
            self,
            params,
            sbright_diff,
            vbright_diff,
            xtomobright_diff,
            steady_ar=None,
            s_weight=1.0,
            v_weight=1.0,
            xtomo_weight=1.0,
            xtomo_rel_uncertainty=0.1,
            sign=1.0
        ):
        r"""Computes the log-posterior corresponding to the given differences.
        
        If there is a NaN in the differences, returns `-scipy.inf`.
        
        Here, the weighted differences :math:`\chi^2` are taken as
        
        .. math::
            
            \chi^2 = \sum_i \left ( w_{i}\frac{b_{STRAHL, i} - b_{data, i}}{\sigma_i} \right )^2
        
        In effect, the weight factors :math:`w_i` (implemented as keywords
        `s_weight`, `v_weight` and `xtomo_weight`) let you scale the uncertainty
        for a given diagnostic up and down. A higher weight corresponds to a
        smaller uncertainty and hence a bigger role in the inference, and a
        lower weight corresponds to a larger uncertainty and hence a smaller
        role in the inference.
        
        The log-posterior itself is then computed as
        
        .. math::
            
            \ln p \propto -\chi^2 / 2 + \ln p(D, V)
        
        Here, :math:`\ln p(D, V)` is the log-prior.
        
        Parameters
        ----------
        params : array of float
            The parameters to use.
        sbright_diff : array of float, (`n_time`, `n_chords`)
            The HiReX-SR differences on the HiReX-SR timebase.
        vbright_diff : array of float, (`n_time`, `n_lines`)
            The VUV differences on the VUV timebase, with XEUS first and LoWEUS
            second.
        xtomobright_diff : dict of arrays of float (`n_time`, `n_chords`)
            The XTOMO differences on the XTOMO timebase. The keys of the dict
            should be the system indices as ints, the values should be arrays of
            float with shape (`n_time`, `n_chords`).
        steady_ar : float, optional
            If None, compute for calcium. If a float, compute for argon.
        s_weight : float or array of float, (`n_chords`,), optional
            The factor to weight the HiReX-SR data by. The default is to use 1.0
            (all chords have the same weight). If this is an array, it applies
            chord-to-chord.
        v_weight : float or array of float, (`n_lines`,), optional
            The factor to weight the VUV data by. The default is to use 1.0 (all
            lines have the same weight). If this is an array, it applies
            line-to-line.
        xtomo_weight : float or dict of float or dict of arrays of float, (`n_chords`,), optional
            The factor to weight the XTOMO data by. The default is to use 1.0
            (all systems and chords have the same weight). If this is a dict,
            the keys should be the XTOMO channel numbers as ints. The values
            can either be scalar floats (a weight to apply to all chords in the
            system) or arrays of floats (weights to apply chord-to-chord).
        xtomo_rel_uncertainty : float, optional
            The relative uncertainty to use with the XTOMO systems. This is
            technically somewhat redundant with `xtomo_weight`, but is provided
            separately for convenience.
        sign : float, optional
            Sign (or other factor) applied to the final result. Set this to -1.0
            to use this function with a minimizer, for instance. Default is 1.0
            (return actual log-posterior).
        """
        # TODO: Put NaN detection back. This doesn't work because NaN's are used
        # as placeholders in the HiReX-SR signal.
        # if (
        #     scipy.isnan(sbright_diff).any() or (
        #         (steady_ar is None) and (
        #             scipy.isnan(vbright_diff).any() or
        #             scipy.any([scipy.isnan(v).any() for v in xtomobright_diff.values()])
        #         )
        #     )
        # ):
        #     print("NaN in normalized brightness!")
        #     return -scipy.inf
        
        if steady_ar is None:
            s2 = (s_weight * sbright_diff / self.run_data.hirex_uncertainty_norm_combined)**2.0
            v2 = (v_weight * vbright_diff / self.run_data.vuv_uncertainties_norm_combined.T)**2.0
            xtomo2 = {}
            for s, b in xtomobright_diff.iteritems():
                xtomo2[s] = (
                    xtomo_weight * xtomobright_diff[s] / (
                        xtomo_rel_uncertainty *
                        self.run_data.xtomo_signal_norm_combined[s].T
                    )
                )**2.0
                # Remove bad channels:
                xtomo2[s] = xtomo2[s][:, self.run_data.xtomo_channel_mask[s]]
            chi2 = (
                s2[~self.run_data.hirex_flagged_combined].sum() +
                v2.sum() +
                scipy.sum([b.sum() for b in xtomo2.values()])
            )
        else:
            ar_mask = (self.run_data.ar_time >= self.time_1) & (self.run_data.ar_time <= self.time_2)
            ar_flagged = self.run_data.ar_flagged[ar_mask, :]
            s = (s_weight * sbright_diff / self.run_data.ar_uncertainty[ar_mask, :])**2.0
            chi2 = s.ravel()[~ar_flagged.ravel()]
        
        lp = sign * (-0.5 * chi2 + self.get_prior()(params))
        # print(lp)
        return lp
    
    
    # The following are all wrapper functions. I explicitly copied the arguments
    # over for the function fingerprints, SO THESE MUST BE CAREFULLY UPDATED
    # WHEN CHANGING ANY OF THE FUNCTIONS ABOVE!!!
    
    def DV2dlines(
            self,
            params,
            explicit_D=None,
            explicit_D_grid=None,
            explicit_V=None,
            explicit_V_grid=None,
            steady_ar=None,
            debug_plots=False,
            no_write=False,
            no_strahl=False,
            compute_view_data=False,
            return_rho_t=False
        ):
        """Computes the local emissivities corresponding to the given parameters.
        
        This is simply a wrapper around the chain of :py:meth:`DV2cs_den` ->
        :py:meth:`cs_den2dlines`. See those functions for argument descriptions.
        
        Parameters
        ----------
        return_rho_t : bool, optional
            If True, the return value is a tuple of (dlines, sqrtpsinorm, time).
            Default is False (just return dlines).
        """
        out = self.DV2cs_den(
            params,
            explicit_D=explicit_D,
            explicit_D_grid=explicit_D_grid,
            explicit_V=explicit_V,
            explicit_V_grid=explicit_V_grid,
            steady_ar=steady_ar,
            debug_plots=debug_plots,
            no_write=no_write,
            no_strahl=no_strahl,
            compute_view_data=compute_view_data
        )
        try:
            cs_den, sqrtpsinorm, time, ne, Te = out
        except (TypeError, ValueError):
            raise RuntimeError(
                "Something went wrong with STRAHL, return value of DV2cs_den is: '"
                + str(out) + "', params are: " + str(params)
            )
        out = self.cs_den2dlines(
            params,
            cs_den,
            sqrtpsinorm,
            time,
            ne,
            Te,
            steady_ar=steady_ar,
            debug_plots=debug_plots
        )
        if return_rho_t:
            return out, sqrtpsinorm, time
        else:
            return out
    
    def cs_den2sig(
            self,
            params,
            cs_den,
            sqrtpsinorm,
            time,
            ne,
            Te,
            steady_ar=None,
            debug_plots=False,
            big_f=None,
            a_H=None,
            a_V=None,
            f_xtomo=None,
            a_xtomo=None,
            ar_f=None,
            ar_a=None,
            label='_nolegend_',
            lc='g',
            alpha=1.0
        ):
        """Computes the diagnostic signals corresponding to the given charge state densities.
        
        This is simply a wrapper around the chain of :py:meth:`cs_den2dlines` ->
        :py:meth:`dlines2sig`. See those functions for argument descriptions.
        """
        dlines = self.cs_den2dlines(
            params,
            cs_den,
            sqrtpsinorm,
            time,
            ne,
            Te,
            steady_ar=steady_ar,
            debug_plots=debug_plots
        )
        return self.dlines2sig(
            params,
            dlines,
            time,
            steady_ar=steady_ar,
            debug_plots=debug_plots,
            big_f=big_f,
            a_H=a_H,
            a_V=a_V,
            f_xtomo=f_xtomo,
            a_xtomo=a_xtomo,
            ar_f=ar_f,
            ar_a=ar_a,
            label=label,
            lc=lc,
            alpha=alpha
        )
    
    def dlines2diffs(
            self,
            params,
            dlines,
            time,
            steady_ar=None,
            debug_plots=False,
            big_f=None,
            a_H=None,
            a_V=None,
            f_xtomo=None,
            a_xtomo=None,
            ar_f=None,
            ar_a=None,
            label='_nolegend_',
            lc='g',
            alpha=1.0,
            no_diff=False
        ):
        """Computes the diagnostic differences corresponding to the given local emissivities.
        
        This is simply a wrapper around the chain of :py:meth:`dlines2sig` ->
        :py:meth:`sig2diffs`. See those functions for argument descriptions.
        """
        sbright, vbright, xtomobright = self.dlines2sig(
            params,
            dlines,
            time,
            steady_ar=steady_ar,
            debug_plots=debug_plots,
            big_f=big_f,
            a_H=a_H,
            a_V=a_V,
            f_xtomo=f_xtomo,
            a_xtomo=a_xtomo,
            ar_f=ar_f,
            ar_a=ar_a,
            label=label,
            lc=lc,
            alpha=alpha
        )
        return self.sig2diffs(
            params,
            sbright,
            vbright,
            xtomobright,
            time,
            steady_ar=steady_ar,
            no_diff=no_diff
        )
    
    def sig2ln_prob(
            self,
            params,
            sbright,
            vbright,
            xtomobright,
            time,
            steady_ar=None,
            s_weight=1.0,
            v_weight=1.0,
            xtomo_weight=1.0,
            xtomo_rel_uncertainty=0.1,
            sign=1.0
        ):
        """Computes the log-posterior corresponding to the given diagnostic signals.
        
        This is simply a wrapper around the chain of :py:meth:`sig2diffs` ->
        :py:meth:`diffs2ln_prob`. See those functions for argument descriptions.
        """
        sbright_diff, vbright_diff, xtomobright_diff = self.sig2diffs(
            params,
            sbright,
            vbright,
            xtomobright,
            time,
            steady_ar=steady_ar
        )
        return self.diffs2ln_prob(
            params,
            sbright_diff,
            vbright_diff,
            xtomobright_diff,
            steady_ar=steady_ar,
            s_weight=s_weight,
            v_weight=v_weight,
            xtomo_weight=xtomo_weight,
            xtomo_rel_uncertainty=xtomo_rel_uncertainty,
            sign=sign
        )
        
    def DV2sig(
            self,
            params,
            explicit_D=None,
            explicit_D_grid=None,
            explicit_V=None,
            explicit_V_grid=None,
            steady_ar=None,
            debug_plots=False,
            no_write=False,
            no_strahl=False,
            compute_view_data=False,
            big_f=None,
            a_H=None,
            a_V=None,
            f_xtomo=None,
            a_xtomo=None,
            ar_f=None,
            ar_a=None,
            label='_nolegend_',
            lc='g',
            alpha=1.0
        ):
        """Predicts the diagnostic signals that would arise from the given parameters.
        
        This is simply a wrapper around the chain of :py:meth:`DV2cs_den` ->
        :py:meth:`cs_den2dlines` -> :py:meth:`dlines2sig`. See those functions
        for argument descriptions.
        """
        out = self.DV2cs_den(
            params,
            explicit_D=explicit_D,
            explicit_D_grid=explicit_D_grid,
            explicit_V=explicit_V,
            explicit_V_grid=explicit_V_grid,
            steady_ar=steady_ar,
            debug_plots=debug_plots,
            no_write=no_write,
            no_strahl=no_strahl,
            compute_view_data=compute_view_data
        )
        try:
            cs_den, sqrtpsinorm, time, ne, Te = out
        except (TypeError, ValueError):
            raise RuntimeError(
                "Something went wrong with STRAHL, return value of DV2cs_den is: '"
                + str(out) + "', params are: " + str(params)
            )
        dlines = self.cs_den2dlines(
            params,
            cs_den,
            sqrtpsinorm,
            time,
            ne,
            Te,
            steady_ar=steady_ar,
            debug_plots=debug_plots
        )
        return self.dlines2sig(
            params,
            dlines,
            time,
            steady_ar=steady_ar,
            debug_plots=debug_plots,
            big_f=big_f,
            a_H=a_H,
            a_V=a_V,
            f_xtomo=f_xtomo,
            a_xtomo=a_xtomo,
            ar_f=ar_f,
            ar_a=ar_a,
            label=label,
            lc=lc,
            alpha=alpha
        )
        
    
    def cs_den2diffs(
            self,
            params,
            cs_den,
            sqrtpsinorm,
            time,
            ne,
            Te,
            steady_ar=None,
            debug_plots=False,
            big_f=None,
            a_H=None,
            a_V=None,
            f_xtomo=None,
            a_xtomo=None,
            ar_f=None,
            ar_a=None,
            label='_nolegend_',
            lc='g',
            alpha=1.0,
            no_diff=False
        ):
        """Computes the diagnostic differences corresponding to the given charge state densities.
        
        This is simply a wrapper around the chain of :py:meth:`cs_den2dlines` ->
        :py:meth:`dlines2sig` -> :py:meth:`sig2diffs`. See those functions for
        argument descriptions.
        """
        dlines = self.cs_den2dlines(
            params,
            cs_den,
            sqrtpsinorm,
            time,
            ne,
            Te,
            steady_ar=steady_ar,
            debug_plots=debug_plots
        )
        sbright, vbright, xtomobright = self.dlines2sig(
            params,
            dlines,
            time,
            steady_ar=steady_ar,
            debug_plots=debug_plots,
            big_f=big_f,
            a_H=a_H,
            a_V=a_V,
            f_xtomo=f_xtomo,
            a_xtomo=a_xtomo,
            ar_f=ar_f,
            ar_a=ar_a,
            label=label,
            lc=lc,
            alpha=alpha
        )
        return self.sig2diffs(
            params,
            sbright,
            vbright,
            xtomobright,
            time,
            steady_ar=steady_ar,
            no_diff=no_diff
        )
    
    def dlines2ln_prob(
            self,
            params,
            dlines,
            time,
            steady_ar=None,
            debug_plots=False,
            big_f=None,
            a_H=None,
            a_V=None,
            f_xtomo=None,
            a_xtomo=None,
            ar_f=None,
            ar_a=None,
            label='_nolegend_',
            lc='g',
            alpha=1.0,
            s_weight=1.0,
            v_weight=1.0,
            xtomo_weight=1.0,
            xtomo_rel_uncertainty=0.1,
            sign=1.0
        ):
        """Computes the log-posterior corresponding to the given local emissivities.
        
        This is simply a wrapper around the chain of :py:meth:`dlines2sig` ->
        :py:meth:`sig2diffs` -> :py:meth:`diffs2ln_prob`. See those functions
        for argument descriptions.
        """
        sbright, vbright, xtomobright = self.dlines2sig(
            params,
            dlines,
            time,
            steady_ar=steady_ar,
            debug_plots=debug_plots,
            big_f=big_f,
            a_H=a_H,
            a_V=a_V,
            f_xtomo=f_xtomo,
            a_xtomo=a_xtomo,
            ar_f=ar_f,
            ar_a=ar_a,
            label=label,
            lc=lc,
            alpha=alpha
        )
        sbright_diff, vbright_diff, xtomobright_diff = self.sig2diffs(
            params,
            sbright,
            vbright,
            xtomobright,
            time,
            steady_ar=steady_ar
        )
        return self.diffs2ln_prob(
            params,
            sbright_diff,
            vbright_diff,
            xtomobright_diff,
            steady_ar=steady_ar,
            s_weight=s_weight,
            v_weight=v_weight,
            xtomo_weight=xtomo_weight,
            xtomo_rel_uncertainty=xtomo_rel_uncertainty,
            sign=sign
        )
    
    def DV2diffs(
            self,
            params,
            explicit_D=None,
            explicit_D_grid=None,
            explicit_V=None,
            explicit_V_grid=None,
            steady_ar=None,
            debug_plots=False,
            no_write=False,
            no_strahl=False,
            compute_view_data=False,
            big_f=None,
            a_H=None,
            a_V=None,
            f_xtomo=None,
            a_xtomo=None,
            ar_f=None,
            ar_a=None,
            label='_nolegend_',
            lc='g',
            alpha=1.0,
            no_diff=False
        ):
        """Computes the diagnostic differences corresponding to the given parameters.
        
        The is simply a wrapper around the chain of :py:meth:`DV2cs_den` ->
        :py:meth:`cs_den2dlines` -> :py:meth:`dlines2sig` -> :py:meth:`sig2diffs`.
        See those functions for argument descriptions.
        """
        out = self.DV2cs_den(
            params,
            explicit_D=explicit_D,
            explicit_D_grid=explicit_D_grid,
            explicit_V=explicit_V,
            explicit_V_grid=explicit_V_grid,
            steady_ar=steady_ar,
            debug_plots=debug_plots,
            no_write=no_write,
            no_strahl=no_strahl,
            compute_view_data=compute_view_data
        )
        try:
            cs_den, sqrtpsinorm, time, ne, Te = out
        except (TypeError, ValueError):
            raise RuntimeError(
                "Something went wrong with STRAHL, return value of DV2cs_den is: '"
                + str(out) + "', params are: " + str(params)
            )
        dlines = self.cs_den2dlines(
            params,
            cs_den,
            sqrtpsinorm,
            time,
            ne,
            Te,
            steady_ar=steady_ar,
            debug_plots=debug_plots
        )
        sbright, vbright, xtomobright = self.dlines2sig(
            params,
            dlines,
            time,
            steady_ar=steady_ar,
            debug_plots=debug_plots,
            big_f=big_f,
            a_H=a_H,
            a_V=a_V,
            f_xtomo=f_xtomo,
            a_xtomo=a_xtomo,
            ar_f=ar_f,
            ar_a=ar_a,
            label=label,
            lc=lc,
            alpha=alpha
        )
        return self.sig2diffs(
            params,
            sbright,
            vbright,
            xtomobright,
            time,
            steady_ar=steady_ar,
            no_diff=no_diff
        )
    
    def cs_den2ln_prob(
            self,
            params,
            cs_den,
            sqrtpsinorm,
            time,
            ne,
            Te,
            steady_ar=None,
            debug_plots=False,
            big_f=None,
            a_H=None,
            a_V=None,
            f_xtomo=None,
            a_xtomo=None,
            ar_f=None,
            ar_a=None,
            label='_nolegend_',
            lc='g',
            alpha=1.0,
            s_weight=1.0,
            v_weight=1.0,
            xtomo_weight=1.0,
            xtomo_rel_uncertainty=0.1,
            sign=1.0
        ):
        """Computes the log-posterior corresponding to the given charge-state densities.
        
        This is simply a wrapper around the chain of :py:meth:`cs_den2dlines` ->
        :py:meth:`dlines2sig` -> :py:meth:`sig2diffs` -> :py:meth:`diffs2ln_prob`.
        See those functions for argument descriptions.
        """
        dlines = self.cs_den2dlines(
            params,
            cs_den,
            sqrtpsinorm,
            time,
            ne,
            Te,
            steady_ar=steady_ar,
            debug_plots=debug_plots
        )
        sbright, vbright, xtomobright = self.dlines2sig(
            params,
            dlines,
            time,
            steady_ar=steady_ar,
            debug_plots=debug_plots,
            big_f=big_f,
            a_H=a_H,
            a_V=a_V,
            f_xtomo=f_xtomo,
            a_xtomo=a_xtomo,
            ar_f=ar_f,
            ar_a=ar_a,
            label=label,
            lc=lc,
            alpha=alpha
        )
        sbright_diff, vbright_diff, xtomobright_diff = self.sig2diffs(
            params,
            sbright,
            vbright,
            xtomobright,
            time,
            steady_ar=steady_ar
        )
        return self.diffs2ln_prob(
            params,
            sbright_diff,
            vbright_diff,
            xtomobright_diff,
            steady_ar=steady_ar,
            s_weight=s_weight,
            v_weight=v_weight,
            xtomo_weight=xtomo_weight,
            xtomo_rel_uncertainty=xtomo_rel_uncertainty,
            sign=sign
        )
    
    def DV2ln_prob(
            self,
            params,
            sign=1.0,
            explicit_D=None,
            explicit_D_grid=None,
            explicit_V=None,
            explicit_V_grid=None,
            steady_ar=None,
            debug_plots=False,
            no_write=False,
            no_strahl=False,
            compute_view_data=False,
            big_f=None,
            a_H=None,
            a_V=None,
            f_xtomo=None,
            a_xtomo=None,
            ar_f=None,
            ar_a=None,
            label='_nolegend_',
            lc='g',
            alpha=1.0,
            s_weight=1.0,
            v_weight=1.0,
            xtomo_weight=1.0,
            xtomo_rel_uncertainty=0.1
        ):
        """Computes the log-posterior corresponding to the given parameters.
        
        This is simply a wrapper around the chain of :py:meth:`DV2cs_den` ->
        :py:meth:`cs_den2dlines` -> :py:meth:`dlines2sig` ->
        :py:meth:`sig2diffs` -> :py:meth:`diffs2ln_prob`. See those functions
        for argument descriptions. This is designed to work as a log-posterior
        function for various MCMC samplers, etc.
        
        THOUGH I MAY WANT TO CONSIDER A STREAMLINED VERSION!!!
        """
        out = self.DV2cs_den(
            params,
            explicit_D=explicit_D,
            explicit_D_grid=explicit_D_grid,
            explicit_V=explicit_V,
            explicit_V_grid=explicit_V_grid,
            steady_ar=steady_ar,
            debug_plots=debug_plots,
            no_write=no_write,
            no_strahl=no_strahl,
            compute_view_data=compute_view_data
        )
        try:
            cs_den, sqrtpsinorm, time, ne, Te = out
        except (TypeError, ValueError):
            raise RuntimeError(
                "Something went wrong with STRAHL, return value of DV2cs_den is: '"
                + str(out) + "', params are: " + str(params)
            )
        dlines = self.cs_den2dlines(
            params,
            cs_den,
            sqrtpsinorm,
            time,
            ne,
            Te,
            steady_ar=steady_ar,
            debug_plots=debug_plots
        )
        sbright, vbright, xtomobright = self.dlines2sig(
            params,
            dlines,
            time,
            steady_ar=steady_ar,
            debug_plots=debug_plots,
            big_f=big_f,
            a_H=a_H,
            a_V=a_V,
            f_xtomo=f_xtomo,
            a_xtomo=a_xtomo,
            ar_f=ar_f,
            ar_a=ar_a,
            label=label,
            lc=lc,
            alpha=alpha
        )
        sbright_diff, vbright_diff, xtomobright_diff = self.sig2diffs(
            params,
            sbright,
            vbright,
            xtomobright,
            time,
            steady_ar=steady_ar
        )
        return self.diffs2ln_prob(
            params,
            sbright_diff,
            vbright_diff,
            xtomobright_diff,
            steady_ar=steady_ar,
            s_weight=s_weight,
            v_weight=v_weight,
            xtomo_weight=xtomo_weight,
            xtomo_rel_uncertainty=xtomo_rel_uncertainty,
            sign=sign
        )
    
    def u2ln_prob(self, u, nl_grad=None, sign=1.0, return_grad=False, grad_only=False, pool=None, eps=scipy.sqrt(sys.float_info.epsilon), **kwargs):
        r"""Convert the log-posterior corresponding to a given set of CDF values.
        
        Passes the values `u` (which lie in :math:`[0, 1]`) through the inverse CDF
        before passing them to :py:meth:`DV2ln_prob`.
        
        Also catches out-of-bounds and exceptions so as to be useful for
        optimization.
        
        Also can compute gradients using finite differences. Is intelligent
        about using forward/backwards differences near bounds. It supports use
        of a pool for parallelizing the gradient calculation, but this adds so
        much overhead it will probably only be worth it for shockingly high-
        dimensional problems. It also seems to promptly consume all system
        memory, which is frustrating to say the least.
        
        Parameters
        ----------
        u : array of float, (`num_params`,)
            The parameters, mapped through the CDF to lie in :math:`[0, 1]`.
        nl_grad : None or array, (`num_params`,), optional
            Container to put the gradient in when used with NLopt. If present
            and `grad.size > 0`, `return_grad` is set to True and the gradient
            is put into `grad` in-place.
        sign : float, optional
            The sign/scaling factor to apply to the result before returning. The
            default is 1.0 (for maximization/sampling). Set to -1.0 for
            minimization.
        return_grad : bool, optional
            If True, the gradient is computed using finite differences. Single-
            order forward differences are preferred, but single-order backward
            differences will be used if the parameters are too close to the
            bounds. If the bounds are tighter than `eps`, the gradient is set to
            zero. (This condition should never be reached if you have reasonable
            bounds.) Default is False (do not compute gradient).
        grad_only : bool, optional
            If `grad_only` and `return_grad` are both True, then only the
            gradient is returned. Default is False.
        pool : pool, optional
            If this is not None, the pool will be used to evaluate the terms in
            the gradient. Note that this adds enough overhead that it probably
            only helps for very high-dimensional problems. It also seems to run
            the system out of memory and crash, so you should probably just
            leave it set to None.
        eps : float, optional
            The step size to use when computing the derivative with finite
            differences. The default is the square root of machine epsilon.
        **kwargs : extra keywords, optional
            All extra keywords are passed to :py:meth:`DV2ln_prob`. 
        """
        if nl_grad is not None and nl_grad.size > 0:
            return_grad = True
            nlopt_format = True
        else:
            nlopt_format = False
        if return_grad:
            start = time_.time()
            print(u)
        u = scipy.asarray(u, dtype=float)
        if (u < 0.0).any():
            print("Lower bound fail!")
            u[u < 0.0] = 0.0
        elif (u > 1.0).any():
            print("Upper bound fail!")
            u[u > 1.0] = 1.0
        params = self.get_prior().sample_u(u)
        try:
            fu = self.DV2ln_prob(params, sign=sign, **kwargs)
        except:
            print(u)
            fu = sign * -scipy.inf
        if return_grad:
            grad = scipy.zeros_like(u)
            if pool is None:
                for k in xrange(0, len(u)):
                    if u[k] + eps <= 1.0:
                        u_mod = u.copy()
                        u_mod[k] = u_mod[k] + eps
                        grad[k] = (self.u2ln_prob(u_mod, sign=sign, return_grad=False, **kwargs) - fu) / eps
                    elif u[k] - eps >= 0.0:
                        u_mod = u.copy()
                        u_mod[k] = u_mod[k] - eps
                        grad[k] = (fu - self.u2ln_prob(u_mod, sign=sign, return_grad=False, **kwargs)) / eps
                    else:
                        print("finite difference fail!")
            else:
                u_mod = []
                signs = []
                for k in xrange(0, len(u)):
                    if u[k] + eps <= 1.0:
                        u_mod.append(u.copy())
                        u_mod[-1][k] = u_mod[-1][k] + eps
                        signs.append(1)
                    elif u[k] - eps >= 0.0:
                        u_mod.append(u.copy())
                        u_mod[-1][k] = u_mod[-1][k] - eps
                        signs.append(-1)
                    else:
                        u_mod.append(u.copy())
                        signs.append(0)
                        print("finite difference fail!")
                f_shifts = pool.map(_UGradEval(self, sign, kwargs), u_mod)
                for k in xrange(0, len(u)):
                    if signs[k] == 1:
                        grad[k] = (f_shifts[k] - fu) / eps
                    elif signs[k] == -1:
                        grad[k] = (fu - f_shifts[k]) / eps
            if grad_only:
                out = grad
            elif nlopt_format:
                out = fu
                nl_grad[:] = grad
            else:
                out = (fu, grad)
        else:
            out = fu
        if return_grad:
            print(time_.time() - start)
        return out
    
    # Old version:
    # def compute_ln_prob(
    #         self,
    #         params,
    #         return_blob=False,
    #         light_blob=False,
    #         sign=1,
    #         no_prior=False,
    #         compute_view_data=False,
    #         debug_plots=False,
    #         compute_NC=False,
    #         explicit_D=None,
    #         explicit_D_grid=None,
    #         explicit_V=None,
    #         explicit_V_grid=None,
    #         no_write=False,
    #         no_strahl=False,
    #         STRAHL_compute_rad=False,
    #         label='_nolegend_',
    #         lc=None,
    #         big_f=None,
    #         a_H=None,
    #         a_V=None,
    #         ar_f=None,
    #         ar_a=None,
    #         steady_ar=None,
    #         alpha=1.0
    #     ):
    #     """Calls STRAHL with the given params and computes the log-posterior.
    #
    #     Returns the log-posterior, or optionally the log-likelihood.
    #
    #     Parameters
    #     ----------
    #     params : array of float, (`num_eig_D` + `num_eig_V` + `num_param_k_D` + `num_param_k_V` + `num_param_source`)
    #         The parameters to use. The order is:
    #
    #         * eig_D: The eigenvalues to use when evaluating the D profile.
    #         * eig_V: The eigenvalues to use when evaluating the V profile.
    #         * param_D: The hyperparameters to use for the D profile.
    #         * param_mu_D: The hyperparameters to use for the mean function of the D profile.
    #         * param_V: The hyperparameters to use for the V profile.
    #         * knots_D: The knots of the D profile.
    #         * knots_V: The knots of the V profile.
    #         * scaling: The scaling factors for each diagnostic.
    #         * param_source: The parameters to use for the model source function.
    #     return_blob : bool, optional
    #         If True, a blob with metadata from the evaluation will be returned
    #         along with the log-probability. The entries of the blob are ordered
    #         as follows:
    #
    #         * Log-likelihood
    #         * Normalized HiReX-SR brightness
    #         * Normalized VUV brightness
    #         * Time array for both
    #         * String with all of the output from STRAHL during the run
    #
    #         Default is False
    #     light_blob : bool, optional
    #         If True, the blob returned when `returb_blob` is True will contain
    #         only the log-likelihood (needed to compute various useful things).
    #         Default is False (return full blob).
    #     sign : float, optional
    #         A factor to apply to the output. This allows the user to change the
    #         sign of the log-probability so that a minimizer can be used to find
    #         the maximum probability. The default is 1.
    #     no_prior : bool, optional
    #         If True, the log-likelihood is returned instead of the log-posterior.
    #         Default is False.
    #     compute_view_data : bool, optional
    #         Set this to True to only compute the view_data.sav file. (Returns
    #         the sqrtpsinorm grid STRAHL uses.)
    #     debug_plots : bool, optional
    #         Set this to True to make a plot of the STRAHL output on top of the
    #         experimental data. Only has an effect if `compute_view_data` is
    #         False. Default is False.
    #     compute_NC : bool, optional
    #         Set this to True to have STRAHL attempt to compute the neoclassical
    #         transport. At present these values are not returned, but a plot will
    #         be produced if `debug_plots` is True. Default is False.
    #     explicit_D : array of float, optional
    #         Explicit values of D to use. Overrides the profile which would have
    #         been obtained from the parameters in `params` (but the scalings/etc.
    #         from `params` are still used).
    #     explicit_D_grid : array of float, optional
    #         Grid of sqrtpsinorm which `explicit_D` is given on.
    #     explicit_V : array of float, optional
    #         Explicit values of V to use. Overrides the profile which would have
    #         been obtained from the parameters in `params` (but the scalings/etc.
    #         from `params` are still used).
    #     explicit_V_grid : array of float, optional
    #         Grid of sqrtpsinorm which `explicit_V` is given on.
    #     no_write : bool, optional
    #         If True, the STRAHL control files are not written. Used for
    #         debugging. Default is False.
    #     no_strahl : bool, optional
    #         If True, STRAHL is not actually called (and the existing results
    #         file is evaluated). Used for debugging. Default is False.
    #     STRAHL_compute_rad : bool, optional
    #         If True, STRAHL is used to compute the line radiation. Default is
    #         False.
    #     label : str, optional
    #         Label to use for lines added to existing figures. Default is
    #         '_nolegend_'.
    #     lc : str, optional
    #         Line color to use for lines added to existing figures. Default is
    #         None.
    #     big_f : Figure instance, optional
    #         The figure containing plots of each data source. If not present, a
    #         new figure is created by calling :py:meth:`RunData.plot_data`.
    #     a_H : list of Axes instances, optional
    #         The axes from `big_f` which contain the HiReX-SR data.
    #     a_V : list of Axes instances, optional
    #         The axes from `big_f` which contain the VUV spectrometer data.
    #     ar_f : Figure instance, optional
    #         The figure containing the argon data. If not present, a new figure
    #         is created by calling :py:meth:`RunData.plot_ar`.
    #     ar_a : Axes instance, optional
    #         The axes from `ar_f` to plot the argon profile on.
    #     steady_ar : float, optional
    #         If present, will compute the steady-state (constant-source) Ar
    #         profiles for the given source instead of the time-evolving Ca
    #         profiles. Default is None.
    #     """
    #     t_start = time_.time()
    #
    #     eig_D, eig_V, knots_D, knots_V, hp_D, hp_mu_D, hp_V, param_scaling, param_source = self.split_params(params)
    #
    #     if (explicit_D is None) or (explicit_V is None):
    #         # First, evaluate the prior:
    #         lnprob = self.get_prior()(params)
    #     else:
    #         # But skip the prior if there is an explicit D or V:
    #         lnprob = 0.0
    #
    #     ll = -scipy.inf
    #
    #     # Set default blob if needed:
    #     if return_blob:
    #         if light_blob:
    #             blob = [-scipy.inf]
    #         else:
    #             blob = [-scipy.inf, None, None, None, 'Infinite prior!']
    #
    #     # Only bother to evaluate if the prior is finite, or force it to run
    #     # through is we are just computing view data:
    #     if not scipy.isinf(lnprob) or compute_view_data:
    #         # Compute and check the profiles:
    #         if (explicit_D is None) or (explicit_V is None):
    #             try:
    #                 D, V = self.eval_DV(params, plot=debug_plots)
    #             except ValueError:
    #                 print("Failure evaluating profiles!")
    #                 if return_blob:
    #                     if not light_blob:
    #                         blob[-1] = 'Failed evaluation of D, V!'
    #                     return (sign * -scipy.inf, blob)
    #                 else:
    #                     return sign * -scipy.inf
    #         # Get the correct grids, handle explicit D and V:
    #         if explicit_D is not None:
    #             D = explicit_D
    #             D_grid = explicit_D_grid
    #         else:
    #             D_grid = scipy.sqrt(self.psinorm_grid_DV)
    #         if explicit_V is not None:
    #             V = explicit_V
    #             V_grid = explicit_V_grid
    #         else:
    #             V_grid = scipy.sqrt(self.psinorm_grid_DV)
    #         # Check for bad values in D, V profiles:
    #         if scipy.isinf(D).any() or scipy.isnan(D).any():
    #             print("inf in D!")
    #             print(params)
    #             if return_blob:
    #                 if not light_blob:
    #                     blob[-1] = 'Infinite D!'
    #                 return (sign * -scipy.inf, blob)
    #             else:
    #                 return sign * -scipy.inf
    #         if scipy.isinf(V).any() or scipy.isnan(V).any():
    #             print("inf in V!")
    #             print(params)
    #             if return_blob:
    #                 if not light_blob:
    #                     blob[-1] = 'Infinite V!'
    #                 return (sign * -scipy.inf, blob)
    #             else:
    #                 return sign * -scipy.inf
    #
    #         # Evaluate the source function:
    #         # if self.source_file is None:
    #         #     t_source = scipy.linspace(
    #         #         self.time_1,
    #         #         self.time_1 + param_source[1] + param_source[3],
    #         #         self.nt_source
    #         #     )
    #         #     s = 1e17 * source_function(t_source, self.time_1, *param_source[1:])
    #
    #         # Evaluate ne, Te:
    #         ne = self.run_data.ne_res['mean_val']
    #         Te = self.run_data.Te_res['mean_val']
    #         # HACK to get rid of negative values in ne, Te:
    #         ne[ne < 0.0] = 0.0
    #         Te[Te < 0.0] = 0.0
    #
    #         # Now write the param and pp files, if required:
    #         if not no_write:
    #             # Need to override the start/end times of steady_ar is not None:
    #             if steady_ar is None:
    #                 time_2_override = None
    #             else:
    #                 time_2_override = self.time_1 + 0.2
    #             self.write_control(time_2_override=time_2_override)
    #             self.write_pp(
    #                 scipy.sqrt(self.psinorm_grid),
    #                 ne,
    #                 Te,
    #                 self.time_2 if steady_ar is None else time_2_override
    #             )
    #             self.write_param(
    #                 D_grid,
    #                 V_grid,
    #                 D,
    #                 V,
    #                 compute_NC=compute_NC,
    #                 const_source=steady_ar,
    #                 element='Ca' if steady_ar is None else 'Ar',
    #                 time_2_override=time_2_override
    #             )
    #
    #         # Now call STRAHL:
    #         try:
    #             if no_strahl:
    #                 out = 'STRAHL not run!'
    #             else:
    #                 command = ['./strahl', 'a']
    #                 # The "n" disables STRAHL's calculation of radiation.
    #                 if not STRAHL_compute_rad:
    #                     command += ['n',]
    #                 out = subprocess.check_output(command, stderr=subprocess.STDOUT)
    #         except subprocess.CalledProcessError as e:
    #             print("STRAHL exited with error code %d." % (e.returncode))
    #             if return_blob:
    #                 if not light_blob:
    #                     blob[-1] = str(e.returncode)
    #                 return (sign * -scipy.inf, blob)
    #             else:
    #                 return sign * -scipy.inf
    #
    #         if return_blob:
    #             if not light_blob:
    #                 blob[-1] = out
    #
    #         # Process the results:
    #         f = scipy.io.netcdf.netcdf_file('result/strahl_result.dat', 'r')
    #         sqrtpsinorm = scipy.asarray(f.variables['rho_poloidal_grid'][:], dtype=float)
    #         if compute_view_data:
    #             return sqrtpsinorm
    #         time = scipy.asarray(f.variables['time'][:], dtype=float)
    #
    #         # Check to make sure it ran through:
    #         if time[-1] <= self.time_2 - 0.1 * (self.time_2 - self.time_1):
    #             print(time[-1])
    #             print(len(time))
    #             print("STRAHL failed (max iterations)!")
    #             if return_blob:
    #                 return (sign * -scipy.inf, blob)
    #             else:
    #                 return sign * -scipy.inf
    #
    #         # Shift the timebase to be centered on the injection, applying the
    #         # temporal shifts from the parameters:
    #         time = time - self.time_1
    #         time_s = time + param_source[0]
    #         time_v = time + param_source[1]
    #
    #         # dlines has shape (n_time, n_lines, n_space)
    #         strahl_dlines = scipy.asarray(f.variables['diag_lines_radiation'][:], dtype=float)
    #
    #         # cs_den has shape (n_time, n_cs, n_space)
    #         cs_den = scipy.asarray(f.variables['impurity_density'][:], dtype=float)
    #
    #         # Compute my own emissivity:
    #         ne = scipy.asarray(f.variables['electron_density'][:], dtype=float)
    #         Te = scipy.asarray(f.variables['electron_temperature'][:], dtype=float)
    #         dlines = scipy.zeros_like(strahl_dlines)
    #         atomdat = self.atomdat if steady_ar is None else self.Ar_atomdat
    #         if steady_ar is None:
    #             for i, chg, cw, hw in zip(
    #                     range(0, len(atomdat[0])),
    #                     atomdat[0],
    #                     atomdat[1],
    #                     atomdat[2]
    #                 ):
    #                 dlines[:, i, :] = compute_emiss(
    #                     self.PEC[chg],
    #                     cw,
    #                     hw,
    #                     ne,
    #                     cs_den[:, chg, :],
    #                     Te
    #                 )
    #         else:
    #             # We need to add up the contributions to the z-line. These are
    #             # stored in the PEC dict in the charge of the state the line is
    #             # populated from.
    #             # Excitation:
    #             dlines[:, 0, :] = compute_emiss(
    #                 self.Ar_PEC[16],
    #                 4.0,
    #                 0.1,
    #                 ne,
    #                 cs_den[:, 16, :],
    #                 Te,
    #                 no_ne=True
    #             )
    #             # Ionization:
    #             dlines[:, 0, :] += compute_emiss(
    #                 self.Ar_PEC[15],
    #                 4.0,
    #                 0.1,
    #                 ne,
    #                 cs_den[:, 15, :],
    #                 Te,
    #                 no_ne=True
    #             )
    #             # Recombination:
    #             dlines[:, 0, :] += compute_emiss(
    #                 self.Ar_PEC[17],
    #                 4.0,
    #                 0.1,
    #                 ne,
    #                 cs_den[:, 17, :],
    #                 Te,
    #                 no_ne=True
    #             )
    #
    #         if debug_plots:
    #             # Plot the charge state densities:
    #             slider_plot(
    #                 sqrtpsinorm,
    #                 time,
    #                 scipy.rollaxis(cs_den.T, 1),
    #                 xlabel=r'$\sqrt{\psi_n}$',
    #                 ylabel=r'$t$ [s]',
    #                 zlabel=r'$n$ [cm$^{-3}$]',
    #                 labels=[str(i) for i in range(0, cs_den.shape[1])]
    #             )
    #             # Plot the emissivity profiles:
    #             slider_plot(
    #                 sqrtpsinorm,
    #                 time,
    #                 scipy.rollaxis(dlines.T, 1),
    #                 xlabel=r'$\sqrt{\psi_n}$',
    #                 ylabel=r'$t$ [s]',
    #                 zlabel=r'$\epsilon$ [W/cm$^3$]',
    #                 labels=[str(i) for i in range(0, dlines.shape[1])]
    #             )
    #             # Plot the emissivity I compute versus what STRAHL computes:
    #             if STRAHL_compute_rad:
    #                 slider_plot(
    #                     sqrtpsinorm,
    #                     time,
    #                     scipy.vstack(
    #                         (
    #                             scipy.rollaxis(dlines.T, 1),
    #                             scipy.rollaxis(strahl_dlines.T, 1)
    #                         ),
    #                     ),
    #                     xlabel=r'$\sqrt{\psi_n}$',
    #                     ylabel=r'$t$ [s]',
    #                     zlabel=r'$\epsilon$ [W/cm$^3$]',
    #                     labels=['bayesimp ' + str(i) for i in range(0, dlines.shape[1])] +
    #                            ['STRAHL ' + str(i) for i in range(0, dlines.shape[1])]
    #                 )
    #
    #             if compute_NC:
    #                 roa = self.efit_tree.psinorm2roa(
    #                     sqrtpsinorm**2,
    #                     (self.time_1 + self.time_2) / 2.0
    #                 )
    #                 D_neo = (
    #                     f.variables['classical_diff_coeff'][:] +
    #                     f.variables['pfirsch_schlueter_diff_coeff'][:] +
    #                     f.variables['banana_plateau_diff_coeff'][:]
    #                 )[-1, :] * 1e-4
    #                 V_neo = (
    #                     f.variables['classical_drift'][:] +
    #                     f.variables['pfirsch_schlueter_drift'][:] +
    #                     f.variables['banana_plateau_drift'][:]
    #                 )[-1, :] * 1e-2
    #                 fig = plt.figure()
    #                 a_D = fig.add_subplot(2, 1, 1)
    #                 a_D.plot(roa, D_neo)
    #                 a_V = fig.add_subplot(2, 1, 2)
    #                 a_V.plot(roa, V_neo)
    #                 a_D.set_xlabel('$r/a$')
    #                 a_D.set_ylabel('$D$ [m$^2$/s]')
    #                 a_V.set_xlabel('$r/a$')
    #                 a_V.set_ylabel('$V$ [m/s]')
    #
    #         # Compute the brightnesses:
    #         # These set the mapping from the indices in the STRAHL output to the
    #         # instruments:
    #         # TODO: THIS SHOULD JUST GET STORED IN RUN_DATA!
    #         hirex_line = 0
    #         xeus_lines = range(1, 1 + len(self.run_data.vuv_lines['XEUS']))
    #         # TODO: Make this robust to leaving out one or more instruments!
    #         if 'LoWEUS' in self.run_data.vuv_lines:
    #             loweus_lines = range(
    #                 1 + len(self.run_data.vuv_lines['XEUS']),
    #                 1 + len(self.run_data.vuv_lines['XEUS']) +
    #                     len(self.run_data.vuv_lines['LoWEUS'])
    #             )
    #         else:
    #             loweus_lines = None
    #
    #         # Compute the HiReX-SR brightness:
    #         if steady_ar is None:
    #             # Compute the HiReX-SR brightness:
    #             sbright = self.weights[:self.run_data.hirex_pos.shape[0]].dot(dlines[:, hirex_line, :].T).T
    #             # Compute the XEUS brightness:
    #             xbright = []
    #             for i in xeus_lines:
    #                 xbright.append(self.weights[self.run_data.hirex_pos.shape[0]].dot(dlines[:, i, :].T))
    #             xbright = scipy.vstack(xbright).T
    #             # Compute the LoWEUS brightness:
    #             if loweus_lines is not None:
    #                 lbright = []
    #                 for i in loweus_lines:
    #                     lbright.append(self.weights[self.run_data.hirex_pos.shape[0] + 1].dot(dlines[:, i, :].T))
    #                 lbright = scipy.vstack(lbright).T
    #             # Normalize the VUV signals:
    #             for k in xrange(0, xbright.shape[1]):
    #                 xbright[:, k] = xbright[:, k] / xbright[:, k].max()
    #             if loweus_lines is not None:
    #                 for k in xrange(0, lbright.shape[1]):
    #                     lbright[:, k] = lbright[:, k] / lbright[:, k].max()
    #
    #             if loweus_lines is not None:
    #                 vbright = scipy.hstack((xbright, lbright))
    #             else:
    #                 vbright = xbright
    #
    #             # Apply scalings:
    #             if self.use_scaling:
    #                 sbright = sbright * param_scaling[0]
    #                 for k in xrange(0, len(param_scaling) - 1):
    #                     vbright[:, k] *= param_scaling[k + 1]
    #         else:
    #             sbright = self.ar_weights.dot(dlines[:, 0, :].T).T
    #
    #         # Normalize the HiReX-SR signal:
    #         sbright = sbright / sbright.max()
    #
    #         # Do this after in case there was a divide-by-zero (or just bad
    #         # STRAHL output):
    #         if scipy.isnan(sbright).any() or ((steady_ar is None) and (scipy.isnan(vbright).any())):
    #             print("NaN in normalized brightness!")
    #             if return_blob:
    #                 return (sign * -scipy.inf, blob)
    #             else:
    #                 return sign * -scipy.inf
    #
    #         # Compute chi^2, add it to lnprob (the log-prior):
    #         # Interpolate sbright onto the HiReX-SR timebase:
    #         sbright_interp = scipy.zeros((len(self.run_data.hirex_time_combined), sbright.shape[1]))
    #         # Use postinj to zero out before the injection:
    #         postinj = self.run_data.hirex_time_combined >= param_source[0]
    #         for k in xrange(0, sbright.shape[1]):
    #             sbright_interp[postinj, k] = scipy.interpolate.InterpolatedUnivariateSpline(
    #                 time_s,
    #                 sbright[:, k]
    #             )(self.run_data.hirex_time_combined[postinj])
    #
    #         if steady_ar is None:
    #             # Interpolate vbright onto the VUV timebase(s):
    #             vbright_interp = scipy.zeros((self.run_data.vuv_times_combined.shape[1], vbright.shape[1]))
    #             for k in xrange(0, vbright.shape[1]):
    #                 postinj = self.run_data.vuv_times_combined[k, :] >= param_source[1]
    #                 vbright_interp[postinj, k] = scipy.interpolate.InterpolatedUnivariateSpline(
    #                     time_v,
    #                     vbright[:, k]
    #                 )(self.run_data.vuv_times_combined[k, postinj])
    #
    #         # Plot the data if requested:
    #         # Argon data:
    #         if steady_ar is not None and (debug_plots or ar_f is not None):
    #             if ar_f is None:
    #                 ar_f, ar_a = self.run_data.plot_ar(boxplot=True, norm=True)
    #             ar_a.plot(range(0, self.run_data.ar_signal.shape[1]), sbright[-1, :], color=lc, label=label, alpha=alpha)
    #             ar_f.canvas.draw()
    #
    #         # Big HiReX-SR plot:
    #         if debug_plots or big_f is not None:
    #             if big_f is None:
    #                 if steady_ar is None:
    #                     big_f, a_H, a_V = self.run_data.plot_data()
    #                 else:
    #                     # Make the figure to hold the time-series data:
    #                     big_f = plt.figure()
    #                     ncol = 6
    #                     nrow = int(scipy.ceil(1.0 * self.run_data.ar_signal.shape[1] / ncol))
    #                     gs = mplgs.GridSpec(nrow, ncol)
    #                     a_H = []
    #                     i_col = 0
    #                     i_row = 0
    #                     for k in xrange(0, self.run_data.ar_signal.shape[1]):
    #                         a_H.append(big_f.add_subplot(gs[i_row, i_col]))
    #                         i_col += 1
    #                         if i_col >= ncol:
    #                             i_col = 0
    #                             i_row += 1
    #
    #                     for k in xrange(0, len(a_H)):
    #                         a_H[k].set_xlabel('$t$ [s]')
    #                         a_H[k].set_ylabel('$b$ [AU]')
    #                         a_H[k].set_title('HiReX-SR chord %d' % (k,))
    #                         # a_H[k].set_ylim(bottom=0)
    #             for a, k in zip(a_H, range(0, len(a_H))):
    #                 l = a.plot(time_s, sbright[:, k], color=lc, alpha=alpha)
    #                 a.plot(
    #                     self.run_data.hirex_time_combined,
    #                     sbright_interp[:, k],
    #                     '.',
    #                     color=plt.getp(l[0], 'color'),
    #                     alpha=alpha
    #                 )
    #             if steady_ar is None:
    #                 for a, k in zip(a_V, range(0, len(a_V))):
    #                     l = a.plot(time_v, vbright[:, k], label=label, color=lc, alpha=alpha)
    #                     a.plot(
    #                         self.run_data.vuv_times_combined[k, :],
    #                         vbright_interp[:, k],
    #                         '.',
    #                         color=plt.getp(l[0], 'color'),
    #                         alpha=alpha
    #                     )
    #             big_f.canvas.draw()
    #
    #         # Form the squared differences and chi**2 itself:
    #         if steady_ar is None:
    #             s_squared_diff = (
    #                 (sbright_interp - self.run_data.hirex_signal_norm_combined) /
    #                 self.run_data.hirex_uncertainty_norm_combined
    #             )**2.0
    #             # TODO: This breaks if LoWEUS isn't present or is in a different
    #             # spot or has different signs.
    #             if True or self.include_loweus:
    #                 v_squared_diff = (
    #                     (vbright_interp - self.run_data.vuv_signals_norm_combined.T) /
    #                     self.run_data.vuv_uncertainties_norm_combined.T
    #                 )**2.0
    #             else:
    #                 v_squared_diff = (
    #                     (vbright_interp[:, :-1] - self.run_data.vuv_signals_norm_combined[:-1, :].T) /
    #                     self.run_data.vuv_uncertainties_norm_combined[:-1, :].T
    #                 )**2.0
    #
    #             # Storing the elements of chi**2 like this will enable us to
    #             # compute figures of merit like the WAIC.
    #             chi2_arr = scipy.hstack(
    #                 (
    #                     s_squared_diff[~self.run_data.hirex_flagged_combined],
    #                     v_squared_diff.ravel()
    #                 )
    #             )
    #             # Go ahead and do the proper normalizations:
    #             ll = -0.5 * chi2_arr.sum() + self.ll_normalization
    #         else:
    #             ar_mask = (self.run_data.ar_time >= self.time_1) & (self.run_data.ar_time <= self.time_2)
    #             ar_signal = self.run_data.ar_signal[ar_mask, :]
    #             ar_uncertainty = self.run_data.ar_uncertainty[ar_mask, :]
    #             ar_flagged = self.run_data.ar_flagged[ar_mask, :]
    #             ar_sim = scipy.tile(sbright[-1, :], (ar_signal.shape[0], 1))
    #             s_squared_diff = ((ar_sim - ar_signal) / ar_uncertainty)**2.0
    #             chi2_arr = s_squared_diff.ravel()[~ar_flagged.ravel()]
    #             ll = -0.5 * chi2_arr.sum() + self.ar_ll_normalization
    #
    #         if no_prior:
    #             lnprob = ll
    #         else:
    #             lnprob += ll
    #
    #         if scipy.isnan(lnprob):
    #             lnprob = -scipy.inf
    #
    #         # Package the blob: store brightnesses only. The differences, D,
    #         # V, and s are cheap to compute, so this will help keep the
    #         # memory footprint down with many samples.
    #         if return_blob:
    #             if light_blob:
    #                 blob = (ll,)
    #             else:
    #                 blob = (
    #                     ll,
    #                     sbright,
    #                     vbright if steady_ar is None else None,
    #                     time,
    #                     out
    #                 )
    #
    #     t_elapsed = time_.time() - t_start
    #     print("Done. Elapsed time is %.3fs, ll=%.4g, lp=%.4g, lprior=%.4g" % (t_elapsed, ll, lnprob, lnprob - ll,))
    #
    #     if return_blob:
    #         return (sign * lnprob, blob)
    #     else:
    #         return sign * lnprob
    
    def explore_optima_GMO(self, samples=100, algo=None, gp=None):
        """Probe the space for local optima using PyGMO.
        
        Parameters
        ----------
        samples : int
            Number of samples to take.
        algo : :py:class:`PyGMO.algorithm` instance
            Local optimizer to use.
        gp : :py:class:`gptools.GaussianProcess` instance
            If present, the starting points for the optimizer are taken to be
            the `samples` best points in the training data of `gp` (which is
            typically read using :py:meth:`assemble_surrogate`).
        """
        if algo is None:
            algo = PyGMO.algorithm.nlopt_sbplx(max_iter=10000)
        prob = MAPProblem(self)
        if gp is not None:
            pop = PyGMO.population(prob)
            sort_args = gp.y.argsort()[::-1]
            sort_args = sort_args[:samples]
            X = gp.X[sort_args]
            for x in X:
                try:
                    pop.push_back(x)
                except ValueError:
                    print("Incompatible sample skipped!")
            inspector = PyGMO.util.analysis(pop, npoints='all')
        else:
            inspector = PyGMO.util.analysis(prob, samples)
        inspector.local_search(clusters_to_show='all', plot_separate_pcp=False, algo=algo)
        return inspector
    
    def iterate_inspector(self, inspector, algo=None, no_cluster=True, mask=None, thresh=-2e5):
        """Start a new local optima search using the current cluster centers as
        starting points.
        """
        if algo is None:
            algo = PyGMO.algorithm.nlopt_sbplx(max_iter=10000)
        prob = MAPProblem(self)
        pop = PyGMO.population(prob)
        bounds = scipy.asarray(self.get_prior().bounds[:], dtype=float)
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        
        if no_cluster:
            lp_vals = scipy.asarray(inspector.local_f)[:, 0]
            lp_vals = -1 * (lp_vals * inspector.f_span + inspector.f_offset)
            X = scipy.asarray(inspector.local_extrema)
            for k in xrange(0, len(lb)):
                X[:, k] = X[:, k] * (ub[k] - lb[k]) + lb[k]
            if mask is None:
                mask = scipy.ones_like(lp_vals, dtype=bool)
            for lp, x, m in zip(lp_vals, X, mask):
                if m and (lp >= thresh):
                    pop.push_back(x)
        else:
            for x in inspector.local_cluster_x_centers:
                # Need to remove the normalization before putting in place:
                pop.push_back([xn * (u - l) + l for xn, u, l in zip(x, ub, lb)])
        
        inspector = PyGMO.util.analysis(pop, 'all')
        inspector.local_search(clusters_to_show='all', plot_separate_pcp=False, algo=algo)
        return inspector
    
    def visualize_all_minima(self, inspector, thresh=-2e5, plot_histories=False):
        """Plot of the of the candidate local minima, along with scatterplots and histograms.
        
        Returns a mask of the "bad" fits, defined according to certain internal
        criteria.
        """
        x = scipy.asarray(inspector.local_extrema)
        # Remove the normalization:
        lb = inspector.lb
        ub = inspector.ub
        # bounds = scipy.asarray(self.get_prior().bounds[:], dtype=float)
        # lb = bounds[:, 0]
        # ub = bounds[:, 1]
        for k in xrange(0, len(lb)):
            x[:, k] = x[:, k] * (ub[k] - lb[k]) + lb[k]
        lp_vals = scipy.asarray(inspector.local_f)[:, 0]
        # Remove the normalization:
        lp_vals = -1 * (lp_vals * inspector.f_span + inspector.f_offset)
        max_lp = lp_vals.max()
        min_lp = lp_vals.min()
        
        f = plt.figure()
        a_D = f.add_subplot(2, 1, 1)
        a_V = f.add_subplot(2, 1, 2, sharex=a_D)
        
        f.suptitle("local extrema identified")
        a_V.set_xlabel('$r/a$')
        a_D.set_ylabel('$D$ [m$^2$/s]')
        a_V.set_ylabel('$V$ [m/s]')
        
        # TODO: Make this generalized!
        eig_D = []
        eig_V = []
        knots_D = []
        knots_V = []
        hp_D = []
        hp_mu_D = []
        hp_V = []
        param_scaling = []
        param_source = []
        
        for p in x:
            peig_D, peig_V, pknots_D, pknots_V, php_D, php_mu_D, php_V, pparam_scaling, pparam_source = self.split_params(p)
            eig_D.append(peig_D)
            eig_V.append(peig_V)
            knots_D.append(pknots_D)
            knots_V.append(pknots_V)
            hp_D.append(php_D)
            hp_mu_D.append(php_mu_D)
            hp_V.append(php_V)
            param_scaling.append(pparam_scaling)
            param_source.append(pparam_source)
        
        eig_D = scipy.asarray(eig_D)
        eig_V = scipy.asarray(eig_V)
        knots_D = scipy.asarray(knots_D)
        knots_V = scipy.asarray(knots_V)
        hp_D = scipy.asarray(hp_D)
        hp_mu_D = scipy.asarray(hp_mu_D)
        hp_V = scipy.asarray(hp_V)
        param_scaling = scipy.asarray(param_scaling)
        param_source = scipy.asarray(param_source)
        
        mask = (
            (lp_vals > thresh) &
            (knots_D != 0.0).all(axis=1) &
            (knots_V != 0.0).all(axis=1) &
            (knots_D != 1.05).all(axis=1) &
            (knots_V != 1.05).all(axis=1) &
            (eig_D != 30).all(axis=1) &
            (eig_V != 200).all(axis=1) &
            (eig_V != -200).all(axis=1) &
            (knots_D >= self.roa_grid_DV[1]).all(axis=1) &
            (knots_V >= self.roa_grid_DV[1]).all(axis=1) &
            (knots_D <= self.roa_grid_DV[-2]).all(axis=1) &
            (knots_V <= self.roa_grid_DV[-2]).all(axis=1)
        )
        
        # mask = (
        #     (lp_vals > thresh) &
        #     (x[:, 12:18] != 0.0).all(axis=1) &
        #     (x[:, 12:18] != 1.05).all(axis=1) &
        #     (x[:, 0:6] != 30).all(axis=1) &
        #     (x[:, 6:12] != 200).all(axis=1) &
        #     # (x[:, 0] <= 10) &
        #     # (x[:, 5] <= 25) &
        #     (x[:, 6:12] != -200).all(axis=1) &
        #     # (x[:, 12] != x[:, 13]) &
        #     # (x[:, 12] != x[:, 13]) &
        #     (x[:, 12:18] >= self.roa_grid_DV[1]).all(axis=1) &
        #     (x[:, 12:18] <= self.roa_grid_DV[-2]).all(axis=1)
        # )
        max_adjusted_lp = lp_vals[mask].max()
        
        # Correct the knot ordering:
        # TODO: Generalize this, too!
        # for i in xrange(0, len(x)):
        #     if x[i, 10] > x[i, 11]:
        #         tmp = x[i, 10]
        #         x[i, 10] = x[i, 11]
        #         x[i, 11] = tmp
        #     if x[i, 12] > x[i, 13]:
        #         tmp = x[i, 12]
        #         x[i, 12] = x[i, 13]
        #         x[i, 13] = tmp
        
        for lp, X, m in zip(lp_vals, x, mask):
            # if (lp >= -2e5) and not (X[10:14] == 0.0).any() and not (X[10:14] == 1.05).any() and not (X[0:5] == 50).any() and not (X[5:10] == 200).any():
            # if (X[10:14] == 0.0).any() and (X[10:14] == 1.05).any():
            #     color = 'm'
            # elif (X[10:14] == 0.0).any():
            #     color = 'g'
            # elif (X[10:14] == 1.05).any():
            #     color = 'r'
            # else:
            #     color = 'k'
            if not m:
                continue
                color = 'r'
                lw = 1
            # elif (X[10:14] <= self.roa_grid_DV[1]).any() or (X[10:14] >= self.roa_grid_DV[-2]).any():
            #     color = 'm'
            #     lw = 0.05
            elif lp == max_adjusted_lp:
                color = 'b'
                lw = 3
            else:
                color = 'k'
                lw = 1
            D, V = self.eval_DV(X)
            a_D.plot(self.roa_grid_DV, D, color, alpha=max_lp / lp, linewidth=lw)
            a_V.plot(self.roa_grid_DV, V, color, alpha=max_lp / lp, linewidth=lw)
        
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        n, bins, patches = a.hist(lp_vals[lp_vals > thresh], bins=25, facecolor='b')
        # a.hist(lp_vals[(x[:, 10:14] == 1.05).any(axis=1)], bins=bins, facecolor='r')
        # a.hist(lp_vals[(x[:, 10:14] == 0.0).any(axis=1)], bins=bins, facecolor='g')
        
        # Make a scatterplot:
        x_mod = scipy.asarray([x[mask, :],])
        gptools.plot_sampler(
            x_mod,
            labels=self.get_labels(),
            plot_samples=True,
            plot_hist=True
        )
        
        if plot_histories:
            f, a_H, a_V = self.run_data.plot_data()
            for lp, X, m in zip(lp_vals, x, mask):
                if m:
                    if lp == max_adjusted_lp:
                        color = 'b'
                    else:
                        color = 'k'
                    self.compute_ln_prob(X, lc=color, big_f=f, a_H=a_H, a_V=a_V, alpha=max_lp / lp)
        
        return (mask, lp_vals, x)
    
    def process_local_search(self, prefix):
        """Plot the objective function against iteration number for a repeated local search.
        """
        files = glob.glob('../' + prefix + '*.pkl')
        N = [int(re.search('.*?([0-9]+)\.pkl', f).group(1)) for f in files]
        N.sort()
        
        lp_vals = []
        for n in N:
            print(n)
            with open('../' + prefix + str(n) + '.pkl', 'rb') as f:
                i = pkl.load(f)
            
            lp = scipy.asarray(i.local_f)[:, 0]
            # Remove the normalization:
            lp = lp * i.f_span + i.f_offset
            # Remove infinite entries, since the mess up the plotting:
            lp = lp[scipy.absolute(lp) != sys.float_info.max]
            lp_vals.append(-1 * scipy.sort(lp))
            
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        # We know the first file will have all of the points:
        for k in range(0, len(lp_vals[0])):
            lp = [-1 * l[k] for l in lp_vals if k < len(l)]
            a.semilogy(range(1, len(lp) + 1), lp, '.-', label='k')
        a.set_xlabel('iteration')
        a.set_ylabel('-1 * log-posterior')
        a.set_title('convergence of local search')
        f.canvas.draw()
        
        return lp_vals
    
    def process_inspector(self, inspector, archi=None):
        """Plot the candidate local extrema.
        """
        # In principle, these should have been computed -- but something went
        # wrong on at least one trial, so instead I will loop over all of the
        # clusters and evaluate it myself:
        pool = multiprocessing.Pool(24)
        ll_eval = _ComputeLnProbWrapper(self, make_dir=True, for_min=True, denormalize=True)
        ll_vals = pool.map(ll_eval, inspector.local_cluster_x_centers)
        pool.close()
        ll_vals = -1 * scipy.asarray(ll_vals, dtype=float)
        normalized = scipy.ones_like(ll_vals, dtype=bool)
        
        X = inspector.local_cluster_x_centers
        
        # If an archipelago is provided, add on its info:
        if archi is not None:
            champs = [i.population.champion for i in archi]
            ll_vals = scipy.concatenate((ll_vals, [-1 * c.f[0] for c in champs]))
            normalized = scipy.concatenate((normalized, scipy.zeros_like(champs, dtype=bool)))
            X = scipy.concatenate((X, [c.x for c in champs]))
        
        max_ll = ll_vals.max()
        
        f = plt.figure()
        a_D = f.add_subplot(2, 1, 1)
        a_V = f.add_subplot(2, 1, 2, sharex=a_D)
        bounds = scipy.asarray(self.get_prior().bounds[:], dtype=float)
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        
        # f_ar, a_ar = self.run_data.plot_ar(boxplot=True, norm=True)
        
        for ll, x, n in zip(ll_vals, X, normalized):
            if n:
                xd = [xn * (u - l) + l for xn, u, l in zip(x, ub, lb)]
            else:
                xd = x
            
            D, V = self.eval_DV(xd)
            a_D.plot(self.roa_grid_DV, D, 'k' if n else 'b', alpha=max_ll / ll)
            a_V.plot(self.roa_grid_DV, V, 'k' if n else 'b', alpha=max_ll / ll)
            
            if ll == max_ll:
                print('best!')
                x_best = xd
            
            # self.compute_ln_prob(xd, debug_plots=True, steady_ar=1e17, ar_f=f_ar, ar_a=a_ar)
        
        D, V = self.eval_DV(x_best)
        a_D.plot(self.roa_grid_DV, D, 'r', linewidth=4)
        a_V.plot(self.roa_grid_DV, V, 'r', linewidth=4)
        
        a_V.set_xlabel('$r/a$')
        a_D.set_ylabel('$D$ [m$^2$/s]')
        a_V.set_ylabel('$V$ [m/s]')
        f.suptitle("Comparison of local extrema")
        
        return (ll_vals, max_ll, x_best)
    
    def find_MAP_estimate_GMO(self):
        """Globally optimize the log-posterior using PyGMO.
        """
        print("Starting global optimizer...")
        # TODO: This needs to be tuned!
        t_start = time_.time()
        prob = MAPProblem(self)
        # prob = PyGMO.problem.death_penalty(prob)
        algo = PyGMO.algorithm.de(gen=10)
        archi = PyGMO.archipelago(algo, prob, 24, 20)#, topology=PyGMO.topology.ring())
        archi.evolve(10)
        archi.join()
        t_elapsed = time_.time() - t_start
        print("Done. Elapsed time is %.1fs" % (t_elapsed,))
        return (prob, algo, archi)
    
    def process_archi(self, archi):
        """Make a plot of the possible global optima in an archipelago.
        
        The darkness of a curve is proportional to its fitness.
        """
        champs = [i.population.champion for i in archi]
        ll = [-1 * c.f[0] for c in champs]
        x = [c.x for c in champs]
        
        max_ll = max(ll)
        print("max log-probability is %.3g" % (max_ll,))
        
        f = plt.figure()
        a_D = f.add_subplot(2, 1, 1)
        a_V = f.add_subplot(2, 1, 2, sharex=a_D)
        for LL, X in zip(ll, x):
            D, V = self.eval_DV(X)
            a_D.plot(self.roa_grid_DV, D, 'r' if LL == max_ll else 'k', alpha=max_ll / LL)
            a_V.plot(self.roa_grid_DV, V, 'k', alpha=max_ll / LL)
        
        a_V.set_xlabel('$r/a$')
        a_D.set_ylabel('$D$ [m$^2$/s]')
        a_V.set_ylabel('$V$ [m/s]')
        f.suptitle("Comparison of solutions found")
    
    def find_MAP_estimate(self, random_starts=None, num_proc=None, pool=None, theta0=None, thresh=None):
        """Find the most likely parameters given the data.
        
        Parameters
        ----------
        random_starts : int, optional
            The number of times to start the optimizer at a random point in
            order to find the global optimum. If set to None (the default), a
            number equal to twice the number of processors will be used.
        num_proc : int, optional
            The number of cores to use. If set to None (the default), half of
            the available cores will be used.
        pool : :py:class:`Pool`, optional
            The multiprocessing pool to use. You should produce this with
            :py:func:`make_pool` so that the correct directories are in place.
        theta0 : array of float, optional
            The initial guess(es) to use. If not specified, random draws from
            the prior will be used. This overrides `random_starts` if present.
            This array have shape (`num_starts`, `ndim`) or (`ndim`,).
        thresh : float, optional
            The threshold to continue with the optimization.
        """
        if num_proc is None:
            num_proc = multiprocessing.cpu_count() // 2
        if num_proc > 1 and pool is None:
            pool = make_pool(num_proc=num_proc)
        if random_starts is None:
            random_starts = 2 * num_proc
        prior = self.get_prior()
        if theta0 is None:
            param_samples = prior.random_draw(size=random_starts).T
        else:
            param_samples = scipy.atleast_2d(theta0)
        t_start = time_.time()
        if pool is not None:
            res = pool.map(
                _OptimizeEval(self, thresh=thresh),
                param_samples
            )
        else:
            res = map(
                _OptimizeEval(self, thresh=thresh),
                param_samples
            )
        t_elapsed = time_.time() - t_start
        print("All done, wrapping up!")
        # res = [r for r in res if r is not None]
        # TODO: Implement this for the new fmin_l_bfgs_b-based version...
        # if res:
        #     res_min = min(res, key=lambda r: r.fun)
        # else:
        #     res_min = None
        
        # D, V = self.eval_DV(res_min.x, plot=True)
        # impath_DV = '../DV_%d_%d.pdf' % (self.shot, self.version,)
        # plt.gcf().savefig(impath_DV)
        
        # if res_min is not None:
        #     lp = self.DV2ln_prob(self.get_prior().sample_u(res_min.x), debug_plots=True)
        # impath_data = '../data_%d_%d.pdf' % (self.shot, self.version,)
        # plt.gcf().savefig(impath_data)
        
        send_email("MAP update", "MAP estimate done.", [])#, [impath_DV, impath_data])
        
        print("MAP estimate complete. Elapsed time is %.2fs. Got %d completed." % (t_elapsed, len(res)))
        
        # return (res_min, res)
        return res
    
    def MAP_from_SQL(self):
        """Get a job to do from the SQL server.
        
        Only runs a single job. This function is designed to allow multiple
        computers to run in parallel for controlled periods of time, without
        running for "too long."
        
        The computer connects to the database hosted on juggernaut.psfc.mit.edu,
        gets a named lock called "bayesimplock", finds the current best unsolved
        case (i.e., status is -100, 5 or 6) and marks it as being in progress
        before releasing the lock. It then pulls in the parameters and starts
        the optimizer. When the optimizer is done, it checks to make sure the
        result was actually improved. If it was, it puts the new results in.
        Otherwise it just sets it back to the old status in the hopes that a run
        with more time will be able to make some progress. If bayesimp crashes,
        it will attempt to mark the case with status 10. Sometimes the catch
        statement fails to execute, in which case the case will be left marked
        with status 0 (in progress).
        """
        db = get_connection()
        c = db.cursor()
        
        # Acquire a mutex lock on the database:
        c.execute("SELECT GET_LOCK('bayesimplock', 600);")
        status, = c.fetchone()
        if not status:
            raise ValueError("Failed to get lock in a reasonable amount of time!")
        # Find the most promising case which hasn't been worked on yet:
        c.execute(
            """SELECT case_id
            FROM results
            WHERE status = -100 OR status = 5 OR status = 6
            ORDER BY log_posterior DESC
            LIMIT 1;"""
        )
        case_id, = c.fetchone()
        # Set the status to 0:
        c.execute(
            """UPDATE results
            SET status = 0
            WHERE case_id = %s
            """,
            (case_id,)
        )
        db.commit()
        c.execute("SELECT RELEASE_LOCK('bayesimplock');")
        
        try:
            c.execute(
                """SELECT ending_params, log_posterior, log_likelihood, status
                FROM results
                WHERE case_id = %s""",
                (case_id,)
            )
            params, old_lp, old_ll, old_status = c.fetchone()
            # Close the connection so it doesn't crash out:
            c.close()
            db.close()
            params = scipy.loads(params)
            setup_working_dir()
            res, = self.find_MAP_estimate(random_starts=0, num_proc=1, theta0=params)
            new_params = self.get_prior().sample_u(res[0])
            lp = res[1]
            ll = res[1] - self.get_prior()(new_params)
            new_status = res[2]
            
            # Only put it in the tree if it improved:
            if lp < old_lp:
                new_params = params
                new_status = old_status
                ll = old_ll
                lp = old_lp
            
            # Re-establish the connection:
            db = get_connection()
            c = db.cursor()
            # c.execute("SELECT GET_LOCK('bayesimplock', 600);")
            # status, = c.fetchone()
            # if not status:
            #     raise ValueError("Failed to get lock in a reasonable amount of time!")
            # Put the new results in:
            c.execute(
                """UPDATE results
                SET status = %s, ending_params = %s, log_likelihood = %s, log_posterior = %s
                WHERE case_id = %s
                """,
                (int(new_status), new_params.dumps(), float(ll), float(lp), case_id)
            )
            db.commit()
            # c.execute("SELECT RELEASE_LOCK('bayesimplock');")
        except:
            print("Abnormal exit!")
            traceback.print_exc()
            db = get_connection()
            c = db.cursor()
            # c.execute("SELECT GET_LOCK('bayesimplock', 600);")
            # status, = c.fetchone()
            # if not status:
            #     raise ValueError("Failed to get lock in a reasonable amount of time!")
            # Set the status to 10 to indicate an error:
            c.execute(
                """UPDATE results
                SET status = 10
                WHERE case_id = %s
                """,
                (case_id,)
            )
            db.commit()
            # c.execute("SELECT RELEASE_LOCK('bayesimplock');")
        finally:
            cleanup_working_dir()
            
            c.close()
            db.close()
            print("All done!")
    
    def process_MAP_from_SQL(self, n=20, plot=False, filter_bad=True, compute_dlines=False):
        """Get the top `n` cases from the SQL database.
        
        Pulls the top `n` cases for which the status is not 10 (bayesimp error).
        Computes D, V for each case and makes a plot with them color-coded
        according to their log-posterior. Also computes the true D, V.
        
        Parameters
        ----------
        n : int, optional
            The number of cases to plot. Default is 20.
        plot : bool, optional
            If True, plot the solutions. Default is False.
        filter_bad : bool, optional
            If True, remove probably bad solutions. This includes solutions
            which have knots too close to the boundaries. Default is True.
        compute_dlines : bool, optional
            If True, the params will be passed through :py:meth:`DV2dlines`
            to compute the line brightness profiles. The default is False.
        """
        db = get_connection()
        c = db.cursor()
        
        c.execute(
            """SELECT log_posterior, status, ending_params
            FROM results
            WHERE status != 10 AND status != -100 AND log_posterior >= -45000000 AND status != 6 AND status != 5 AND status !=0
            ORDER BY log_posterior DESC
            LIMIT %s
            """,
            (n,)
        )
        res = c.fetchall()
        
        lp = scipy.asarray([r[0] for r in res], dtype=float)
        status = scipy.asarray([r[1] for r in res], dtype=int)
        params = scipy.asarray([scipy.loads(r[2]) for r in res], dtype=float)
        
        # Sort the knots:
        for i in range(params.shape[0]):
            params[
                i,
                self.num_eig_D + self.num_eig_V:
                self.num_eig_D + self.num_eig_V + self.num_eig_D -
                self.spline_k_D
            ] = scipy.sort(
                params[
                    i,
                    self.num_eig_D + self.num_eig_V:
                    self.num_eig_D + self.num_eig_V + self.num_eig_D -
                    self.spline_k_D
                ]
            )
            params[
                i,
                self.num_eig_D + self.num_eig_V + self.num_eig_D -
                    self.spline_k_D:
                self.num_eig_D + self.num_eig_V + self.num_eig_D -
                    self.spline_k_D + self.num_eig_V - self.spline_k_V
            ] = scipy.sort(
                params[
                    i,
                    self.num_eig_D + self.num_eig_V + self.num_eig_D -
                        self.spline_k_D:
                    self.num_eig_D + self.num_eig_V + self.num_eig_D -
                        self.spline_k_D + self.num_eig_V - self.spline_k_V
                ]
            )
        
        if filter_bad:
            is_bad = scipy.zeros_like(lp, dtype=bool)
            for i, p in enumerate(params):
                eig_D, eig_V, knots_D, knots_V, hp_D, hp_mu_D, hp_V, param_scaling, param_source = self.split_params(p)
                is_bad[i] = (
                    knots_D.min() <= self.roa_grid_DV[3] or
                    knots_D.max() >= self.roa_grid_DV[-4] or
                    knots_V.min() <= self.roa_grid_DV[3] or
                    knots_V.max() >= self.roa_grid_DV[-4] or
                    eig_V[-1] > 0 or
                    scipy.diff(knots_D).min() < (self.roa_grid_DV[1] - self.roa_grid_DV[0]) or
                    scipy.diff(knots_V).min() < (self.roa_grid_DV[1] - self.roa_grid_DV[0])
                )
            mask = ~is_bad
            params = params[mask, :]
            lp = lp[mask]
            status = status[mask]
        
        D = scipy.zeros((len(params), len(self.roa_grid_DV)), dtype=float)
        V = scipy.zeros((len(params), len(self.roa_grid_DV)), dtype=float)
        
        D_true, V_true = self.eval_DV(self.params_true)
        
        if compute_dlines:
            dlines_true, sqrtpsinorm, time = self.DV2dlines(self.params_true, return_rho_t=True)
            dlines = scipy.zeros(scipy.concatenate(([len(params),], dlines_true.shape)))
            
            sbright_true, vbright_true, xtomobright_true = self.dlines2sig(self.params_true, dlines_true, time)
            sbright = scipy.zeros(scipy.concatenate(([len(params),], sbright_true.shape)))
            vbright = scipy.zeros(scipy.concatenate(([len(params),], vbright_true.shape)))
            xtomobright = {}
            for k, v in xtomobright_true.iteritems():
                xtomobright[k] = scipy.zeros(scipy.concatenate(([len(params),], v.shape)))
        
        if plot:
            f = plt.figure()
            a_D = f.add_subplot(2, 1, 1)
            a_V = f.add_subplot(2, 1, 2, sharex=a_D)
        
        for i, (p, lpv, s) in enumerate(zip(params, lp, status)):
            D[i, :], V[i, :] = self.eval_DV(p)
            if compute_dlines:
                dlines[i] = self.DV2dlines(p)
                sbright[i], vbright[i], x = self.dlines2sig(p, dlines[i], time)
                for k, v in x.iteritems():
                    xtomobright[k][i] = v
            if plot:
                if s == 3:
                    lc = 'g'
                elif s == 4:
                    lc = 'b'
                elif s == 0:
                    lc = 'k'
                else:
                    lc = 'r'
                a_D.plot(self.roa_grid_DV, D[i, :], color=lc, alpha=(lpv - lp.min()) / (lp.max() - lp.min()))
                a_V.plot(self.roa_grid_DV, V[i, :], color=lc, alpha=(lpv - lp.min()) / (lp.max() - lp.min()))
        
        if plot:
            # Overplot true solution:
            a_D.plot(self.roa_grid_DV, D_true, color='orange', lw=3)
            a_V.plot(self.roa_grid_DV, V_true, color='orange', lw=3)
            
            a_D.set_ylabel('$D$ [m$^2$/s]')
            a_V.set_xlabel('$r/a$')
            a_V.set_ylabel('$V$ [m/s]')
            a_D.set_title("Results from local optima search")
        
        c.close()
        db.close()
        
        if compute_dlines:
            return (
                params, lp, status, D, V, D_true, V_true, dlines_true, dlines,
                sqrtpsinorm, time, sbright_true, vbright_true, xtomobright_true,
                sbright, vbright, xtomobright
            )
        else:
            return params, lp, status, D, V, D_true, V_true
    
    def make_MAP_solution_line_plot(self, params, lp, status):
        """Make a solution line plot from the output of :py:math:`process_MAP_from_SQL`.
        
        The parameter samples are mapped to [0, 1] so they all fit on the same
        axis. The lines are color-coded according to their status and the alpha
        is selected based on the log-posterior. The true solution is shown in
        orange with large squares and boxplots for each parameter are
        superimposed to show how good/bad the fit is doing reaching the true
        parameters.
        
        Parameters
        ----------
        params : array, (`n`, `num_params`)
            The parameters to use.
        lp : array, (`n`,)
            The log-posterior values.
        status : array, (`n`,)
            The status of each fit, as stored in the MySQL database.
        """
        u = [self.get_prior().elementwise_cdf(p) for p in params]
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        x = range(params.shape[1])
        for uv, lpv, s in zip(u, lp, status):
            if s == 3:
                lc = 'g'
            elif s == 4:
                lc = 'b'
            elif s == 0:
                lc = 'k'
            else:
                lc = 'r'
            a.plot(x, uv, 'o:', markersize=8, color=lc, alpha=(lpv - lp.min()) / (lp.max() - lp.min()))
        
        u_true = self.get_prior().elementwise_cdf(self.params_true)
        a.plot(x, u_true, 's-', markersize=12, color='orange')
        a.boxplot(scipy.asarray(u, dtype=float), positions=x)
        a.set_xlabel('parameter')
        a.set_ylabel('$u=F_P(p)$')
        a.set_xticks(x)
        a.set_xticklabels(self.get_labels())
    
    def make_MAP_solution_hist_plot(self, params, nbins=None):
        """Make histogram "fingerprint" plot of the parameter samples.
        
        Parameters
        ----------
        params : array, (`n`, `num_params`)
            The parameters to plot.
        nbins : int or array of int, (`D`,), optional
            The number of bins dividing [0, 1] to use for each histogram. If a
            single int is given, this is used for all of the hyperparameters. If an
            array of ints is given, these are the numbers of bins for each of the
            hyperparameters. The default is to determine the number of bins using
            the Freedman-Diaconis rule.
        """
        gptools.plot_sampler_fingerprint(
            params[None, :, :],
            self.get_prior(),
            labels=self.get_labels(),
            points=self.params_true,
            point_color='r',
            rot_x_labels=True
        )
    
    def make_MAP_solution_corr_plot(self, params):
        """Make a plot of the correlation matrix of the parameter samples.
        
        Parameters
        ----------
        params : array, (`n`, `num_params`)
            The parameters to plot.
        """
        gptools.plot_sampler_cov(
            params[None, :, :],
            labels=self.get_labels(),
            rot_x_labels='True'
        )
    
    def make_MAP_param_dist_plot(self, params):
        """Make a plot of the univariate and bivariate histograms of the parameter samples.
        
        Parameters
        ----------
        params : array, (`n`, `num_params`)
            The parameters to plot.
        """
        gptools.plot_sampler(
            params[None, :, :],
            labels=self.get_labels(),
            points=self.params_true
        )
    
    def make_MAP_slider_plot(
            self, params, lp, status, D, V, D_true, V_true,
            dlines_true=None, dlines=None, sqrtpsinorm=None, time=None,
            sbright_true=None, vbright_true=None, xtomobright_true=None,
            sbright=None, vbright=None, xtomobright=None
        ):
        """Make a plot which lets you explore the output of :py:meth:`process_MAP_from_SQL`.
        
        All of the cases are plotted as thin grey lines, and the true curve is
        plotted as a thick green line. The slider selects a pair of D, V curves
        and highlights them so that pairs of "wrong" fits can be explored.
        
        Parameters
        ----------
        params : array, (`n`, `num_params`)
            The parameters to use.
        lp : array, (`n`,)
            The log-posterior values.
        status : array, (`n`,)
            The status of each fit, as stored in the MySQL database.
        D : array, (`n`, `num_rho`)
            The D values for each set of params.
        V : array, (`n`, `num_rho`)
            The V values for each set of params.
        D_true : array, (`num_rho`,)
            The true values of D.
        V_true : array, (`num_rho`,)
            The true values of V.
        dlines_true : array, (`num_time`, `num_lines`, `num_space`), optional
            The true diagnostic emissivity profiles. If not present, will not be
            plotted.
        dlines : array, (`num_samp`, `num_time`, `num_lines`, `num_space`), optional
            The diagnostic emissivity profiles for each sample. If not present,
            will not be plotted.
        sqrtpsinorm : array, (`num_space`,), optional
            The sqrtpsinorm grid that `dlines` is given on. If not present, will
            not be plotted.
        time : array, (`num_time`,), optional
            The time grid that `dlines` is given on. If not present, will not be
            plotted.
        sbright_true : array, (`num_time`, `num_chords`), optional
            The true HiReX-SR signal.
        vbright_true : array, (`num_time`, `num_lines`), optional
            The true VUV signals.
        xtomobright_true : dict of array, (`num_time`, `num_chords`), optional
            The true XTOMO signals.
        sbright : array, (`num_samp`, `num_time`, `num_chords`), optional
            The HiReX-SR signals.
        vbright : array, (`num_samp`, `num_time`, `num_lines`), optional
            The VUV signals.
        xtomobright : dict of array, (`num_samp`, `num_time`, `num_chords`), optional
            The XTOMO signals.
        """
        def arrow_respond(t_slider, samp_slider, event):
            if event.key == 'right':
                if t_slider is not None:
                    t_slider.set_val(min(t_slider.val + 1, t_slider.valmax))
                else:
                    samp_slider.set_val(min(samp_slider.val + 1, samp_slider.valmax))
            elif event.key == 'left':
                if t_slider is not None:
                    t_slider.set_val(max(t_slider.val - 1, t_slider.valmin))
                else:
                    samp_slider.set_val(max(samp_slider.val - 1, samp_slider.valmin))
            elif event.key == 'up':
                samp_slider.set_val(min(samp_slider.val + 1, samp_slider.valmax))
            elif event.key == 'down':
                samp_slider.set_val(max(samp_slider.val - 1, samp_slider.valmin))
        
        f = plt.figure()
        
        if dlines is not None:
            outer_grid = mplgs.GridSpec(1, 3, width_ratios=[1, 2, 1])
        else:
            outer_grid = mplgs.GridSpec(1, 1)
        
        gs_DV = mplgs.GridSpecFromSubplotSpec(3, 1, outer_grid[0, 0], height_ratios=[5, 5, 1])
        a_D = f.add_subplot(gs_DV[0, 0])
        a_V = f.add_subplot(gs_DV[1, 0], sharex=a_D)
        a_s = f.add_subplot(gs_DV[2, 0])
        
        if dlines is not None:
            dlines_norm = dlines / dlines.max(axis=(1, 3))[:, None, :, None]
            dlines_norm_true = dlines_true / dlines_true.max(axis=(0, 2))[None, :, None]
            
            n_lines = dlines_true.shape[1]
            
            gs_dlines = mplgs.GridSpecFromSubplotSpec(
                n_lines + 1,
                2,
                outer_grid[0, 1],
                height_ratios=[5,] * n_lines + [1,]
            )
            
            a_lines = []
            for i in xrange(n_lines):
                a_lines.append(
                    f.add_subplot(
                        gs_dlines[i:i + 1, 0],
                        sharex=a_lines[0] if i > 0 else None
                    )
                )
                if i < n_lines - 1:
                    plt.setp(a_lines[-1].get_xticklabels(), visible=False)
                else:
                    a_lines[-1].set_xlabel(r"$\sqrt{\psi_{\mathrm{n}}}$")
                a_lines[-1].set_ylabel(r"$\epsilon$ [AU]")
                if i < n_lines - 1:
                    a_lines[-1].set_title(r"Ca$^{%d+}$, %.2f nm" % (self.atomdat[0][i], self.atomdat[1][i] / 10.0))
                else:
                    a_lines[-1].set_title("SXR")
            a_t_s = f.add_subplot(gs_dlines[-1, :])
            
            a_lines_norm = []
            for i in xrange(n_lines):
                a_lines_norm.append(
                    f.add_subplot(
                        gs_dlines[i:i + 1, 1],
                        sharex=a_lines[0]
                    )
                )
                if i < n_lines - 1:
                    plt.setp(a_lines_norm[-1].get_xticklabels(), visible=False)
                else:
                    a_lines_norm[-1].set_xlabel(r"$\sqrt{\psi_{\mathrm{n}}}$")
                a_lines_norm[-1].set_ylabel(r"normalized $\epsilon$ [AU]")
                if i < n_lines - 1:
                    a_lines_norm[-1].set_title(r"Ca$^{%d+}$, %.2f nm" % (self.atomdat[0][i], self.atomdat[1][i] / 10.0))
                else:
                    a_lines_norm[-1].set_title("SXR")
            
            gs_sig = mplgs.GridSpecFromSubplotSpec(
                2 + len(xtomobright),
                1,
                outer_grid[0, 2]
            )
            
            a_sr = f.add_subplot(gs_sig[0, 0])
            a_sr.set_title("HiReX-SR")
            a_sr.set_ylabel("normalized signal [AU]")
            a_sr.set_ylim(bottom=0)
            a_vuv = f.add_subplot(gs_sig[1, 0])
            a_vuv.set_title("VUV lines")
            a_vuv.set_ylabel("normalized signal [AU]")
            a_vuv.set_ylim(bottom=0)
            a_vuv.set_xlim(-0.5, 2.5)
            a_xtomo = []
            for i, k in enumerate(xtomobright.iterkeys()):
                a_xtomo.append(f.add_subplot(gs_sig[2 + i, 0]))
                a_xtomo[-1].set_title("XTOMO %d" % (k,))
                a_xtomo[-1].set_ylabel("normalized signal [AU]")
                a_xtomo[-1].set_ylim(bottom=0)
            a_xtomo[-1].set_xlabel("chord")
        
        plt.setp(a_D.get_xticklabels(), visible=False)
        a_V.set_xlabel(r"$r/a$")
        a_D.set_ylabel(r"$D$ [m$^2$/s]")
        a_V.set_ylabel(r"$V$ [m/s]")
        title = f.suptitle('')
        
        a_D.plot(self.roa_grid_DV, D_true, zorder=len(lp) + 1, color='g', lw=3)
        a_V.plot(self.roa_grid_DV, V_true, zorder=len(lp) + 1, color='g', lw=3)
        
        off_alpha = 0.25
        lines_D = a_D.plot(self.roa_grid_DV, D.T, color='k', alpha=off_alpha)
        lines_V = a_V.plot(self.roa_grid_DV, V.T, color='k', alpha=off_alpha)
        
        if dlines is not None:
            # Plot the raw emissivity data:
            lines_dlines = []
            lines_dlines_true = []
            for i, a in enumerate(a_lines):
                lines = a.plot(sqrtpsinorm, dlines[:, 0, i, :].T, color='k', alpha=off_alpha)
                lines_dlines.append(lines)
                line, = a.plot(sqrtpsinorm, dlines_true[0, i, :], color='g', zorder=len(lp) + 1, lw=3)
                lines_dlines_true.append(line)
                a.set_ylim(bottom=0.0)
            
            # Plot the normalized emissivity data:
            lines_dlines_norm = []
            lines_dlines_norm_true = []
            for i, a in enumerate(a_lines_norm):
                lines_norm = a.plot(sqrtpsinorm, dlines_norm[:, 0, i, :].T, color='k', alpha=off_alpha)
                lines_dlines_norm.append(lines_norm)
                line_norm, = a.plot(sqrtpsinorm, dlines_norm_true[0, i, :], color='g', zorder=len(lp) + 1, lw=3)
                lines_dlines_norm_true.append(line_norm)
                a.set_ylim(bottom=0.0)
            
            # Plot the line integrals:
            lines_sig_sr = a_sr.plot(
                range(sbright.shape[-1]),
                sbright[:, 0, :].T,
                color='k',
                alpha=off_alpha,
                ls='None',
                marker='o'
            )
            line_sig_sr_true, = a_sr.plot(
                range(sbright.shape[-1]),
                sbright_true[0, :],
                color='g',
                zorder=len(lp) + 1,
                ls='None',
                marker='s'
            )
            
            lines_sig_vuv = a_vuv.plot(
                range(vbright.shape[-1]),
                vbright[:, 0, :].T,
                color='k',
                alpha=off_alpha,
                ls='None',
                marker='o'
            )
            line_sig_vuv_true, = a_vuv.plot(
                range(vbright.shape[-1]),
                vbright_true[0, :],
                color='g',
                zorder=len(lp) + 1,
                ls='None',
                marker='s'
            )
            
            # Plot the actual data:
            t_idx_sr = profiletools.get_nearest_idx(time[0] - self.time_1, self.run_data.hirex_time_combined)
            line_data_sr, (erry_top_data_sr, erry_bot_data_sr), (barsy_data_sr,) = a_sr.errorbar(
                range(sbright.shape[-1]),
                self.run_data.hirex_signal_norm_combined[t_idx_sr, :],
                yerr=self.run_data.hirex_uncertainty_norm_combined[t_idx_sr, :],
                color='r',
                ls='None',
                marker='^',
                zorder=len(lp) + 2
            )
            
            # Assume all VUV diagnostics have the same timebase:
            t_idx_vuv = profiletools.get_nearest_idx(time[0] - self.time_1, self.run_data.vuv_times_combined[0, :])
            line_data_vuv, (erry_top_data_vuv, erry_bot_data_vuv), (barsy_data_vuv,) = a_vuv.errorbar(
                range(vbright.shape[-1]),
                self.run_data.vuv_signals_norm_combined[:, t_idx_vuv],
                yerr=self.run_data.vuv_uncertainties_norm_combined[:, t_idx_vuv],
                color='r',
                ls='None',
                marker='^',
                zorder=len(lp) + 2
            )
            
            line_data_xtomo = {}
            erry_top_data_xtomo = {}
            erry_bot_data_xtomo = {}
            barsy_data_xtomo = {}
            
            for i, (k, s) in enumerate(self.run_data.xtomo_signal_norm_combined.iteritems()):
                t_idx_xtomo = profiletools.get_nearest_idx(time[0] - self.time_1, self.run_data.xtomo_times_combined[k])
                line_data_xtomo[k], (erry_top_data_xtomo[k], erry_bot_data_xtomo[k]), (barsy_data_xtomo[k],) = a_xtomo[i].errorbar(
                    range(xtomobright[k].shape[-1]),
                    s[:, t_idx_xtomo],
                    yerr=0.1 * s[:, t_idx_xtomo],
                    color='r',
                    ls='None',
                    marker='^',
                    zorder=len(lp) + 2
                )
            
            lines_xtomo = {}
            lines_xtomo_true = {}
            for k, a in zip(xtomobright.iterkeys(), a_xtomo):
                lines_xtomo[k] = a.plot(
                    range(xtomobright[k].shape[-1]),
                    xtomobright[k][:, 0, :].T,
                    color='k',
                    alpha=off_alpha,
                    ls='None',
                    marker='o'
                )
                lines_xtomo_true[k], = a.plot(
                    range(xtomobright[k].shape[-1]),
                    xtomobright_true[k][0, :],
                    color='g',
                    zorder=len(lp) + 1,
                    ls='None',
                    marker='s'
                )
        
        sl = mplw.Slider(a_s, 'case index', 0, len(status) - 1, valinit=0, valfmt='%d')
        
        if dlines is not None:
            t_sl = mplw.Slider(a_t_s, 'time index', 0, len(time) - 1, valinit=0, valfmt='%d')
        else:
            t_sl = None
        
        def update_samp(idx):
            print("updating...")
            idx = int(idx)
            lines_D[update_samp.old_idx].set_alpha(off_alpha)
            lines_V[update_samp.old_idx].set_alpha(off_alpha)
            lines_D[update_samp.old_idx].set_color('k')
            lines_V[update_samp.old_idx].set_color('k')
            lines_D[update_samp.old_idx].set_linewidth(1)
            lines_V[update_samp.old_idx].set_linewidth(1)
            lines_D[update_samp.old_idx].set_zorder(2)
            lines_V[update_samp.old_idx].set_zorder(2)
            if dlines is not None:
                for i, lines in enumerate(lines_dlines + [lines_sig_sr, lines_sig_vuv] + lines_xtomo.values() + lines_dlines_norm):
                    lines[update_samp.old_idx].set_alpha(off_alpha)
                    lines[update_samp.old_idx].set_color('k')
                    lines[update_samp.old_idx].set_linewidth(1)
                    lines[update_samp.old_idx].set_zorder(2)
            
            update_samp.old_idx = idx
            lines_D[idx].set_alpha(1)
            lines_V[idx].set_alpha(1)
            lines_D[idx].set_color('b')
            lines_V[idx].set_color('b')
            lines_D[idx].set_linewidth(5)
            lines_V[idx].set_linewidth(5)
            lines_D[idx].set_zorder(len(lp) + 3)
            lines_V[idx].set_zorder(len(lp) + 3)
            if dlines is not None:
                for i, lines in enumerate(lines_dlines + [lines_sig_sr, lines_sig_vuv] + lines_xtomo.values() + lines_dlines_norm):
                    lines[idx].set_alpha(1)
                    lines[idx].set_color('b')
                    lines[idx].set_linewidth(5)
                    lines[idx].set_zorder(len(lp) + 3)
            
            if t_sl is not None:
                title.set_text("%s, lp=%.4g, t=%.3gs" % (OPT_STATUS[status[idx]], lp[idx], time[t_sl.val] - self.time_1))
            else:
                title.set_text("%s, lp=%.4g" % (OPT_STATUS[status[idx]], lp[idx]))
            
            f.canvas.draw()
            print('done!')
        
        update_samp.old_idx = 0
        
        def update_time(idx):
            print("updating...")
            idx = int(idx)
            for i, lines in enumerate(lines_dlines):
                for j, l in enumerate(lines):
                    l.set_ydata(dlines[j, idx, i, :])
            for i, lines in enumerate(lines_dlines_norm):
                for j, l in enumerate(lines):
                    l.set_ydata(dlines_norm[j, idx, i, :])
            for i, l in enumerate(lines_sig_sr):
                l.set_ydata(sbright[i, idx, :])
            for i, l in enumerate(lines_sig_vuv):
                l.set_ydata(vbright[i, idx, :])
            for k, lines_x in lines_xtomo.iteritems():
                for i, l in enumerate(lines_x):
                    l.set_ydata(xtomobright[k][i, idx, :])
            
            # Update the errorbar plots:
            t_idx_sr = profiletools.get_nearest_idx(time[idx] - self.time_1, self.run_data.hirex_time_combined)
            y = self.run_data.hirex_signal_norm_combined[t_idx_sr, :]
            yerr = self.run_data.hirex_uncertainty_norm_combined[t_idx_sr, :]
            line_data_sr.set_ydata(y)
            erry_top_data_sr.set_ydata(y + yerr)
            erry_bot_data_sr.set_ydata(y - yerr)
            new_segments_y = [
                scipy.array([[x, yt], [x, yb]]) for x, yt, yb in zip(line_data_sr.get_xdata(), y + yerr, y - yerr)
            ]
            barsy_data_sr.set_segments(new_segments_y)
            
            t_idx_vuv = profiletools.get_nearest_idx(time[idx] - self.time_1, self.run_data.vuv_times_combined[0, :])
            y = self.run_data.vuv_signals_norm_combined[:, t_idx_vuv]
            yerr = self.run_data.vuv_uncertainties_norm_combined[:, t_idx_vuv]
            line_data_vuv.set_ydata(y)
            erry_top_data_vuv.set_ydata(y + yerr)
            erry_bot_data_vuv.set_ydata(y - yerr)
            new_segments_y = [
                scipy.array([[x, yt], [x, yb]]) for x, yt, yb in zip(line_data_vuv.get_xdata(), y + yerr, y - yerr)
            ]
            barsy_data_vuv.set_segments(new_segments_y)
            
            for i, (k, s) in enumerate(self.run_data.xtomo_signal_norm_combined.iteritems()):
                t_idx_xtomo = profiletools.get_nearest_idx(time[idx] - self.time_1, self.run_data.xtomo_times_combined[k])
                y = s[:, t_idx_xtomo]
                yerr = 0.1 * s[:, t_idx_xtomo]
                line_data_xtomo[k].set_ydata(y)
                erry_top_data_xtomo[k].set_ydata(y + yerr)
                erry_bot_data_xtomo[k].set_ydata(y - yerr)
                new_segments_y = [
                    scipy.array([[x, yt], [x, yb]]) for x, yt, yb in zip(line_data_xtomo[k].get_xdata(), y + yerr, y - yerr)
                ]
                barsy_data_xtomo[k].set_segments(new_segments_y)
            
            for i, (l, a) in enumerate(zip(lines_dlines_true, a_lines)):
                l.set_ydata(dlines_true[idx, i, :])
                a.relim()
                a.autoscale(axis='y')
            for i, (l, a) in enumerate(zip(lines_dlines_norm_true, a_lines_norm)):
                l.set_ydata(dlines_norm_true[idx, i, :])
                a.relim()
                a.autoscale(axis='y')
            
            line_sig_sr_true.set_ydata(sbright_true[idx, :])
            a_sr.relim()
            a_sr.autoscale(axis='y')
            
            line_sig_vuv_true.set_ydata(vbright_true[idx, :])
            a_vuv.relim()
            a_vuv.autoscale(axis='y')
            
            for (k, l), a in zip(lines_xtomo_true.iteritems(), a_xtomo):
                l.set_ydata(xtomobright_true[k][idx, :])
                a.relim()
                a.autoscale(axis='y')
            
            title.set_text("%s, lp=%.4g, t=%.3gs" % (OPT_STATUS[status[sl.val]], lp[sl.val], time[idx] - self.time_1))
            f.canvas.draw()
            print('done!')
        
        sl.on_changed(update_samp)
        update_samp(0)
        
        if dlines is not None:
            t_sl.on_changed(update_time)
            update_time(0)
        
        f.canvas.mpl_connect('key_press_event', lambda evt: arrow_respond(t_sl, sl, evt))
    
    def sample_posterior(
            self,
            nsamp,
            burn=None,
            num_proc=None,
            nwalkers=None,
            ntemps=20,
            a=2.0,
            make_backup=True,
            pool=None,
            samp_type='Ensemble',
            theta0=None,
            ball_samples=0,
            ball_std=0.01,
            adapt=False,
            **sampler_kwargs
        ):
        """Initialize and run the MCMC sampler.
        
        Parameters
        ----------
        nsamp : int
            The number of samples to draw from each walker.
        burn : int, optional
            The number of samples to drop from the start. If absent, `nsamp` // 2
            samples are burned from the start of each walker.
        num_proc : int, optional
            The number of processors to use. If absent, the number of cores on
            the machine divided by two is used.
        nwalkers : int, optional
            The number of walkers to use. If absent, the `num_proc` times the
            number of dimensions of the parameter space times two is used.
        ntemps : int, optional
            The number of temperatures to use with a parallel-tempered sampler.
        Tmax : float, optional
            The maximum temperature to use with a parallel-tempered sampler. If
            using adaptive sampling, `scipy.inf` is a good choice.
        make_backup : bool, optional
            If True, the sampler (less its pool) will be written to sampler.pkl
            when sampling is complete. Default is True (backup sampler).
        pool : :py:class:`emcee.InterruptiblePool` instance
            The pool to use for multiprocessing. If present overrides num_proc.
        samp_type : {'Ensemble', 'PT'}, optional
            The type of sampler to construct. Options are the affine-invariant
            ensemble sampler (default) and the parallel-tempered sampler.
        ball_samples : int, optional
            The number of samples to take in a ball around each entry in
            `theta0`. Default is 0 (just use the values in `theta0` directly).
        ball_std : float, optional
            The standard deviation to use when constructing the ball of samples
            to start from. This is given as a fraction of the value. Default is
            0.01 (i.e., 1%%).
        adapt : bool, optional
            Whether or not to use an adaptive temperature ladder with the PT
            sampler. You must have put the appropriately-modified form of emcee
            on your sys.path for this to work, and have selected
            `samp_type` = 'PT'.
        **sampler_kwargs : optional keyword args
            Optional arguments passed to construct the sampler. The most useful
            one is `a`, the width of the proposal distribution. You can also use
            this to adjust `adaptation_lag` (the timescale for adaptation of the
            temperature ladder to slow down on) and `adaptation_time` (the
            timescale of the temperature adaptation dynamics themselves).
        """
        if self.method == 'GP':
            ndim = (
                self.num_eig_D +
                self.num_eig_V +
                self.k_D.num_free_params +
                self.mu_D.num_free_params +
                self.k_V.num_free_params +
                ((7 if self.clusters else 5) if self.source_file is None else 3)
            )
        elif self.method == 'spline':
            ndim = (
                self.num_eig_D +
                self.num_eig_V +
                (self.num_eig_D - self.spline_k_D if self.free_knots else 0) +
                (self.num_eig_V - self.spline_k_V if self.free_knots else 0) +
                ((7 if self.clusters else 5) if self.source_file is None else 3)
            )
        elif self.method == 'linterp':
            ndim = (
                self.num_eig_D +
                self.num_eig_V +
                (self.num_eig_D - 1 if self.free_knots else 0) +
                (self.num_eig_V - 1 if self.free_knots else 0) +
                ((7 if self.clusters else 5) if self.source_file is None else 3)
            )
        if self.use_scaling:
            ndim += 1 + self.run_data.vuv_signals_norm_combined.shape[0]
        if burn is None:
            burn = nsamp // 2
        if num_proc is None:
            if pool is not None:
                num_proc = pool._processes
            else:
                num_proc = multiprocessing.cpu_count()
        if nwalkers is None:
            nwalkers = num_proc * ndim * 2
        
        if num_proc > 1 and pool is None:
            pool = make_pool(num_proc=num_proc)
        
        if samp_type == 'Ensemble':
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                _ComputeLnProbWrapper(self),
                pool=pool,
                **sampler_kwargs
            )
        elif samp_type == 'PT':
            # TODO: This needs to be cleaned up -- the adaptive sampler version
            # has a different fingerprint.
            if adapt:
                sampler = emcee.PTSampler(
                    nwalkers,
                    ndim,
                    _ComputeLnProbWrapper(self),
                    self.get_prior(),
                    ntemps=ntemps,
                    pool=pool,
                    loglkwargs={'no_prior': True},
                    **sampler_kwargs
                )
            else:
                sampler = emcee.PTSampler(
                    ntemps,
                    nwalkers,
                    ndim,
                    _ComputeLnProbWrapper(self),
                    self.get_prior(),
                    pool=pool,
                    loglkwargs={'no_prior': True},
                    **sampler_kwargs
                )
        else:
            raise ValueError("Unknown sampler type: %s" % (samp_type,))
        
        return self.add_samples(
            sampler,
            nsamp,
            burn=burn,
            make_backup=make_backup,
            first_run=True,
            theta0=theta0,
            adapt=adapt,
            ball_samples=ball_samples,
            ball_std=ball_std
        )
    
    def add_samples(
            self,
            sampler,
            nsamp,
            burn=0,
            make_backup=True,
            first_run=False,
            resample_infs=True,
            ll_thresh=None,
            theta0=None,
            ball_samples=None,
            ball_std=None,
            adapt=False
        ):
        """Add samples to the given sampler.
        
        Parameters
        ----------
        sampler : :py:class:`emcee.EnsembleSampler` instance
            The sampler to add to.
        nsamp : int
            The number of samples to add.
        burn : int, optional
            The number of samples to burn when plotting. Default is 0.
        make_backup : bool, optional
            If True, the sampler will be backed up to
            ../sampler_<SHOT>_<VERSION>.pkl when done. Default is True.
        first_run : bool, optional
            If True, the initial state is taken to be a draw from the prior
            (i.e., for the initial run of the sampler). Otherwise, the initial
            state is taken to be the current state. Default is False (use
            current state of sampler).
        resample_infs : bool, optional
            If True, any chain whose log-probability is presently infinite will
            be replaced with a draw from the prior. Only has an effect when
            `first_run` is False. Default is True.
        ll_thresh : float, optional
            The threshold of log-probability, below which the chain will be
            re-drawn from the prior. Default is to not redraw any chains with
            finite log-probabilities.
        theta0 : array of float, optional
            The starting points for each chain. If omitted and `first_run` is
            True then a draw from the prior will be used. If omitted and
            `first_run` is False then the last state of the chain will be used.
        ball_samples : int, optional
            The number of samples to take in a ball around each entry in
            `theta0`. Default is 0 (just use the values in `theta0` directly).
        ball_std : float, optional
            The standard deviation to use when constructing the ball of samples
            to start from. This is given as a fraction of the value. Default is
            0.01 (i.e., 1%%).
        adapt : bool, optional
            Whether or not to use an adaptive temperature ladder with the PT
            sampler. You must have put the appropriately-modified form of emcee
            on your sys.path for this to work, and have passed a
            :py:class:`emcee.PTSampler` instance for `sampler`.
        """
        if theta0 is None:
            if first_run or resample_infs:
                prior = self.get_prior()
                if isinstance(sampler, emcee.EnsembleSampler):
                    draw = prior.random_draw(size=sampler.chain.shape[0]).T
                elif isinstance(sampler, emcee.PTSampler):
                    draw = prior.random_draw(size=(sampler.nwalkers, sampler.ntemps)).T
                else:
                    raise ValueError("Unknown sampler class: %s" % (type(sampler),))
                if first_run:
                    theta0 = draw
                else:
                    if isinstance(sampler, emcee.EnsembleSampler):
                        theta0 = sampler.chain[:, -1, :]
                        bad = (
                            scipy.isinf(sampler.lnprobability[:, -1])# |
                            #(sampler.lnprobability[:, -1] <= -5.0e4)
                        )
                        if ll_thresh is not None:
                            bad = bad | (sampler.lnprobability[:, -1] <= ll_thresh)
                    elif isinstance(sampler, emcee.PTSampler):
                        theta0 = sampler.chain[:, :, -1, :]
                        bad = (
                            scipy.isinf(sampler.lnprobability[:, :, -1]) |
                            scipy.isnan(sampler.lnprobability[:, :, -1])
                            #(sampler.lnprobability[:, :, -1] <= -5.0e4)
                        )
                        if ll_thresh is not None:
                            bad = bad | (sampler.lnprobability[:, :, -1] <= ll_thresh)
                    else:
                        raise ValueError("Unknown sampler class: %s" % (type(sampler),))
                
                    theta0[bad, :] = draw[bad, :]
            else:
                if isinstance(sampler, emcee.EnsembleSampler):
                    theta0 = sampler.chain[:, -1, :]
                elif isinstance(sampler, emcee.PTSampler):
                    theta0 = sampler.chain[:, :, -1, :]
                else:
                    raise ValueError("Unknown sampler class: %s" % (type(sampler),))
        else:
            if ball_samples > 0:
                theta0 = scipy.asarray(theta0, dtype=float)
                if theta0.ndim == 1:
                    theta0 = emcee.utils.sample_ball(
                        theta0,
                        ball_std * theta0,
                        size=ball_samples
                    )
                else:
                    theta0 = [
                        emcee.utils.sample_ball(
                            x,
                            ball_std * x,
                            size=ball_samples
                        )
                        for x in theta0
                    ]
                    theta0 = scipy.vstack(theta0)
                    print(theta0.shape)
                # Check against bounds:
                bounds = scipy.asarray(self.get_prior().bounds[:])
                for i in xrange(0, len(bounds)):
                    theta0[theta0[:, i] < bounds[i, 0], i] = bounds[i, 0]
                    theta0[theta0[:, i] > bounds[i, 1], i] = bounds[i, 1]
        
        print("Starting MCMC sampler...this will take a while.")
        try:
            subprocess.call('fortune -a | cowsay -f vader-koala', shell=True)
        except:
            pass
        
        t_start = time_.time()
        if isinstance(sampler, emcee.EnsembleSampler):
            sampler.run_mcmc(theta0, nsamp)
        elif isinstance(sampler, emcee.PTSampler):
            sampler.run_mcmc(theta0, nsamp, adapt=adapt)
        t_elapsed = time_.time() - t_start
        print("MCMC sampler done, elapsed time is %.2fs." % (t_elapsed,))
        
        labels = self.get_labels()
        
        # gptools.plot_sampler(
        #     sampler,
        #     burn=burn,
        #     labels=labels
        # )
        # impath = '../sampler_%d_%d.pdf' % (self.shot, self.version,)
        # plt.gcf().savefig(impath)
        
        if make_backup:
            try:
                # pools aren't pickleable, so we need to ditch it:
                pool = sampler.pool
                sampler.pool = None
                # Put the file one level up so we don't copy it with our directory each time we open a pool!
                # Perform an atomic save so we don't nuke it if there is a failure.
                with open('../tmp_%d_%d.pkl' % (self.shot, self.version), 'wb') as f:
                    pkl.dump(sampler, f, protocol=pkl.HIGHEST_PROTOCOL)
                os.rename(
                    '../tmp_%d_%d.pkl' % (self.shot, self.version),
                    '../sampler_%d_%d.pkl' % (self.shot, self.version,)
                )
            except SystemError:
                # Failback on the basic pickle if it fails:
                warnings.warn("cPickle failed, trying pickle!", RuntimeWarning)
                import pickle as pkl2
                with open('../tmp_%d_%d.pkl' % (self.shot, self.version,), 'wb') as f:
                    pkl2.dump(sampler, f)
                os.rename(
                    '../tmp_%d_%d.pkl' % (self.shot, self.version),
                    '../sampler_%d_%d.pkl' % (self.shot, self.version,)
                )
            finally:
                sampler.pool = pool
        
        send_email("MCMC update", "MCMC sampler done.", []) #, [impath])
        
        return sampler
    
    def restore_sampler(self, pool=None, spath=None):
        """Restore the most recent sampler, optionally setting its pool.
        """
        # TODO: MAKE THIS PULL IN INFO ON OTHER SETTINGS!
        if spath is None:
            spath = '../sampler_%d_%d.pkl' % (self.shot, self.version)
        with open(spath, 'rb') as f:
            s = pkl.load(f)
        s.pool = pool
        
        return s
    
    def combine_samplers(self, v_start=0, lp=None, ll=None, chain=None, beta=None, tswap_acceptance_fraction=None, make_plots=True):
        """Stitch together multiple samplers from sequentially-numbered files.
        
        Parameters
        ----------
        v_start : int, optional
            The sampler index to start reading at. Use this to avoid re-reading
            old samplers. Default is 0 (read all samplers).
        lp : array of float, optional
            The log-posterior histories which have been previously read. If
            present, the new data will be concatenated on.
        ll : array of float, optional
            The log-likelihood histories which have been previously read. If
            present, the new data will be concatenated on.
        chain : array of float, optional
            The parameter histories which have been previously read. If present,
            the new data will be concatenated on.
        beta : array of float, optional
            The inverse temperature histories which have been previously read.
            If present, the new data will be concatenated on.
        tswap_acceptance_fraction : array of float, optional
            The accepted temperature swap fractions for each temperature which
            have been previously read. If present, new data will be combined in.
        make_plots : bool, optional
            If True, plots of the log-posterior and beta histories will be
            produced. Default is True (make plots).
        """
        v = glob.glob('../sampler_%d_%d_*.pkl' % (self.shot, self.version))
        v = [
            int(
                re.split(
                    r'^\.\./sampler_%d_%d_([0-9]+)\.pkl$' % (self.shot, self.version),
                    s
                )[1]
            ) for s in v
        ]
        v.sort()
        v = scipy.asarray(v)
        # Remove previously-read samplers:
        v = v[v >= v_start]
        # Get the shapes by simply handling the first fencepost if previous
        # values were not provided:
        if (lp is None) or (ll is None) or (chain is None) or (beta is None) or (tswap_acceptance_fraction is None):
            vv = v[0]
            v = v[1:]
            print(vv)
            s = self.restore_sampler(
                spath='../sampler_%d_%d_%d.pkl' % (self.shot, self.version, vv)
            )
            lp = s.lnprobability
            ll = s.lnlikelihood
            chain = s.chain
            beta = s.beta_history
            tswap_acceptance_fraction = s.tswap_acceptance_fraction
        for vv in v:
            print(vv)
            s = self.restore_sampler(
                spath='../sampler_%d_%d_%d.pkl' % (self.shot, self.version, vv)
            )
            lp = scipy.concatenate((lp, s.lnprobability), axis=2)
            ll = scipy.concatenate((lp, s.lnlikelihood), axis=2)
            tswap_acceptance_fraction = (
                chain.shape[2] * tswap_acceptance_fraction +
                s.chain.shape[2] * s.tswap_acceptance_fraction
            ) / (chain.shape[2] + s.chain.shape[2])
            chain = scipy.concatenate((chain, s.chain), axis=2)
            beta = scipy.concatenate((beta, s.beta_history), axis=1)
        
        if make_plots:
            self.plot_lp_chains(lp)
            
            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            a.semilogy(beta.T)
            a.set_xlabel('step')
            a.set_ylabel(r'$\beta$')
        
        return (lp, ll, chain, beta, tswap_acceptance_fraction)
    
    def plot_prior_samples(self, nsamp):
        """Plot samples from the prior distribution.
        
        Parameters
        ----------
        nsamp : int
            The number of samples to plot.
        """
        prior = self.get_prior()
        draw = prior.random_draw(size=nsamp).T
        
        f = plt.figure()
        aD = f.add_subplot(2, 1, 1)
        aV = f.add_subplot(2, 1, 2)
        
        for d in draw:
            D, V = self.eval_DV(d)
            aD.plot(self.roa_grid_DV, D, alpha=0.1)
            aV.plot(self.roa_grid_DV, V, alpha=0.1)
        
        aD.set_xlabel('$r/a$')
        aV.set_xlabel('$r/a$')
        aD.set_ylabel('$D$ [m$^2$/s]')
        aV.set_ylabel('$V$ [m/s]')
        
        f.canvas.draw()
    
    def get_labels(self):
        """Get the labels for each of the variables included in the sampler.
        """
        labels = (
            ['$u_{D,%d}$' % (n,) for n in xrange(0, self.num_eig_D)] +
            ['$u_{V,%d}$' % (n,) for n in xrange(0, self.num_eig_V)]
        )
        if self.method == 'GP':
            labels += (
                ['$' + n + ', D$' for n in self.k_D.param_names] +
                ['$' + n + ', D$' for n in self.mu_D.param_names] +
                ['$' + n + ', V$' for n in self.k_V.param_names]
            )
        elif self.free_knots:
            if self.method == 'spline':
                labels += ['$x_{D,%d}$' % (n + 1,) for n in xrange(0, self.num_eig_D - self.spline_k_D)]
                labels += ['$x_{V,%d}$' % (n + 1,) for n in xrange(0, self.num_eig_V - self.spline_k_V)]
            elif self.method == 'linterp':
                labels += ['$x_{V,%d}$' % (n + 1,) for n in xrange(0, self.num_eig_D - 1)]
                labels += ['$x_{D,%d}$' % (n + 1,) for n in xrange(0, self.num_eig_V - 1)]
        if self.use_scaling:
            labels += [r'$s$ H']
            for k in xrange(0, self.run_data.vuv_signals_norm_combined.shape[0]):
                labels += [r'$s$ V%d' % (k + 1,)]
            for k in self.run_data.xtomo_sig.keys():
                if self.run_data.xtomo_sig[k] is not None:
                    labels += [r'$s$ XTOMO %d' % (k,)]
        labels += [r'$\Delta t$ H', r'$\Delta t$ V', r'$\Delta t$ XTOMO']
        if self.source_file is None:
            labels += ['$t_{rise}$', '$n_{rise}$', '$t_{fall}$', '$n_{fall}']
            if self.clusters:
                labels += ['$t_{cluster}$', '$h_{cluster}$']
        return labels
    
    def process_sampler(self, sampler, burn=0, thin=1):
        """Processes the sampler.
        
        Performs the following tasks:
        
        * Marginalizes the D, V profiles and brightness histories.
        * Makes interactive plots to browse the state at each sample on each walker.
        
        Parameters
        ----------
        sampler : :py:class:`emcee.EnsembleSampler` instance
            The sampler to process the data from.
        burn : int, optional
            The number of samples to burn from the front of each walker. Default
            is zero.
        thin : int, optional
            The amount by which to thin the samples. Default is 1.
        """
        
        self.plot_marginalized_brightness(sampler, burn=burn, thin=thin)
        self.plot_marginalized_DV(sampler, burn=burn, thin=thin)
        self.explore_chains(samper)
    
    def plot_lp_chains(self, sampler, temp_idx=0):
        """Plot the log-posterior trajectories of the chains in the given sampler.
        """
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        if isinstance(sampler, emcee.EnsembleSampler):
            a.semilogy(-sampler.lnprobability.T, alpha=0.1)
        elif isinstance(sampler, emcee.PTSampler):
            a.semilogy(-sampler.lnprobability[temp_idx].T, alpha=0.1)
        elif isinstance(sampler, scipy.ndarray):
            if sampler.ndim == 3:
                a.semilogy(-sampler[temp_idx].T, alpha=0.1)
            else:
                a.semilogy(-sampler.T, alpha=0.1)
        else:
            raise ValueError("Unknown sampler class: %s" % (type(sampler),))
        a.set_xlabel('step')
        a.set_ylabel('-log-posterior')
    
    def plot_marginalized_DV(self, sampler, burn=0, thin=1, chain_mask=None, lp=None, pool=None):
        """Computes and plots the marginal D, V profiles.
        
        Parameters
        ----------
        sampler : :py:class:`emcee.EnsembleSampler` instance
            The sampler to process the data from.
        burn : int, optional
            The number of samples to burn from the front of each walker. Default
            is zero.
        thin : int, optional
            The amount by which to thin the samples. Default is 1.
        chain_mask : mask array
            The chains to keep when computing the marginalized D, V profiles.
            Default is to use all chains.
        lp : array, optional
            The log-probability. Only to be passed if `sampler` is an array.
        pool : object with `map` method, optional
            Multiprocessing pool to use. If None, `sampler.pool` will be used.
        """
        if pool is None:
            pool = sampler.pool
        if chain_mask is None:
            if isinstance(sampler, emcee.EnsembleSampler):
                chain_mask = scipy.ones(sampler.chain.shape[0], dtype=bool)
            elif isinstance(sampler, emcee.PTSampler):
                chain_mask = scipy.ones(sampler.chain.shape[1], dtype=bool)
            elif isinstance(sampler, scipy.ndarray):
                if sampler.ndim == 4:
                    chain_mask = scipy.ones(sampler.shape[1], dtype=bool)
                else:
                    chain_mask = scipy.ones(sampler.shape[0], dtype=bool)
            else:
                raise ValueError("Unknown sampler class: %s" % (type(sampler),))
        
        if isinstance(sampler, emcee.EnsembleSampler):
            flat_trace = sampler.chain[chain_mask, burn::thin, :]
        elif isinstance(sampler, emcee.PTSampler):
            flat_trace = sampler.chain[0, chain_mask, burn::thin, :]
        elif isinstance(sampler, scipy.ndarray):
            if sampler.ndim == 4:
                flat_trace = sampler[0, chain_mask, burn::thin, :]
            else:
                flat_trace = sampler[chain_mask, burn::thin, :]
        else:
            raise ValueError("Unknown sampler class: %s" % (type(sampler),))
        flat_trace = flat_trace.reshape((-1, flat_trace.shape[2]))
        
        # Throw out bad samples:
        if isinstance(sampler, emcee.EnsembleSampler):
            flat_lp = sampler.lnprobability[chain_mask, burn::thin].ravel()
        elif isinstance(sampler, emcee.PTSampler):
            flat_lp = sampler.lnprobability[0, chain_mask, burn::thin].ravel()
        elif isinstance(sampler, scipy.ndarray):
            if sampler.ndim == 4:
                flat_lp = lp[0, chain_mask, burn::thin].ravel()
            else:
                flat_lp = lp[chain_mask, burn::thin].ravel()
        else:
            raise ValueError("Unknown sampler class: %s" % (type(sampler),))
        good = ~(scipy.isinf(flat_lp) | scipy.isnan(flat_lp))
        flat_trace = flat_trace[good, :]
        
        DV_samp = scipy.asarray(
            pool.map(
                _ComputeProfileWrapper(self),
                flat_trace
            )
        )
        
        D_samp = DV_samp[:, 0, :]
        bad = scipy.isinf(D_samp).any(axis=1)
        print(str(bad.sum()) + " samples had inf in D.")
        D_mean = scipy.mean(D_samp[~bad], axis=0)
        D_std = scipy.std(D_samp[~bad], axis=0, ddof=1)
        
        V_samp = DV_samp[:, 1, :]
        V_mean = scipy.mean(V_samp[~bad], axis=0)
        V_std = scipy.std(V_samp[~bad], axis=0, ddof=1)
        
        f_DV = plt.figure()
        f_DV.suptitle('Marginalized Ca transport coefficient profiles')
        a_D = f_DV.add_subplot(2, 1, 1)
        a_D.plot(self.roa_grid_DV, D_mean, 'b')
        a_D.fill_between(
            self.roa_grid_DV,
            D_mean - D_std,
            D_mean + D_std,
            color='b',
            alpha=0.5
        )
        a_D.set_xlabel('$r/a$')
        a_D.set_ylabel('$D$ [m$^2$/s]')
        
        a_V = f_DV.add_subplot(2, 1, 2, sharex=a_D)
        a_V.plot(self.roa_grid_DV, V_mean, 'b')
        a_V.fill_between(
            self.roa_grid_DV,
            V_mean - V_std,
            V_mean + V_std,
            color='b',
            alpha=0.5
        )
        a_V.set_xlabel('$r/a$')
        a_V.set_ylabel('$V$ [m/s]')
        
        return (D_mean, D_std, V_mean, V_std)
    
    def plot_marginalized_brightness(self, sampler, burn=0, thin=1, chain_mask=None):
        """Averages the brightness histories over all samples/chains and makes a plot.
        
        Parameters
        ----------
        sampler : :py:class:`emcee.EnsembleSampler` instance
            The sampler to process the data from.
        burn : int, optional
            The number of samples to burn from the front of each walker. Default
            is zero.
        thin : int, optional
            The amount by which to thin the samples. Default is 1.
        """
        if not isinstance(sampler, emcee.EnsembleSampler):
            raise NotImplementedError(
                "plot_marginalized_brightness is only supported for EnsembleSamplers!"
            )
        if chain_mask is None:
            chain_mask = scipy.ones(sampler.chain.shape[0], dtype=bool)
        blobs = scipy.asarray(sampler.blobs[burn::thin], dtype=object)[:, chain_mask, :]
        chains = scipy.swapaxes(sampler.chain[chain_mask, burn::thin, :], 0, 1)
        
        # Flatten it out to compute the marginal stuff (we need to keep the
        # chain info for the slider plots, though):
        blobs_flat = scipy.reshape(blobs, (-1, blobs.shape[2]))
        chain_flat = scipy.reshape(chains, (-1, chains.shape[2]))
        
        ll_flat = scipy.asarray(blobs_flat[:, 0], dtype=float)
        good = ~(scipy.isinf(ll_flat) | scipy.isnan(ll_flat))
        ll_flat = ll_flat[good]
        
        sbright = blobs_flat[good, 1]
        vbright = blobs_flat[good, 2]
        time = blobs_flat[good, 3]
        t_s = chain_flat[good, -2]
        t_v = chain_flat[good, -1]
        
        # We need to interpolate sbright, vbright onto a uniform timebase:
        t = scipy.linspace(
            self.run_data.hirex_time_combined.min(),
            self.run_data.hirex_time_combined.max(),
            100
        )
        
        wrapper = _InterpBrightWrapper(t, sbright[0].shape[1], vbright[0].shape[1])
        out = sampler.pool.map(wrapper, zip(sbright, vbright, time, t_s, t_v))
        
        sbright_interp = scipy.asarray([o[0] for o in out], dtype=float)
        vbright_interp = scipy.asarray([o[1] for o in out], dtype=float)
        
        # Now we can compute the summary statistics:
        mean_sbright = scipy.mean(sbright_interp, axis=0)
        std_sbright = scipy.std(sbright_interp, axis=0, ddof=1)
        
        mean_vbright = scipy.mean(vbright_interp, axis=0)
        std_vbright = scipy.std(vbright_interp, axis=0, ddof=1)
        
        # And make a big plot:
        f_D, a_H, a_V = self.run_data.plot_data()
        for i, a in enumerate(a_H):
            a.plot(t, mean_sbright[:, i], 'g')
            a.fill_between(
                t,
                mean_sbright[:, i] - std_sbright[:, i],
                mean_sbright[:, i] + std_sbright[:, i],
                color='g',
                alpha=0.5
            )
        
        for i, a in enumerate(a_V):
            a.plot(t, mean_vbright[:, i], 'g')
            a.fill_between(
                t,
                mean_vbright[:, i] - std_vbright[:, i],
                mean_vbright[:, i] + std_vbright[:, i],
                color='g',
                alpha=0.5
            )
        f_D.canvas.draw()
    
    def compute_IC(self, sampler, burn, chain_mask=None, debug_plots=False, lp=None, ll=None):
        """Compute the DIC and AIC information criteria.
        
        Parameters
        ----------
        sampler : :py:class:`emcee.EnsembleSampler`
            The sampler to compute the criteria for.
        burn : int
            The number of samples to burn before computing the criteria.
        chain_mask : array, optional
            The chains to include in the computation.
        debug_plots : bool, optional
            If True, plots will be made of the conditions at the posterior mean
            and a histogram of the log-likelihood will be drawn.
        lp : array, optional
            The log-posterior. Only to be passed if `sampler` is an array.
        ll : array, optional
            The log-likelihood. Only to be passed if `sampler` is an array.
        """
        # Compute the DIC:
        if chain_mask is None:
            if isinstance(sampler, emcee.EnsembleSampler):
                chain_mask = scipy.ones(sampler.chain.shape[0], dtype=bool)
            elif isinstance(sampler, emcee.PTSampler):
                chain_mask = scipy.ones(sampler.chain.shape[1], dtype=bool)
            elif isinstance(sampler, scipy.ndarray):
                if sampler.ndim == 4:
                    chain_mask = scipy.ones(sampler.shape[1], dtype=bool)
                else:
                    chain_mask = scipy.ones(sampler.shape[0], dtype=bool)
            else:
                raise ValueError("Unknown sampler class: %s" % (type(sampler),))
        if isinstance(sampler, emcee.EnsembleSampler):
            flat_trace = sampler.chain[chain_mask, burn:, :]
        elif isinstance(sampler, emcee.PTSampler):
            flat_trace = sampler.chain[0, chain_mask, burn:, :]
        elif isinstance(sampler, scipy.ndarray):
            if sampler.ndim == 4:
                flat_trace = sampler[0, chain_mask, burn:, :]
            else:
                flat_trace = sampler[chain_mask, burn:, :]
        else:
            raise ValueError("Unknown sampler class: %s" % (type(sampler),))
        flat_trace = flat_trace.reshape((-1, flat_trace.shape[2]))
        
        theta_hat = flat_trace.mean(axis=0)
        
        lp_theta_hat, blob = self.compute_ln_prob(theta_hat, debug_plots=debug_plots, return_blob=True)
        ll_theta_hat = blob[0]
        
        if isinstance(sampler, emcee.EnsembleSampler):
            blobs = scipy.asarray(sampler.blobs, dtype=object)
            ll = scipy.asarray(blobs[burn:, chain_mask, 0], dtype=float)
        elif isinstance(sampler, emcee.PTSampler):
            ll = sampler.lnlikelihood[0, chain_mask, burn:]
        elif isinstance(sampler, scipy.ndarray):
            if sampler.ndim == 4:
                ll = ll[0, chain_mask, burn:]
            else:
                ll = ll[chain_mask, burn:]
        E_ll = ll.mean()
        
        if debug_plots:
            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            a.hist(ll.ravel(), 50)
            a.axvline(ll_theta_hat, label=r'$LL(\hat{\theta})$', color='r', lw=3)
            a.axvline(E_ll, label=r'$E[LL]$', color='g', lw=3)
            a.legend(loc='best')
            a.set_xlabel('LL')
        
        pD_1 = 2 * (ll_theta_hat - E_ll)
        pD_2 = 2 * ll.var(ddof=1)
        
        DIC_1 = -2 * ll_theta_hat + 2 * pD_1
        DIC_2 = -2 * ll_theta_hat + 2 * pD_2
        
        # Compute AIC:
        try:
            p = sampler.dim
        except AttributeError:
            p = sampler.shape[-1]
        ll_max = ll.max()
        AIC = 2 * p - 2 * ll_max
        
        # Compute WAIC:
        # TODO!
        
        # Compute log-evidence:
        try:
            lev, e_lev = sampler.thermodynamic_integration_log_evidence(fburnin=burn / sampler.chain.shape[2])
        except:
            lev = None
            e_lev = None
            warnings.warn("Thermodynamic integration failed!", RuntimeWarning)
        
        out = {
            'DIC_1': DIC_1,
            'DIC_2': DIC_2,
            'pD_1': pD_1,
            'pD_2': pD_2,
            'AIC': AIC,
            'p': p,
            'theta_hat': theta_hat,
            'log_evidence': lev,
            'err_log_evidence': e_lev
        }
        return out
    
    def explore_chains(self, sampler):
        """Interactively explore the chains in `sampler`.
        
        Creates three plots: the D, V profiles, the brightness histories and the
        chain histories. Interact with the chain histories using the arrow keys.
        """
        l = []
        f_b, a_H, a_VUV = self.run_data.plot_data()
        title_f_b = f_b.suptitle('')
        
        f_DV = plt.figure()
        title_f_DV = f_DV.suptitle('')
        a_D = f_DV.add_subplot(2, 1, 1)
        plt.setp(a_D.get_xticklabels(), visible=False)
        a_D.set_ylabel('$D$ [m$^2$/s]')
        a_V = f_DV.add_subplot(2, 1, 2, sharex=a_D)
        a_V.set_xlabel('$r/a$')
        a_V.set_ylabel('$V$ [m/s]')
        
        f_chains = plt.figure()
        title_f_chains = f_chains.suptitle('')
        gs = mplgs.GridSpec(3, sampler.chain.shape[2], height_ratios=[10, 1, 1])
        a_chains = []
        for k, label in enumerate(self.get_labels()):
            a_chains.append(
                f_chains.add_subplot(gs[0, k], sharex=a_chains[0] if k > 0 else None)
            )
            a_chains[-1].set_xlabel('step')
            a_chains[-1].set_ylabel(label)
            a_chains[-1].plot(sampler.chain[:, :, k].T, color='k', alpha=0.1)
        a_chain_slider = f_chains.add_subplot(gs[1, :])
        a_step_slider = f_chains.add_subplot(gs[2, :])
        
        def update(dum):
            """Update the chain and/or step index.
            """
            print("Updating...")
            remove_all(l)
            while l:
                l.pop()
            
            i_chain = int(chain_slider.val)
            i_step = int(step_slider.val)
            
            b = sampler.blobs[i_step][i_chain]
            
            print(b[-1])
            
            title_text = "walker %d, step %d, ll=%g, lp=%g" % (
                i_chain,
                i_step,
                b[0],
                sampler.lnprobability[i_chain, i_step]
            )
            title_f_b.set_text(title_text)
            title_f_DV.set_text(title_text)
            title_f_chains.set_text(title_text)
            
            # Plot the brightness histories:
            if b[1] is not None:
                for k, a in enumerate(a_H):
                    l.append(a.plot(b[3] + sampler.chain[i_chain, i_step, -2], b[1][:, k], 'g'))
                for k, a in enumerate(a_VUV):
                    l.append(a.plot(b[3] + sampler.chain[i_chain, i_step, -1], b[2][:, k], 'g'))
            
            D, V = self.eval_DV(sampler.chain[i_chain, i_step, :])
            
            l.append(a_D.plot(self.roa_grid_DV, D, 'b'))
            l.append(a_V.plot(self.roa_grid_DV, V, 'b'))
            
            a_D.relim()
            a_D.autoscale_view()
            a_V.relim()
            a_V.autoscale_view()
            
            for k in xrange(0, sampler.chain.shape[2]):
                l.append(
                    a_chains[k].plot(
                        sampler.chain[i_chain, :, k],
                        color='r',
                        linewidth=3
                    )
                )
                l.append(a_chains[k].axvline(i_step, color='r', linewidth=3))
            
            f_b.canvas.draw()
            f_DV.canvas.draw()
            f_chains.canvas.draw()
            print("Done.")
        
        def arrow_respond(up_slider, side_slider, event):
            """Event handler for arrow key events in plot windows.
            
            Pass the slider object to update as a masked argument using a lambda function::
            
                lambda evt: arrow_respond(my_up_slider, my_side_slider, evt)
            
            Parameters
            ----------
            up_slider : Slider instance associated with up/down keys for this handler.
            side_slider : Slider instance associated with left/right keys for this handler.
            event : Event to be handled.
            """
            if event.key == 'right':
                side_slider.set_val(min(side_slider.val + 1, side_slider.valmax))
            elif event.key == 'left':
                side_slider.set_val(max(side_slider.val - 1, side_slider.valmin))
            elif event.key == 'up':
                up_slider.set_val(min(up_slider.val + 1, up_slider.valmax))
            elif event.key == 'down':
                up_slider.set_val(max(up_slider.val - 1, up_slider.valmin))
        
        chain_slider = mplw.Slider(
            a_chain_slider,
            'walker index',
            0,
            sampler.chain.shape[0] - 1,
            valinit=0,
            valfmt='%d'
        )
        step_slider = mplw.Slider(
            a_step_slider,
            'step index',
            0,
            sampler.chain.shape[1] - 1,
            valinit=0,
            valfmt='%d'
        )
        chain_slider.on_changed(update)
        step_slider.on_changed(update)
        update(0)
        f_chains.canvas.mpl_connect(
            'key_press_event',
            lambda evt: arrow_respond(chain_slider, step_slider, evt)
        )
    
    def find_closest_representation(self, D_other, V_other, guess=None):
        """Find the closest representation of the given D, V profiles with the current basis functions.
        
        Parameters
        ----------
        D_other : array of float
            The values of D. Must be given on the same internal roa_grid_DV as
            the current run instance.
        V_other : array of float
            The values of V. Must be given on the same internal roa_grid_DV as
            the current run instance.
        guess : array of float, optional
            The initial guess to use for the parameters when running the
            optimizer. If not present, a random draw from the prior is used.
        """
        # TODO: This needs random starts!
        b = self.get_prior().bounds[:-4]
        bounds = [list(v) for v in b]
        for v in bounds:
            if scipy.isinf(v[0]):
                v[0] = None
            if scipy.isinf(v[1]):
                v[1] = None
        res = scipy.optimize.minimize(
            self.objective_func,
            self.get_prior().random_draw(size=1).ravel()[:-4] if guess is None else guess,
            args=(D_other, V_other),
            method='L-BFGS-B',
            # method='SLSQP',
            bounds=bounds
        )
        self.compute_ln_prob(scipy.concatenate((res.x, [1, 1, 0, 0])), debug_plots=True)
        
        D, V = self.eval_DV(scipy.concatenate((res.x, [1, 1, 0, 0])))
        f = plt.figure()
        aD = f.add_subplot(2, 1, 1)
        aV = f.add_subplot(2, 1, 2)
        aD.plot(self.roa_grid_DV, D)
        aD.plot(self.roa_grid_DV, D_other)
        aV.plot(self.roa_grid_DV, V)
        aV.plot(self.roa_grid_DV, V_other)
        
        return res
    
    def objective_func(self, params, D_other, V_other):
        """Objective function for the minimizer in :py:meth:`find_closest_representation`.
        """
        D, V = self.eval_DV(scipy.concatenate((params, [1, 1, 0, 0])))
        return scipy.sqrt((scipy.concatenate((D - D_other, V - V_other))**2).sum())
    
    @property
    def working_dir(self):
        """Returns the directory name for the given settings.
        """
        return 'strahl_%d_%d' % (self.shot, self.version)
    
    @property
    def ll_normalization(self):
        """Returns the normalization constant for the log-likelihood.
        """
        # TODO: This way of dropping LoWEUS breaks if a.) LoWEUS is not loaded
        # or b.) if LoWEUS is not the last line or c.) if there are multiple
        # LoWEUS lines. This should be corrected at some point...
        if self._ll_normalization is None:
            good_err = scipy.hstack(
                (
                    self.run_data.hirex_uncertainty_norm_combined[
                        ~self.run_data.hirex_flagged_combined
                    ],
                    (
                        self.run_data.vuv_uncertainties_norm_combined.ravel()
                        if True or self.include_loweus
                        else self.run_data.vuv_uncertainties_norm_combined[:-1, :].ravel()
                    )
                )
            )
            self._ll_normalization = (
                -scipy.log(good_err).sum() - 0.5 * len(good_err) * scipy.log(2 * scipy.pi)
            )
        return self._ll_normalization
    
    @property
    def ar_ll_normalization(self):
        """Returns the normalization constant for the log-likelihood of the Ar data.
        """
        if self._ar_ll_normalization is None:
            ar_mask = (self.run_data.ar_time >= self.time_1) & (self.run_data.ar_time <= self.time_2)
            ar_uncertainty = self.run_data.ar_uncertainty[ar_mask, :]
            ar_flagged = self.run_data.ar_flagged[ar_mask, :]
            good_err = ar_uncertainty.ravel()[~ar_flagged.ravel()]
            self._ar_ll_normalization = (
                -scipy.log(good_err).sum() - 0.5 * len(good_err) * scipy.log(2 * scipy.pi)
            )
        return self._ar_ll_normalization
    
    
    def setup_files(self):
        """Sets up a copy of the STRAHL directory with the relevant files.
        
        Must be run from the directory containing bayesimp.
        """
        
        print("Setting up bayesimp...")
        
        current_dir = os.getcwd()
        
        # Make a copy of the master STRAHL directory:
        print("Cloning master STRAHL directory...")
        new_dir = os.path.join(current_dir, self.working_dir)
        copy_tree(os.path.abspath('strahl'), new_dir)
        print("Created %s." % (new_dir,))
        
        # Switch to that directory to initialize the IDL side of things:
        print("Running setup_strahl_run...")
        os.chdir(new_dir)
        cmd = "idl <<EOF\n.compile setup_strahl_run.pro\nsetup_strahl_run, {shot}, {time_1}, {time_2}".format(
            shot=self.shot,
            time_1=self.time_1,
            time_2=self.time_2
        )
        try:
            cmd += ', tht={tht}'.format(tht=self.tht)
        except AttributeError:
            pass
        try:
            cmd += ', line={line}'.format(line=self.line)
        except AttributeError:
            pass
        cmd += '\nexit\nEOF'
        
        subprocess.call(cmd, shell=True)
        
        print("Setup of IDL files complete.")
    
    def write_control(self, filepath=None, time_2_override=None):
        """Writes the strahl.control file used to automate STRAHL.
        """
        if filepath is None:
            filepath = 'strahl.control'
        contents = (
            "run_{shot:d}.0\n"
            "  {time_2:.2f}\n"
            "E\n".format(
                shot=self.shot,
                time_2=self.time_2 if time_2_override is None else time_2_override
            )
        )
        
        with open(filepath, 'w') as f:
            f.write(contents)
        
        return contents
    
    def write_pp(self, sqrtpsinorm, ne, Te, t, filepath=None):
        """Write STRAHL plasma background (pp) file for the given ne, Te profiles.
        
        At present, this is a very simplistic script that has the functionality
        needed for :py:mod:`bayesimp` to run and not much else.
        
        Does not write the neutral density or ion temperature blocks. Assumes
        you have fit the WHOLE profile (i.e., uses the `interpa` option).
        
        Parameters
        ----------
        sqrtpsinorm : array of float, (`M`,)
            The square root of normalized poloidal flux grid the profiles are
            given on.
        ne : array of float, (`M`,) or (`N`, `M`)
            The electron density in units of 10^20 m^-3.
        Te : array of float, (`M`,) or (`N`, `M`)
            The electron temperature on units of keV.
        t : float or array of float (`N`,)
            The times the profiles are specified at. If using a single value,
            this should be equal to the end time of your simulation.
        filepath : str, optional
            The path to write the file to. By default, nete/pp<SHOT>.0 is used.
        """
        if filepath is None:
            filepath = 'nete/pp{shot:d}.0'.format(shot=self.shot)
        
        try:
            iter(t)
        except TypeError:
            t = scipy.atleast_1d(t)
            ne = scipy.atleast_2d(ne)
            Te = scipy.atleast_2d(Te)
        else:
            t = scipy.asarray(t, dtype=float)
            ne = scipy.asarray(ne, dtype=float)
            Te = scipy.asarray(Te, dtype=float)
        
        t_str = '    '.join(map(str, t))
        rho_str = '    '.join(map(str, sqrtpsinorm))
        ne_str = ''
        for row in ne:
            ne_max = row.max()
            ne_str += (
                str(ne_max * 1e14) + '    ' +
                '    '.join(map(str, row / ne_max)) + '\n'
            )
        Te_str = ''
        for row in Te:
            Te_max = row.max()
            Te_str += (
                str(Te_max * 1e3) + '    ' +
                '    '.join(map(str, row / Te_max)) + '\n'
            )
        
        contents = (
            "\n"
            "cv    time-vector\n"
            "      {num_time:d}\n"
            "      {time_points:s}\n"
            "\n"
            "cv    Ne-function\n"
            "      interpa\n"
            "\n"
            "\n"
            "cv    x-coordinate\n"
            "      'poloidal rho'\n"
            "\n"
            "\n"
            "cv    # of interpolation points\n"
            "      {num_rho:d}\n"
            "\n"
            "\n"
            "cv    x-grid for ne-interpolation\n"
            "      {rho_points:s}\n"
            "\n"
            "\n"
            "cv    DATA\n"
            "      {ne_points:s}"
            "\n"
            "\n"
            "cv    time-vector\n"
            "      {num_time:d}\n"
            "      {time_points:s}\n"
            "\n"
            "cv    Te-function\n"
            "      interpa\n"
            "\n"
            "\n"
            "cv    x-coordinate\n"
            "      'poloidal rho'\n"
            "\n"
            "\n"
            "cv    # of interpolation points\n"
            "      {num_rho:d}\n"
            "\n"
            "\n"
            "cv    x-grid for Te-interpolation\n"
            "      {rho_points:s}\n"
            "\n"
            "\n"
            "cv    DATA\n"
            "      {Te_points:s}"
            "\n"
            "\n"
            "cv    time-vector\n"
            "      0\n".format(
                num_time=len(t),
                time_points=t_str,
                num_rho=len(sqrtpsinorm),
                rho_points=rho_str,
                ne_points=ne_str,
                Te_points=Te_str
            )
        )
        
        with open(filepath, 'w') as f:
            f.write(contents)
        
        return contents
    
    def write_param(
            self,
            D_grid,
            V_grid,
            D,
            V,
            filepath=None,
            compute_NC=False,
            const_source=None,
            element='Ca',
            time_2_override=None
        ):
        """Write plasma param file for the given D, V profiles.
        
        At present this is a very stripped-down version that only implements the
        functionality needed for :py:mod:`bayesimp` to run.
        
        Note that this assumes you have written the source file to the right spot.
        
        Also note that there is a bug in STRAHL that causes it to crash if you
        use more than 100 points for the D, V profiles -- so don't do that!
        
        Parameters
        ----------
        D_grid : array of float
            The sqrtpsinorm points D is given on.
        V_grid : array of float
            The sqrtpsinorm points V is given on.
        D : array of float
            Values of D.
        V : array of float
            Values of V.
        filepath : str, optional
            The path to write the file to. If absent, param_files/run_SHOT.0 is
            used.
        compute_NC : bool, optional
            If True, neoclassical (NEOART) transport will be computed. Default
            is False.
        const_source : float, optional
            The constant source rate (particles/second) to use. Default is to
            use a time-varying source from a file.
        element : str, optional
            The element (symbol) to use. Default is 'Ca'.
        """
        if filepath is None:
            filepath = 'param_files/run_{shot:d}.0'.format(shot=self.shot)
        
        rho_str_D = '    '.join(map(str, D_grid))
        rho_str_V = '    '.join(map(str, V_grid))
        D_str = '    '.join(map(str, D))
        V_str = '    '.join(map(str, V))
        
        contents = (
            "                    E L E M E N T\n"
            "cv    element   atomic weight(amu)   energy of neutrals(eV)\n"
            "      '{elsym}'            {mass:.2f}                   1.00\n"
            "\n"
            "cv    main ion:  atomic weight(amu)    charge\n"
            "                       2.014               1\n"
            "\n"
            "                    G R I D - F I L E\n"
            "cv   shot         index\n"
            "     {shot:d}    0\n"
            "\n"
            "                    G R I D  P O I N T S  A N D  I T E R A T I O N\n"
            "cv    K    number of grid points  dr_center(cm)  dr_edge(cm)\n"
            "     6.0            100               0.3            0.1\n"
            "\n"
            "cv         max iterations at fixed time      stop iteration if change below (%)\n"
            "                      2000                                 0.001\n"
            "\n"
            "                    S T A R T  C O N D I T I O N S\n"
            "cv    start new=0/from old calc=1   take distr. from shot   at time\n"
            "                 0               0        0.000\n"
            "\n"
            "\n"
            "                    O U T P U T\n"
            "cv    save all cycles = 1, save final and start distribution = 0\n"
            "            1\n"
            "\n"
            "                    T I M E S T E P S\n"
            "cv    number of changes(start-time+....+stop-time)\n"
            "                  2\n"
            "\n"
            "cv    time   dt at start   increase of dt after cycle   steps per cycle\n"
            "    {time_1:.5f}     0.00010               1.001                      10\n"
            "    {time_2:.5f}     0.00010               1.001                      10\n"
            "\n"
            "                    S O U R C E\n"
            "cv    position(cm)    constant rate (1/s)    time dependent rate from file\n"
            "         90.5            {source:.5g}                         {from_file:d}\n"
            "\n"
            # MLR recommends putting -1 for each for stability:
            "cv    divertor puff   source width in(cm)    source width out(cm)\n"
            "           0                  -1                        -1\n"
            "\n"
            "                    E D G E ,  R E C Y C L I N G\n"
            "cv    decay length of impurity outside last grid point (cm)\n"
            "                              1.0 \n"
            "\n"
            # NOTE: NTH uses different values for these, but he also appears to
            # have used the exact values from the manual...
            "cv    Rec.:ON=1/OFF=0   wall-rec   Tau-div->SOL(ms)   Tau-pump(ms)\n"
            "             0             0            1.             1000.\n"
            "\n"
            "cv    SOL=width(cm)\n"
            "          1.0\n"
            "\n"
            "                    D E N S I T Y,  T E M P E R A T U R E,  A N D  N E U T R A L  H Y D R O G R E N  F O R  C X\n"
            "cv    take from file with:    shot      index\n"
            "                           {shot:d}     0\n"
            "\n"
            "                    N E O C L A S S I C A L  T R A N S P O R T\n"
            "                    method\n"
            "    0 = off,    >0 = % of Drift,   1 = approx\n"
            "cv  <0 = figure out, but dont use  2/3 = NEOART   neoclassics for rho_pol <\n"
            "                {NC:d}                      2                   0.99\n"
            "\n"
            "                    A N A M A L O U S  T R A N S P O R T\n"
            "cv   # of changes for transport\n"
            "      1\n"
            "\n"
            "cv   time-vector\n"
            "      0.00000\n"
            "\n"
            "cv   parallel loss times (ms)\n"
            "      2.50000\n"
            "cv   Diffusion [m^2/s]\n"
            "     'interp'\n"
            "\n"
            "cv   # of interpolation points\n"
            "        {num_rho_D:d}\n"
            "\n"
            "cv   rho_pol grid\n"
            "     {rho_points_D:s}\n"
            "\n"
            "cv   Diffusion Coefficient Grid\n"
            "     {D_points:s}\n"
            "\n"
            "cv   Drift function        only for drift\n"
            "     'interp'             'velocity'\n"
            "\n"
            "cv   # of interpolation points\n"
            "            {num_rho_V:d}\n"
            "\n"
            "cv   rho_pol grid\n"
            "     {rho_points_V:s}\n"
            "\n"
            "cv   Velocity Coefficient Grid\n"
            "\n"
            "     {V_points:s}\n"
            "\n"
            "cv   # of sawteeth       inversion radius (cm)\n"
            "          0                       1.00\n"
            "\n"
            "cv   times of sawteeth\n"
            "      0.00000\n".format(
                elsym=element,
                mass=periodictable.__dict__[element].mass,
                shot=self.shot,
                time_1=self.time_1,
                time_2=(self.time_2 if time_2_override is None else time_2_override),
                NC=-1 * int(compute_NC),
                num_rho_D=len(D_grid),
                num_rho_V=len(V_grid),
                rho_points_D=rho_str_D,
                rho_points_V=rho_str_V,
                D_points=D_str,
                V_points=V_str,
                source=1e17 if const_source is None else const_source,
                from_file=const_source is None
            )
        )
        
        with open(filepath, 'w') as f:
            f.write(contents)
        
        return contents
    
    def compute_view_data(self, debug_plots=False, write=True, contour_axis=None, **kwargs):
        """Compute the quadrature weights to line-integrate the emission profiles.
        
        Writes the output to view_data.pkl.
        """
        # First, do a dummy run of STRAHL to get the grid:
        # Just use random draws for the parameters, we just need it to run through:
        sqrtpsinormgrid = self.DV2cs_den(
            self.get_prior().random_draw(),
            compute_view_data=True,
            **kwargs
        )
        # print(sqrtpsinormgrid)
        # Temporary HACK:
        sqrtpsinormgrid[sqrtpsinormgrid < 0] = 0.0
        
        tokamak = TRIPPy.plasma.Tokamak(self.efit_tree)
        rays = [TRIPPy.beam.pos2Ray(p, tokamak) for p in self.run_data.hirex_pos]
        rays.append(TRIPPy.beam.pos2Ray(self.run_data.xeus_pos, tokamak))
        rays.append(TRIPPy.beam.pos2Ray(self.run_data.loweus_pos, tokamak))
        
        ar_rays = [TRIPPy.beam.pos2Ray(p, tokamak) for p in self.run_data.ar_pos]
        
        xtomo_1_beams = TRIPPy.XTOMO.XTOMO1beam(tokamak)
        xtomo_3_beams = TRIPPy.XTOMO.XTOMO3beam(tokamak)
        xtomo_5_beams = TRIPPy.XTOMO.XTOMO5beam(tokamak)
        
        # fluxFourierSens returns shape (n_time, n_chord, n_quad), we just have
        # one time element.
        self.weights = TRIPPy.invert.fluxFourierSens(
            rays,
            self.efit_tree.rz2psinorm,
            tokamak.center,
            (self.time_1 + self.time_2) / 2.0,
            sqrtpsinormgrid**2.0,
            ds=1e-5
        )[0]
        
        self.ar_weights = TRIPPy.invert.fluxFourierSens(
            ar_rays,
            self.efit_tree.rz2psinorm,
            tokamak.center,
            (self.time_1 + self.time_2) / 2.0,
            sqrtpsinormgrid**2.0,
            ds=1e-5
        )[0]
        
        self.xtomo_weights = {}
        self.xtomo_weights[1] = TRIPPy.invert.fluxFourierSens(
            xtomo_1_beams,
            self.efit_tree.rz2psinorm,
            tokamak.center,
            (self.time_1 + self.time_2) / 2.0,
            sqrtpsinormgrid**2.0,
            ds=1e-5
        )[0]
        
        self.xtomo_weights[3] = TRIPPy.invert.fluxFourierSens(
            xtomo_3_beams,
            self.efit_tree.rz2psinorm,
            tokamak.center,
            (self.time_1 + self.time_2) / 2.0,
            sqrtpsinormgrid**2.0,
            ds=1e-5
        )[0]
        
        self.xtomo_weights[5] = TRIPPy.invert.fluxFourierSens(
            xtomo_5_beams,
            self.efit_tree.rz2psinorm,
            tokamak.center,
            (self.time_1 + self.time_2) / 2.0,
            sqrtpsinormgrid**2.0,
            ds=1e-5
        )[0]
        
        if debug_plots:
            i_flux = profiletools.get_nearest_idx(
                (self.time_1 + self.time_2) / 2.0,
                self.efit_tree.getTimeBase()
            )
            
            color_vals = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            style_vals = ['-', '--', '-.', ':']
            ls_vals = []
            for s in style_vals:
                for c in color_vals:
                    ls_vals.append(c + s)
            
            ls_cycle = itertools.cycle(ls_vals)
            
            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            for w in self.weights:
                a.plot(sqrtpsinormgrid**2, w, ls_cycle.next())
            a.set_xlabel(r"$\psi_n$")
            a.set_ylabel("quadrature weights")
            a.set_title("Calcium")
            
            ls_cycle = itertools.cycle(ls_vals)
            vuv_cycle = itertools.cycle(['b', 'g'])
            
            from TRIPPy.plot.pyplot import plotTokamak, plotLine
            if contour_axis is None:
                f = plt.figure()
                a = f.add_subplot(1, 1, 1)
                # Only plot the tokamak if an axis was not provided:
                plotTokamak(tokamak)
            else:
                a = contour_axis
                plt.sca(a)
            # Plot VUV in different color:
            for r in rays[:-2]:
                plotLine(r, pargs='r')#ls_cycle.next())
            for r in rays[-2:]:
                plotLine(r, pargs=vuv_cycle.next(), lw=3)
            if contour_axis is None:
                a.contour(
                    self.efit_tree.getRGrid(),
                    self.efit_tree.getZGrid(),
                    self.efit_tree.getFluxGrid()[i_flux, :, :],
                    50
                )
            a.set_title("HiReX-SR, VUV")
            
            # Do it over again for Ar:
            ls_cycle = itertools.cycle(ls_vals)
            
            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            for w in self.ar_weights:
                a.plot(sqrtpsinormgrid**2, w, ls_cycle.next())
            a.set_xlabel(r"$\psi_n$")
            a.set_ylabel("quadrature weights")
            a.set_title("HiReXSR, argon")
            
            ls_cycle = itertools.cycle(ls_vals)
            
            if contour_axis is None:
                f = plt.figure()
                a = f.add_subplot(1, 1, 1)
                # Only plot the tokamak if an axis was not provided:
                plotTokamak(tokamak)
            else:
                a = contour_axis
                plt.sca(a)
            for r in ar_rays:
                plotLine(r, pargs=ls_cycle.next())
            if contour_axis is None:
                a.contour(
                    self.efit_tree.getRGrid(),
                    self.efit_tree.getZGrid(),
                    self.efit_tree.getFluxGrid()[i_flux, :, :],
                    50
                )
            a.set_title("Argon")
            
            # And for XTOMO 1:
            ls_cycle = itertools.cycle(ls_vals)
            
            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            for w in self.xtomo_weights[1]:
                a.plot(sqrtpsinormgrid**2, w, ls_cycle.next())
            a.set_xlabel(r"$\psi_n$")
            a.set_ylabel("quadrature weights")
            a.set_title("XTOMO 1")
            
            ls_cycle = itertools.cycle(ls_vals)
            
            if contour_axis is None:
                f = plt.figure()
                a = f.add_subplot(1, 1, 1)
                # Only plot the tokamak if an axis was not provided:
                plotTokamak(tokamak)
            else:
                a = contour_axis
                plt.sca(a)
            for r in xtomo_1_beams:
                plotLine(r, pargs='r')#ls_cycle.next())
            if contour_axis is None:
                a.contour(
                    self.efit_tree.getRGrid(),
                    self.efit_tree.getZGrid(),
                    self.efit_tree.getFluxGrid()[i_flux, :, :],
                    50
                )
            a.set_title("XTOMO 1")
            
            # And for XTOMO 3:
            ls_cycle = itertools.cycle(ls_vals)
            
            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            for w in self.xtomo_weights[3]:
                a.plot(sqrtpsinormgrid**2, w, ls_cycle.next())
            a.set_xlabel(r"$\psi_n$")
            a.set_ylabel("quadrature weights")
            a.set_title("XTOMO 3")
            
            ls_cycle = itertools.cycle(ls_vals)
            
            if contour_axis is None:
                f = plt.figure()
                a = f.add_subplot(1, 1, 1)
                # Only plot the tokamak if an axis was not provided:
                plotTokamak(tokamak)
            else:
                a = contour_axis
                plt.sca(a)
            for r in xtomo_3_beams:
                plotLine(r, pargs='r')#ls_cycle.next())
            if contour_axis is None:
                a.contour(
                    self.efit_tree.getRGrid(),
                    self.efit_tree.getZGrid(),
                    self.efit_tree.getFluxGrid()[i_flux, :, :],
                    50
                )
            a.set_title("XTOMO 3")
            
            # And for XTOMO 5:
            ls_cycle = itertools.cycle(ls_vals)
            
            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            for w in self.xtomo_weights[5]:
                a.plot(sqrtpsinormgrid**2, w, ls_cycle.next())
            a.set_xlabel(r"$\psi_n$")
            a.set_ylabel("quadrature weights")
            a.set_title("XTOMO 5")
            
            ls_cycle = itertools.cycle(ls_vals)
            
            if contour_axis is None:
                f = plt.figure()
                a = f.add_subplot(1, 1, 1)
                # Only plot the tokamak if an axis was not provided:
                plotTokamak(tokamak)
            else:
                a = contour_axis
                plt.sca(a)
            for r in xtomo_5_beams:
                plotLine(r, pargs='r')#ls_cycle.next())
            if contour_axis is None:
                a.contour(
                    self.efit_tree.getRGrid(),
                    self.efit_tree.getZGrid(),
                    self.efit_tree.getFluxGrid()[i_flux, :, :],
                    50
                )
            a.set_title("XTOMO 5")
        
        if write:
            with open('view_data.pkl', 'wb') as f:
                pkl.dump(self.weights, f)
            with open('ar_view_data.pkl', 'wb') as f:
                pkl.dump(self.ar_weights, f)
            with open('xtomo_view_data.pkl', 'wb') as f:
                pkl.dump(self.xtomo_weights, f)
        
        print("Done finding view data!")
    
    def load_PEC(self):
        """Load the photon emissivity coefficients from the ADF15 files.
        """
        self._PEC = {}
        atom_dir = 'atomdat/adf15/ca'
        for p in os.listdir(atom_dir):
            if p[0] == '.':
                continue
            res = re.split(".*ca([0-9]+)\.dat", os.path.basename(p))
            self._PEC[int(res[1])] = read_ADF15(os.path.join('atomdat/adf15/ca', p))
            #, debug_plots=[3.173, 19.775, 19.79]
    
    @property
    def PEC(self):
        """Reload the photon emissivity coefficients from the ADF15 files only as needed.
        """
        if self._PEC is None:
            self.load_PEC()
        return self._PEC
    
    def load_Ar_PEC(self, use_ADAS=False, debug_plots=False):
        """Load the photon emissivity coefficients from the ADF15 files.
        """
        self._Ar_PEC = {}
        if use_ADAS:
            self._Ar_PEC[16] = read_ADF15(
                'atomdat/adf15/ar/fpk#ar16.dat',
                debug_plots=[4.0,] if debug_plots else []
            )
        else:
            f = scipy.io.readsav('../ar_rates.sav')
            Te = scipy.asarray(f.Te, dtype=float) * 1e3
            exc = scipy.asarray(f.exc, dtype=float)
            rec = scipy.asarray(f.rec, dtype=float)
            ion = scipy.asarray(f.ion, dtype=float)
            
            # Excitation:
            self._Ar_PEC[16] = {
                4.0: [scipy.interpolate.InterpolatedUnivariateSpline(scipy.log10(Te), exc)]
            }
            
            # Recombination:
            self._Ar_PEC[17] = {
                4.0: [scipy.interpolate.InterpolatedUnivariateSpline(scipy.log10(Te), rec)]
            }
            
            # Ionization:
            self._Ar_PEC[15] = {
                4.0: [scipy.interpolate.InterpolatedUnivariateSpline(scipy.log10(Te), ion)]
            }
            
            if debug_plots:
                f = plt.figure()
                a = f.add_subplot(1, 1, 1)
                a.plot(Te, exc, '.', label='exc')
                a.plot(Te, rec, '.', label='rec')
                a.plot(Te, ion, '.', label='ion')
                a.set_xlabel('$T_e$ [eV]')
                a.set_ylabel('PEC')
                a.legend(loc='best')
    
    @property
    def Ar_PEC(self):
        """Reload the photon emissivity coefficients from the ADF15 files only as needed.
        """
        if self._Ar_PEC is None:
            self.load_Ar_PEC()
        return self._Ar_PEC
    
    def __getstate__(self):
        """Pitch the PEC's while loading because scipy.interpolate is stupid and not pickleable.
        """
        self._PEC = None
        self._Ar_PEC = None
        return self.__dict__
    
    def assemble_surrogate(self, stub, thresh=None):
        """Assemble a GP surrogate model from files with names stub_*.pkl.
        
        Returns a :py:class:`GaussianProcess` instance, trained with the data.
        """
        bounds = [(0, 1e7),] + [(0, r[1] - r[0]) for r in self.get_prior().bounds]
        k = gptools.SquaredExponentialKernel(
            num_dim=len(bounds) - 1,
            param_bounds=bounds,
            initial_params=[(b[0] + b[1]) / 4.0 for b in bounds]
        )
        gp = gptools.GaussianProcess(k)
        files = glob.glob(stub + '*.pkl')
        for fn in files:
            with open(fn, 'rb') as f:
                d = pkl.load(f)
            params = scipy.asarray(d['params'])
            lp = scipy.asarray(d['lp'])
            mask = (~scipy.isinf(lp)) & (~scipy.isnan(lp))
            if thresh is not None:
                mask = mask & (lp >= thresh)
            params = params[mask]
            lp = lp[mask]
            gp.add_data(params, lp)
        return gp
    
    def sample_surrogate(
        self,
        gp,
        nsamp,
        burn=None,
        num_proc=None,
        nwalkers=None,
        pool=None,
        **sampler_kwargs
    ):
        """Run MCMC on the GP surrogate.
        """
        # Make sure this has been run *before* sending out to nodes:
        # gp.compute_K_L_alpha_ll()
        
        ndim = gp.num_dim
        
        if burn is None:
            burn = nsamp // 2
        if num_proc is None:
            if pool is not None:
                num_proc = pool._processes
            else:
                num_proc = multiprocessing.cpu_count()
        if nwalkers is None:
            nwalkers = num_proc * ndim * 2
        
        if num_proc > 1 and pool is None:
            pool = InterruptiblePool(processes=num_proc)
        
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            _CallGPWrapper(gp),
            pool=pool,
            kwargs={'return_std': False},
            **sampler_kwargs
        )
        
        # Construct the initial points for the sampler:
        y_sort = gp.y.argsort()[::-1]
        X_sorted = gp.X[y_sort]
        theta0 = X_sorted[0:nwalkers]
        
        print("Starting MCMC sampler...this will take a while.")
        try:
            subprocess.call('fortune -a | cowsay -f vader-koala', shell=True)
        except:
            pass
        
        t_start = time_.time()
        sampler.run_mcmc(theta0, nsamp)
        t_elapsed = time_.time() - t_start
        print("MCMC sampler done, elapsed time is %.2fs." % (t_elapsed,))
        
        return sampler
    
    def analyze_envelopes(self, gp, max_include):
        """Make plots of the envelopes of parameters
        """
        sort_arg = gp.y.argsort()[::-1]
        y = gp.y[sort_arg[:max_include]]
        X = gp.X[sort_arg[:max_include]]
        
        X_mins = scipy.zeros_like(X)
        X_maxs = scipy.zeros_like(X)
        
        idxs = xrange(1, max_include)
        
        for k in idxs:
            X_mins[k] = X[:k].min(axis=0)
            X_maxs[k] = X[:k].max(axis=0)
        
        X_mins = X_mins[1:]
        X_maxs = X_maxs[1:]
        
        for k, l in zip(xrange(0, X.shape[1]), self.get_labels()):
            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            a.plot(idxs, X_mins[:, k], 'b')
            a.plot(idxs, X_maxs[:, k], 'b')
            a.set_title(l)
    
    def read_surrogate(self, stub, num_proc=None):
        """Attempt to construct an importance sampling estimate from the surrogate samples.
        
        DOESN'T WORK!
        """
        lp = None
        ll = None
        params = None
        files = glob.glob(stub + '*.pkl')
        for fn in files:
            with open(fn, 'rb') as f:
                d = pkl.load(f)
            if params is None:
                params = scipy.asarray(d['params'])
                lp = scipy.asarray(d['lp'])
                ll = scipy.asarray(d['ll'])
            else:
                params = scipy.vstack((params, d['params']))
                lp = scipy.concatenate((lp, d['lp']))
                ll = scipy.concatenate((ll, d['ll']))
        mask = (~scipy.isinf(lp)) & (~scipy.isnan(lp))
        params = params[mask]
        lp = lp[mask]
        ll = ll[mask]
        
        if num_proc is None:
            num_proc = multiprocessing.cpu_count()
        pool = InterruptiblePool(processes=num_proc)
        try:
            DV = pool.map(_ComputeProfileWrapper(self), params)
        finally:
            pool.close()
        DV = scipy.asarray(DV)
        D = DV[:, 0, :]
        V = DV[:, 1, :]
        
        lprior = lp - ll
        # Compute self-normalized importance sampling weights:
        lw = lp - lprior
        lw = lw - scipy.misc.logsumexp(lw)
        # Compute mean profile:
        # THIS DOESN'T WORK -- THE WEIGHTS ARE ALL CONCENTRATED ON ONE PROFILE.
        return lw
    
    def plot_surrogate_samples(self, gp):
        """Plots the samples from the surrogate.
        
        The alpha is chosen based on the log-posterior.
        """
        f = plt.figure()
        a_D = f.add_subplot(2, 1, 1)
        a_V = f.add_subplot(2, 1, 2, sharex=a_D)
        
        a_V.set_xlabel('$r/a$')
        a_D.set_ylabel('$D$ [m$^2$/s]')
        a_V.set_ylabel('$V$ [m/s]')
        
        max_lp = gp.y.max()
        min_lp = gp.y.min()
        
        for lp, p in zip(gp.y, gp.X):
            D, V = self.eval_DV(p)
            a_D.plot(self.roa_grid_DV, D, 'k', alpha=0.1 * (lp - min_lp) / (max_lp - min_lp))
            a_V.plot(self.roa_grid_DV, V, 'k', alpha=0.1 * (lp - min_lp) / (max_lp - min_lp))

class RunData(object):
    """Class to store the run data (both raw and edited).
    
    Assumes the current directory contains the IDL save file "run_data.sav".
    
    Performs the following operations:
    
    * Loads the HiReX-SR data from run_data.sav (previously produced with a
      call to :py:meth:`Run.setup_files`).
    * Launches a GUI to let the user flag bad points in the HiReX-SR data.
    * Launches a GUI to let the user select the diagnostic lines to use for the
      XEUS instrument. EACH LINE CAN ONLY HAVE ONE CHARGE STATE, OR STRAHL WILL
      BREAK!
    * Launches a GUI to do the same for the LoWEUS instrument.
    * Writes Ca.atomdat to reflect the desired spectral lines.
    * Launches gpfit for the user to fit the Te profile.
    * Launches gpfit for the user to fit the ne profile.
    * Bins the data into injections and normalizes according to the
      GP-interpolated maximum.
    
    Parameters
    ----------
    settings : :py:class:`Run` instance
        The imported settings to use.
    """
    def __init__(self, settings):
        self.settings = settings
        
        print("Reading run_data.sav...")
        # Load the data from run_data.sav:
        data = scipy.io.readsav('run_data.sav')
        self.xeus_pos = scipy.asarray(data.xeus_pos, dtype=float)
        self.loweus_pos = scipy.asarray(data.loweus_pos, dtype=float)
        
        # Line used for injections:
        self.hirex_signal = scipy.asarray(data.hirex_data.srsignal[0], dtype=float)
        self.hirex_uncertainty = scipy.asarray(data.hirex_data.srerr[0], dtype=float)
        self.hirex_pos = scipy.asarray(data.hirex_data.pos[0], dtype=float)
        self.hirex_time = scipy.asarray(data.hirex_data.t[0], dtype=float)
        self.hirex_tht = data.hirex_data.tht[0]
        self.hirex_line = data.hirex_data.line[0]
        
        # Argon line used for steady-state profile checking:
        self.ar_signal = scipy.asarray(data.ar_data.srsignal[0], dtype=float)
        self.ar_uncertainty = scipy.asarray(data.ar_data.srerr[0], dtype=float)
        self.ar_pos = scipy.asarray(data.ar_data.pos[0], dtype=float)
        self.ar_time = scipy.asarray(data.ar_data.t[0], dtype=float)
        self.ar_tht = data.ar_data.tht[0]
        self.ar_line = data.ar_data.line[0]
        
        self.shot = data.shot
        self.time_1 = data.time_1
        self.time_2 = data.time_2
        
        print("Processing HiReX-SR data...")
        self.hirex_flagged = (
            (self.hirex_uncertainty > HIREX_THRESH) |
            (self.hirex_uncertainty == 0.0)
        )
        self.ar_flagged = (
            (self.ar_uncertainty > HIREX_THRESH) |
            (self.ar_uncertainty == 0.0)
        )
        
        # Flag the bad HiReX-SR points:
        self.flag_hirex()
        self.flag_hirex(ar=True)
        
        if self.settings.debug_plots:
            self.plot_hirex()
            self.plot_hirex(ar=True)
        
        # Load the two VUV instruments:
        # We use vuv_lines to write the atomdat file, so we need it to have a
        # defined order:
        self.vuv_lines = collections.OrderedDict()
        self.vuv_signal = {}
        self.vuv_time = {}
        self.vuv_lam = {}
        self.vuv_uncertainty = {}
        
        # Load the XEUS data:
        self.load_vuv('XEUS')
        
        # Load the LoWEUS data:
        # self.load_vuv('LoWEUS')
        
        # Write the atomdat file:
        self.write_atomdat()
        
        # Load the source data:
        # TODO: Load source data (once we have a shot it doesn't suck for...)
        
        # Load the ne, Te data:
        print("Loading Te data...")
        self.load_Te()
        
        print("Loading ne data...")
        self.load_ne()
        
        print("Loading XTOMO data...")
        self.load_xtomo()
        self.flag_xtomo()
        
        print("Processing injections...")
        self.process_injections()
        
        self.plot_data()
        
        print("Fetching source data...")
        # TODO: This needs to be generalized!
        shutil.copyfile(
            self.settings.source_file,
            'nete/Caflx%d.dat' % (self.settings.shot,)
        )
        
        print("Loading and processing of data complete.")
    
    def patch_data(self):
        """Read the Ar data from an updated run_data.sav in to patch run_data.
        """
        print("Reading run_data.sav...")
        # Load the data from run_data.sav:
        data = scipy.io.readsav('run_data.sav')
        
        # Argon line used for steady-state profile checking:
        self.ar_signal = scipy.asarray(data.ar_data.srsignal[0], dtype=float)
        self.ar_uncertainty = scipy.asarray(data.ar_data.srerr[0], dtype=float)
        self.ar_pos = scipy.asarray(data.ar_data.pos[0], dtype=float)
        self.ar_time = scipy.asarray(data.ar_data.t[0], dtype=float)
        self.ar_tht = data.ar_data.tht[0]
        self.ar_line = data.ar_data.line[0]
    
    def flag_hirex(self, ar=False):
        """Interactively flag bad points on HiReX-SR.
        """
        root = HirexWindow(self, ar=ar)
        root.mainloop()
    
    def plot_hirex(self, z_max=None, norm=False, ar=False):
        """Make a 3d scatterplot of the HiReX-SR data.
        """
        f = plt.figure()
        a = f.add_subplot(1, 1, 1, projection='3d')
        if norm:
            t = self.hirex_time_combined
            keep = ~(self.hirex_flagged_combined.ravel())
            signal = self.hirex_signal_norm_combined
            uncertainty = self.hirex_uncertainty_norm_combined
        else:
            if ar:
                t = self.ar_time
                keep = ~(self.ar_flagged.ravel())
                signal = self.ar_signal
                uncertainty = self.ar_uncertainty
            else:
                t = self.hirex_time
                keep = ~(self.hirex_flagged.ravel())
                signal = self.hirex_signal
                uncertainty = self.hirex_uncertainty
        CHAN, T = scipy.meshgrid(range(0, signal.shape[1]), t)
        profiletools.errorbar3d(
            a,
            T.ravel()[keep],
            CHAN.ravel()[keep],
            signal.ravel()[keep],
            zerr=uncertainty.ravel()[keep]
        )
        a.set_zlim(0, z_max)
        a.set_xlabel('$t$ [s]')
        a.set_ylabel('channel')
        if norm:
            a.set_zlabel('normalized and combined HiReX-SR signal')
        else:
            a.set_zlabel('HiReX-SR signal [AU]')
    
    # def plot_xtomo(self, norm=False):
    #     """Make a 3d scatterplot of the XTOMO data.
    #
    #     DO NOT USE THIS FUNCTION -- IT WILL FREEZE YOUR PYTHON SESSION! TOO MANY
    #     DATA!
    #     """
    #     # XTOMO 1:
    #     f = plt.figure()
    #     a = f.add_subplot(1, 1, 1, projection='3d')
    #     if norm:
    #         t = self.xtomo_time_combined[1]
    #         signal = self.xtomo_signal_norm_combined[1]
    #     else:
    #         t = self.xtomo_t[1]
    #         signal = self.xtomo_sig[1]
    #     T, CHAN = scipy.meshgrid(t, range(0, signal.shape[0]))
    #     a.scatter(CHAN.ravel(), T.ravel(), signal.ravel())
    #     a.set_ylabel('$t$ [s]')
    #     a.set_xlabel('channel')
    #     if norm:
    #         a.set_zlabel('normalized and combined XTOMO 1 signal')
    #     else:
    #         a.set_xlabel('XTOMO 1 signal [AU]')
    
    def plot_xtomo(self, system, norm=False, f=None, boxcar=1, share_y=False, y_title=0.9, y_label='$b$ [AU]', max_ticks=None, rot_label=False, suptitle=None):
        """Make 2d scatterplots of the XTOMO data.
        
        Bad channels will be shown in red, good channels in blue. (But all will
        be plotted...)
        
        Parameters
        ----------
        system : int
            The XTOMO system to plot.
        norm : bool, optional
            If True, will plot the normalized data. Default is False (plot data
            from tree).
        f : Figure instance, optional
            If provided, the plots will be drawn on this figure instance.
        boxcar : int, optional
            The number of boxcar points to use. Default is 1 (no smoothing).
        """
        if norm:
            signal = self.xtomo_signal_norm_combined[system]
            t = self.xtomo_times_combined[system]
        else:
            signal = self.xtomo_sig[system]
            t = self.xtomo_t[system]
        
        if boxcar != 1:
            sort = t.argsort()
        else:
            sort = scipy.ones_like(t, dtype=bool)
        
        if f is None:
            f = plt.figure()
        ncol = 6.0
        nrow = scipy.ceil(1.0 * signal.shape[0] / ncol)
        gs = mplgs.GridSpec(int(nrow), int(ncol))
        a = []
        i_col = 0
        i_row = 0
        for k in xrange(0, signal.shape[0]):
            a.append(
                f.add_subplot(
                    gs[i_row, i_col],
                    sharex=a[0] if len(a) >= 1 else None,
                    sharey=a[0] if len(a) >= 1 and share_y else None
                )
            )
            if i_col > 0 and share_y:
                plt.setp(a[-1].get_yticklabels(), visible=False)
            else:
                a[-1].set_ylabel(y_label)
            if i_row == nrow - 1 or i_row == nrow - 2 and i_col >= signal.shape[0] - (nrow - 1) * ncol:
                a[-1].set_xlabel('$t$ [s]')
                if rot_label:
                    plt.setp(a[-1].xaxis.get_majorticklabels(), rotation=90)
                    for tick in a[-1].get_yaxis().get_major_ticks():
                        tick.label1 = tick._get_text1()
            else:
                plt.setp(a[-1].get_xticklabels(), visible=False)
            i_col += 1
            if i_col >= ncol:
                i_col = 0
                i_row += 1
            
            a[-1].set_title(
                'chord {k:d}'.format(k=k), y=y_title
            )
            if boxcar == 1:
                sig = signal[k, sort]
            else:
                sig = scipy.convolve(
                    signal[k, sort],
                    scipy.ones(int(boxcar)) / float(boxcar),
                    mode='same'
                )
            a[-1].plot(
                t[sort],
                sig,
                '.' if norm else '-',
                color='b' if self.xtomo_channel_mask[system][k] else 'r'
            )
        
        if max_ticks is not None:
            for aa in a:
                aa.xaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks - 1))
                aa.yaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks - 1))
        
        if suptitle is None:
            f.suptitle("XTOMO %d" % (system,))
        else:
            f.suptitle(suptitle)
        
        f.canvas.draw()
        
        return (f, a)
    
    @property
    def ar_normalization(self):
        # TODO: This should be cached!
        CHAN, T = scipy.meshgrid(range(0, self.ar_signal.shape[1]), self.ar_time)
        keep = ~((self.ar_flagged.ravel()) | (T.ravel() < self.time_1) | (T.ravel() > self.time_2))
        return max(
            [
                scipy.median(self.ar_signal.ravel()[keep][CHAN.ravel()[keep] == c])
                for c in range(0, self.ar_signal.shape[1])
            ]
        )
    
    def plot_ar(self, boxplot=False, norm=False):
        """Make a 2d plot of the HiReX-SR argon data.
        """
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        CHAN, T = scipy.meshgrid(range(0, self.ar_signal.shape[1]), self.ar_time)
        keep = ~((self.ar_flagged.ravel()) | (T.ravel() < self.time_1) | (T.ravel() > self.time_2))
        if norm:
            normalization = self.ar_normalization
        else:
            normalization = 1.0
        if boxplot:
            # TODO: This write this more elegantly!
            a.boxplot(
                [
                    self.ar_signal.ravel()[keep][CHAN.ravel()[keep] == c] / normalization
                    for c in range(0, self.ar_signal.shape[1])
                ]
            )
        else:
            a.errorbar(
                CHAN.ravel()[keep],
                self.ar_signal.ravel()[keep] / normalization,
                yerr=self.ar_uncertainty.ravel()[keep] / normalization,
                fmt='.'
            )
        a.set_xlabel('channel')
        a.set_ylabel('HiReX-SR signal [AU]')
        
        return (f, a)
    
    def load_vuv(self, system):
        """Load the data from a VUV instrument.
        
        Parameters
        ----------
        system : {'XEUS', 'LoWEUS'}
            The VUV instrument to load the data from.
        """
        print("Loading {system} data...".format(system=system))
        self.vuv_lines[system] = []
        t = MDSplus.Tree('spectroscopy', self.settings.shot)
        N = t.getNode(system + '.spec')
        self.vuv_signal[system] = scipy.asarray(N.data(), dtype=float)
        self.vuv_time[system] = scipy.asarray(N.dim_of(idx=1).data(), dtype=float)
        self.vuv_lam[system] = scipy.asarray(N.dim_of(idx=0).data(), dtype=float) / 10.0
        
        # Get the raw count data to compute the uncertainty:
        self.vuv_uncertainty[system] = (
            self.vuv_signal[system] /
            scipy.sqrt(t.getNode(system + '.raw:data').data())
        )
        
        print("Processing {system} data...".format(system=system))
        self.select_vuv(system)
    
    def select_vuv(self, system):
        """Select the lines to use from the given VUV spectrometer.
        """
        # TODO: Make a way to specify this manually in the settings file!
        root = VuvWindow(self, system)
        root.mainloop()
    
    def plot_vuv(self, system):
        """Make a contour plot of the VUV spectrum.
        """
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        LAM, T = scipy.meshgrid(self.vuv_lam[system], self.vuv_time[system])
        a.pcolormesh(
            LAM,
            T,
            self.vuv_signal[system],
            cmap='gray'
        )
        xlim = a.get_xlim()
        for x in CA_17_LINES:
            a.axvline(x, color='r')
        for x in CA_16_LINES:
            a.axvline(x, color='c')
        a.set_xlim(xlim)
        a.set_xlabel(r'$\lambda$ [nm]')
        a.set_ylabel('$t$ [s]')
    
    def plot_vuv_normed(self):
        """Plot each of the normalized VUV lines in a separate figure.
        """
        for k in xrange(0, self.vuv_signals_norm_combined.shape[0]):
            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            a.errorbar(
                self.vuv_times_combined[k, :],
                self.vuv_signals_norm_combined[k, :],
                yerr=self.vuv_uncertainties_norm_combined[k, :],
                fmt='.'
            )
            a.set_xlabel('$t$ [s]')
            a.set_ylabel('normalized signal')
    
    def write_atomdat(self):
        """Write the Ca.atomdat file:
        """
        # TODO: This should be generalized for other HiReX-SR lines.
        line_spec = LINE_SPEC_TEMPLATE.format(charge=18, wavelength=3.173, halfwidth=0.001)
        self.hirex_line_idx = 0
        self.xeus_line_idxs = []
        self.loweus_line_idxs = []
        k = 1
        for spectrometer, s in self.vuv_lines.iteritems():
            for l in s:
                if l.diagnostic_lines is not None:
                    if spectrometer == 'XEUS':
                        self.xeus_line_idxs.append(k)
                    elif spectrometer == 'LoWEUS':
                        self.loweus_line_idxs.append(k)
                    k += 1
                    # New way: pick the highest and lowest line, and go
                    # 0.0001nm to either side:
                    # Note that this assumes you haven't mixed charge states in
                    # a given line, since STRAHL doesn't appear to support this!
                    # TODO: I should be able to add support for that in my own
                    # code, if needed.
                    lam = CA_LINES[l.diagnostic_lines]
                    i_max = lam.argmax()
                    i_min = lam.argmin()
                    l_max = lam[i_max]
                    l_min = lam[i_min]
                    cwl = (l_max + l_min) / 2.0
                    halfwidth = (l_max - l_min) / 2.0 + 0.0001
                    if max(l.diagnostic_lines) < len(CA_17_LINES):
                        line_spec += LINE_SPEC_TEMPLATE.format(
                            charge=17,
                            wavelength=cwl * 10.0,
                            halfwidth=halfwidth * 10.0
                        )
                    else:
                        line_spec += LINE_SPEC_TEMPLATE.format(
                            charge=16,
                            wavelength=cwl * 10.0,
                            halfwidth=halfwidth * 10.0
                        )
                    
                    # Old way: put each line in with a tiny window:
                    # for i in l.diagnostic_lines:
                    #     if i < len(CA_17_LINES):
                    #         line_spec += LINE_SPEC_TEMPLATE.format(
                    #             charge=17,
                    #             wavelength=CA_17_LINES[i] * 10.0
                    #         )
                    #     else:
                    #         line_spec += LINE_SPEC_TEMPLATE.format(
                    #             charge=16,
                    #             wavelength=CA_16_LINES[i - len(CA_17_LINES)] * 10.0
                    #         )
        with open('Ca.atomdat', 'w') as f:
            f.write(
                CA_ATOMDAT_TEMPLATE.format(
                    num_lines=len(line_spec.splitlines()),
                    line_spec=line_spec
                )
            )
    
    def load_Te(self):
        """Load and fit the Te data using gpfit.
        """
        self.Te_X, self.Te_res, self.Te_p = self.load_prof('Te', self.settings.Te_args)
    
    def load_ne(self):
        """Load and fit the ne data using gpfit.
        """
        self.ne_X, self.ne_res, self.ne_p = self.load_prof('ne', self.settings.ne_args)
    
    def load_prof(self, prof, flags):
        """Load the specified profile using gpfit.
        
        Parameters
        ----------
        prof : {'ne', 'Te'}
            The profile to fit.
        flags : list of str
            The command line flags to pass to gpfit. Must not contain --signal,
            --shot, --t-min, --t-max or --coordinate.
        """
        print(
            "Use gpfit to fit the %s profile. When the profile has been fit to "
            "your liking, press the 'exit' button." % (prof,)
        )
        argv = [
            '--signal', prof,
            '--shot', str(self.settings.shot),
            '--t-min', str(self.settings.time_1),
            '--t-max', str(self.settings.time_2),
            '--coordinate', 'r/a',
            '--no-a-over-L',
            '--x-pts'
        ]
        argv += [str(x) for x in self.settings.roa_grid]
        argv += flags
        return profiletools.gui.run_gui(argv=argv)
    
    def load_xtomo(self):
        """Load the data from each of the three XTOMO systems.
        """
        tree = MDSplus.Tree('xtomo', self.settings.shot)
        self.xtomo_sig = {}
        self.xtomo_t = {}
        self.xtomo_channel_mask = {}
        # Just put something dumb as a placeholder for the baseline subtraction
        # ranges. This will need to be set by hand in the GUI. This is a list
        # of lists of tuples. The outer list has one entry per injection. Each
        # injection then has one or more 2-tuples with the (start, stop) values
        # of the range(s) to use for baseline subtraction.
        self.xtomo_baseline_ranges = [[(0, 0.1),],] * len(self.settings.injections)
        for s in (1, 3, 5):
            self.xtomo_sig[s], self.xtomo_t[s] = self.load_xtomo_array(s, tree)
            if self.xtomo_sig[s] is not None:
                self.xtomo_channel_mask[s] = scipy.ones(
                    self.xtomo_sig[s].shape[0],
                    dtype=bool
                )
                self.plot_xtomo(s)
    
    def flag_xtomo(self):
        """Flag the bad XTOMO channels, select baseline subtraction.
        """
        root = XtomoWindow(self)
        root.mainloop()
        print('flag_xtomo done, returning interactive control.')
    
    def load_xtomo_array(self, array_num, tree, n_chords=38):
        """Load the data from a given XTOMO array.
        
        Returns a tuple of `sig`, `t`, where `sig` is an array of float with
        shape (`n_chords`, `len(t)`) holding the signals from each chord and `t`
        is the timebase (assumed to be the same for all chords).
        
        Parameters
        ----------
        array_num : int
            The XTOMO array number. Nominally one of {1, 3, 5}.
        tree : :py:class:`MDSplus.Tree` instance
            The XTOMO tree for the desired shot.
        n_chords : int, optional
            The number of chords in the array. The default is 38.
        """
        N = tree.getNode('brightnesses.array_{a_n:d}.chord_01'.format(a_n=array_num))
        try:
            t = scipy.asarray(N.dim_of().data(), dtype=float)
        except MDSplus.TdiException:
            warnings.warn(
                "No data for XTOMO {a_n:d}!".format(a_n=array_num),
                RuntimeWarning
            )
            return None, None
        
        sig = scipy.zeros((n_chords, len(t)))
        
        for n in xrange(0, n_chords):
            N = tree.getNode(
                'brightnesses.array_{a_n:d}.chord_{n:02d}'.format(
                    a_n=array_num,
                    n=n + 1
                )
            )
            # Some of the XTOMO 3 arrays on old shots have channels which are 5
            # points short. Doing it this way prevents that from being an issue.
            d = scipy.asarray(N.data(), dtype=float)
            sig[n, :len(d)] = d
        
        return sig, t
    
    def process_injections(self):
        """Parcel out the data into the individual injections, normalize and combine.
        """
        for k, i in enumerate(self.settings.injections):
            # First handle the HiReX-SR data:
            t_hirex_start, t_hirex_stop = profiletools.get_nearest_idx(
                [i.t_start, i.t_stop],
                self.hirex_time
            )
            i.hirex_signal = self.hirex_signal[t_hirex_start:t_hirex_stop + 1, :]
            i.hirex_flagged = self.hirex_flagged[t_hirex_start:t_hirex_stop + 1, :]
            i.hirex_uncertainty = self.hirex_uncertainty[t_hirex_start:t_hirex_stop + 1, :]
            i.hirex_time = self.hirex_time[t_hirex_start:t_hirex_stop + 1] - i.t_inj
            
            if self.settings.debug_plots:
                i.plot_hirex()
            
            # Then handle each of the VUV lines:
            i.vuv_signals = []
            i.vuv_uncertainties = []
            i.vuv_times = []
            
            for s in self.vuv_lines.keys():
                i_start, i_stop = profiletools.get_nearest_idx(
                    [i.t_start, i.t_stop],
                    self.vuv_time[s]
                )
                for l in self.vuv_lines[s]:
                    if l.diagnostic_lines is not None:
                        i.vuv_signals.append(
                            l.signal[i_start:i_stop + 1]
                        )
                        i.vuv_uncertainties.append(
                            l.uncertainty[i_start:i_stop + 1]
                        )
                        i.vuv_times.append(
                            self.vuv_time[s][i_start:i_stop + 1] - i.t_inj
                        )
            
            i.vuv_signals = scipy.asarray(i.vuv_signals)
            i.vuv_uncertainties = scipy.asarray(i.vuv_uncertainties)
            i.vuv_times = scipy.asarray(i.vuv_times)
            
            # Then handle the XTOMO systems:
            i.xtomo_channel_mask = self.xtomo_channel_mask
            i.xtomo_signal = {}
            i.xtomo_times = {}
            for s in (1, 3, 5):
                if self.xtomo_sig[s] is not None:
                    i_start, i_stop = profiletools.get_nearest_idx(
                        [i.t_start, i.t_stop],
                        self.xtomo_t[s]
                    )
                    
                    # Apply the baseline subtraction:
                    bsub_idxs = []
                    for r in self.xtomo_baseline_ranges[k]:
                        lb_idx, ub_idx = profiletools.get_nearest_idx(
                            r,
                            self.xtomo_t[s]
                        )
                        bsub_idxs.extend(range(lb_idx, ub_idx + 1))
                    # Reduce to just the unique values:
                    bsub_idxs = list(set(bsub_idxs))
                    bsub = scipy.mean(self.xtomo_sig[s][:, bsub_idxs], axis=1)
                    
                    i.xtomo_signal[s] = self.xtomo_sig[s][:, i_start:i_stop + 1] - bsub[:, None]
                    i.xtomo_times[s] = self.xtomo_t[s][i_start:i_stop + 1] - i.t_inj
                else:
                    i.xtomo_signal[s] = None
                    i.xtomo_times[s] = None
            
            i.normalize_data(debug_plots=self.settings.debug_plots)
        
        # Combine the injections for HiReX-SR:
        self.hirex_signal_norm_combined = scipy.vstack(
            [i.hirex_signal_norm for i in self.settings.injections]
        )
        self.hirex_uncertainty_norm_combined = scipy.vstack(
            [i.hirex_uncertainty_norm for i in self.settings.injections]
        )
        self.hirex_flagged_combined = scipy.vstack(
            [i.hirex_flagged for i in self.settings.injections]
        )
        self.hirex_time_combined = scipy.hstack(
            [i.hirex_time for i in self.settings.injections]
        )
        
        # Combine the injections for VUV:
        self.vuv_signals_norm_combined = scipy.hstack(
            [i.vuv_signals_norm for i in self.settings.injections]
        )
        self.vuv_uncertainties_norm_combined = scipy.hstack(
            [i.vuv_uncertainties_norm for i in self.settings.injections]
        )
        self.vuv_times_combined = scipy.hstack(
            [i.vuv_times for i in self.settings.injections]
        )
        
        # Combine the injections for the XTOMO systems:
        self.xtomo_signal_norm_combined = {}
        self.xtomo_times_combined = {}
        for s in (1, 3, 5):
            if self.xtomo_sig[s] is not None:
                self.xtomo_signal_norm_combined[s] = scipy.hstack(
                    [i.xtomo_signal_norm[s] for i in self.settings.injections]
                )
                # TODO: Handle uncertainties properly!
                self.xtomo_times_combined[s] = scipy.hstack(
                    [i.xtomo_times[s] for i in self.settings.injections]
                )
            else:
                self.xtomo_signal_norm_combined[s] = None
                self.xtomo_times_combined[s] = None
        
        if self.settings.debug_plots:
            self.plot_hirex(norm=True)
            self.plot_vuv_normed()
            for s in (1, 3, 5):
                if self.xtomo_signal_norm_combined[s] is not None:
                    self.plot_xtomo(s, norm=True)
    
    def plot_data(self, f=None, share_y=False, y_title=0.85, y_label='$b$ [AU]', max_ticks=None, rot_label=False, suptitle='HiReX-SR and VUV'):
        """Make a big plot with all of the HiReX-SR data and VUV line data.
        """
        if f is None:
            f = plt.figure()
        
        ncol = 6
        nrow = int(
            scipy.ceil(1.0 * self.hirex_signal_norm_combined.shape[1] / ncol) +
            scipy.ceil(1.0 * self.vuv_signals_norm_combined.shape[0] / ncol)
        )
        gs = mplgs.GridSpec(nrow, ncol)
        
        a_H = []
        i_col = 0
        i_row = 0
        for k in xrange(0, self.hirex_signal_norm_combined.shape[1]):
            a_H.append(
                f.add_subplot(
                    gs[i_row, i_col],
                    sharex=a_H[0] if len(a_H) >= 1 else None,
                    sharey=a_H[0] if len(a_H) >= 1 and share_y else None
                )
            )
            if i_col > 0 and share_y:
                plt.setp(a_H[-1].get_yticklabels(), visible=False)
            else:
                a_H[-1].set_ylabel(y_label)
            if i_col < self.vuv_signals_norm_combined.shape[0] or i_row < scipy.ceil(1.0 * self.hirex_signal_norm_combined.shape[1] / ncol) - 2:
                plt.setp(a_H[-1].get_xticklabels(), visible=False)
            else:
                a_H[-1].set_xlabel('$t$ [s]')
                if rot_label:
                    plt.setp(a_H[-1].xaxis.get_majorticklabels(), rotation=90)
                    for tick in a_H[-1].get_yaxis().get_major_ticks():
                        tick.label1 = tick._get_text1()
            i_col += 1
            if i_col >= ncol:
                i_col = 0
                i_row += 1
        
        a_V = []
        # Only increment row if it hasn't been done already:
        if i_col != 0:
            i_row += 1
        i_col = 0
        for k in xrange(0, self.vuv_signals_norm_combined.shape[0]):
            a_V.append(
                f.add_subplot(
                    gs[i_row, i_col],
                    sharex=a_H[0],
                    sharey=a_H[0] if share_y else None
                )
            )
            if i_col > 0 and share_y:
                plt.setp(a_V[-1].get_yticklabels(), visible=False)
            else:
                a_V[-1].set_ylabel(y_label)
            i_col += 1
            if i_col >= ncol:
                i_col = 0
                i_row += 1
        
        for k in xrange(0, len(a_H)):
            a_H[k].set_title('chord %d' % (k,), y=y_title)
            # a_H[k].set_ylim(bottom=0)
            good = ~self.hirex_flagged_combined[:, k]
            a_H[k].errorbar(
                self.hirex_time_combined[good],
                self.hirex_signal_norm_combined[good, k],
                yerr=self.hirex_uncertainty_norm_combined[good, k],
                fmt='.'
            )
            if max_ticks is not None:
                a_H[k].xaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks - 1))
                a_H[k].yaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks - 1))
        
        for k in xrange(0, len(a_V)):
            a_V[k].set_xlabel('$t$ [s]')
            if rot_label:
                plt.setp(a_V[k].xaxis.get_majorticklabels(), rotation=90)
                for tick in a_V[k].get_yaxis().get_major_ticks():
                    tick.label1 = tick._get_text1()
            a_V[k].set_title('line %d' % (k,), y=y_title)
            # a_V[k].set_ylim(bottom=0)
            a_V[k].errorbar(
                self.vuv_times_combined[k, :],
                self.vuv_signals_norm_combined[k, :],
                yerr=self.vuv_uncertainties_norm_combined[k, :],
                fmt='.',
                color='g'
            )
            if max_ticks is not None:
                a_V[k].xaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks - 1))
                a_V[k].yaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks - 1))
        
        if share_y:
            a_H[0].set_ylim(bottom=0.0)
            a_H[0].set_xlim(self.hirex_time_combined.min(), self.hirex_time_combined.max())
        
        f.suptitle(suptitle)
        
        f.canvas.draw()
        
        return (f, a_H, a_V)

class Injection(object):
    """Class to store information on a given injection.
    """
    def __init__(self, t_inj, t_start, t_stop):
        self.t_inj = t_inj
        self.t_start = t_start
        self.t_stop = t_stop
    
    def normalize_data(self, debug_plots=False):
        # First handle the HiReX-SR data:
        # Normalize to the brightest interpolated max on the brightest chord:
        maxs = scipy.zeros(self.hirex_signal.shape[1])
        s_maxs = scipy.zeros_like(maxs)
        
        for k in xrange(0, self.hirex_signal.shape[1]):
            good = ~self.hirex_flagged[:, k]
            maxs[k], s_maxs[k] = interp_max(
                self.hirex_time[good],
                self.hirex_signal[good, k],
                err_y=self.hirex_uncertainty[good, k],
                debug_plots=debug_plots > 1
            )
        
        i = maxs.argmax()
        m = maxs[i]
        s = s_maxs[i]
        
        self.hirex_signal_norm = self.hirex_signal / m
        self.hirex_signal_norm[self.hirex_flagged] = scipy.nan
        
        self.hirex_uncertainty_norm = (
            scipy.absolute(self.hirex_signal_norm) *
            scipy.sqrt(
                (self.hirex_uncertainty / self.hirex_signal)**2.0 +
                (s / m)**2.0
                # Covariance???
            )
        )
        
        # Now handle the VUV data:
        # We don't have a brightness cal for XEUS or LoWEUS, so normalize to
        # the peak:
        self.vuv_signals_norm = scipy.nan * scipy.zeros_like(self.vuv_signals)
        self.vuv_uncertainties_norm = scipy.nan * scipy.zeros_like(self.vuv_uncertainties)
        
        for k in xrange(0, self.vuv_signals.shape[0]):
            m, s = interp_max(
                self.vuv_times[k, :],
                self.vuv_signals[k, :],
                err_y=self.vuv_uncertainties[k, :],
                debug_plots=debug_plots > 1,
                s_max=100.0
            )
            self.vuv_signals_norm[k, :] = self.vuv_signals[k, :] / m
            self.vuv_uncertainties_norm[k, :] = (
                scipy.absolute(self.vuv_signals_norm[k, :]) *
                scipy.sqrt(
                    (self.vuv_uncertainties[k, :] / self.vuv_signals[k, :])**2.0 +
                    (s / m)**2.0
                    # Covariance???
                )
            )
        
        # Now handle the XTOMO data:
        # Trying to do this like I did HiReX-SR runs my system out of memory.
        # Instead I will just normalize to the highest value, since the time
        # resolution is so high. But, I need to make sure I throw out the bad
        # channels!
        self.xtomo_signal_norm = {}
        for s in (1, 3, 5):
            if self.xtomo_signal[s] is not None:
                self.xtomo_signal_norm[s] = self.xtomo_signal[s] / self.xtomo_signal[s][self.xtomo_channel_mask[s], :].max()
                # TODO: Handle uncertainty properly!
            else:
                self.xtomo_signal_norm[s] = None
    
    def plot_hirex(self):
        """Plot the HiReX-SR data with the interpolation.
        """
        f = plt.figure()
        gs = mplgs.GridSpec(2, 1, height_ratios=[10, 1])
        l = []
        
        a_data = f.add_subplot(gs[0, :])
        a_slider = f.add_subplot(gs[1, :])
        a_data.set_xlabel("$t$ [s]")
        a_data.set_ylabel("HiReX-SR data [AU]")
        
        grid = scipy.linspace(self.t_start, self.t_stop, 1000) - self.t_inj
        
        def update(dum):
            remove_all(l)
            while len(l) > 0:
                l.pop()
            
            i = int(slider.val)
            
            good = ~self.hirex_flagged[:, i]
            
            l.append(
                a_data.errorbar(
                    self.hirex_time[good],
                    self.hirex_signal[good, i],
                    yerr=self.hirex_uncertainty[good, i],
                    color='b',
                    fmt='.'
                )
            )
            
            k = gptools.SquaredExponentialKernel(
                param_bounds=[(0, 10), (0, 2.0)],
                initial_params=[0.2, 0.03],
                fixed_params=[False, True]
            )
            gp = gptools.GaussianProcess(
                k,
                X=self.hirex_time[good],
                y=self.hirex_signal[good, i],
                err_y=self.hirex_uncertainty[good, i]
            )
            gp.optimize_hyperparameters(verbose=True)
            m_gp, s_gp = gp.predict(grid)
            
            l.append(a_data.plot(grid, m_gp, 'g'))
            l.append(
                a_data.fill_between(
                    grid,
                    m_gp - s_gp,
                    m_gp + s_gp,
                    alpha=0.25,
                    color='g'
                )
            )
            
            a_data.relim()
            a_data.autoscale_view()
            
            f.canvas.draw()
        
        def arrow_respond(slider, event):
            if event.key == 'right':
                slider.set_val(min(slider.val + 1, slider.valmax))
            elif event.key == 'left':
                slider.set_val(max(slider.val - 1, slider.valmin))
        
        slider = mplw.Slider(
            a_slider,
            'channel',
            0,
            self.hirex_signal.shape[1] - 1,
            valinit=0,
            valfmt='%d'
        )
        slider.on_changed(update)
        update(0)
        f.canvas.mpl_connect(
            'key_press_event',
            lambda evt: arrow_respond(slider, evt)
        )

def slider_plot(x, y, z, xlabel='', ylabel='', zlabel='', labels=None, **kwargs):
    """Make a plot to explore multidimensional data.
    
    x : array of float, (`M`,)
        The abscissa.
    y : array of float, (`N`,)
        The variable to slide over.
    z : array of float, (`P`, `M`, `N`)
        The variables to plot.
    xlabel : str, optional
        The label for the abscissa.
    ylabel : str, optional
        The label for the slider.
    zlabel : str, optional
        The label for the ordinate.
    labels : list of str with length `P`
        The labels for each curve in `z`.
    """
    if labels is None:
        labels = ['' for v in z]
    f = plt.figure()
    gs = mplgs.GridSpec(2, 1, height_ratios=[10, 1])
    a_plot = f.add_subplot(gs[0, :])
    a_slider = f.add_subplot(gs[1, :])
    
    a_plot.set_xlabel(xlabel)
    a_plot.set_ylabel(zlabel)
    
    l = []
    
    title = f.suptitle('')
    
    color_vals = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    style_vals = ['-', '--', '-.', ':']
    ls_vals = []
    for s in style_vals:
        for c in color_vals:
            ls_vals.append(c + s)
    
    def update(dum):
        ls_cycle = itertools.cycle(ls_vals)
        remove_all(l)
        while l:
            l.pop()
        
        i = int(slider.val)
        
        for v, l_ in zip(z, labels):
            l.append(a_plot.plot(x, v[:, i], ls_cycle.next(), label=l_, **kwargs))
        
        a_plot.relim()
        a_plot.autoscale_view()
        
        a_plot.legend(loc='best')
        
        title.set_text('%s = %.5f' % (ylabel, y[i]) if ylabel else '%.5f' % (y[i],))
        
        f.canvas.draw()
    
    def arrow_respond(slider, event):
        if event.key == 'right':
            slider.set_val(min(slider.val + 1, slider.valmax))
        elif event.key == 'left':
            slider.set_val(max(slider.val - 1, slider.valmin))
    
    slider = mplw.Slider(
        a_slider,
        ylabel,
        0,
        len(y) - 1,
        valinit=0,
        valfmt='%d'
    )
    slider.on_changed(update)
    update(0)
    f.canvas.mpl_connect(
        'key_press_event',
        lambda evt: arrow_respond(slider, evt)
    )

if _have_PyGMO:
    class MAPProblem(PyGMO.problem.base):
        """Problem to be used with PyGMO for global optimization.
        """
        def __init__(self, run=None):
            # This goofy argument nonsense is to deal with how copy.deepcopy works.
            if run is None:
                super(MAPProblem, self).__init__(1)
            else:
                self.run = run
                if run.method == 'GP':
                    ndim = (
                        run.num_eig_D +
                        run.num_eig_V +
                        run.k_D.num_free_params +
                        run.mu_D.num_free_params +
                        run.k_V.num_free_params +
                        ((7 if run.clusters else 5) if run.source_file is None else 2)
                    )
                elif run.method == 'spline':
                    ndim = (
                        run.num_eig_D +
                        run.num_eig_V +
                        (run.num_eig_D - run.spline_k_D if run.free_knots else 0) +
                        (run.num_eig_V - run.spline_k_V if run.free_knots else 0) +
                        ((7 if run.clusters else 5) if run.source_file is None else 2)
                    )
                elif run.method == 'linterp':
                    ndim = (
                        run.num_eig_D +
                        run.num_eig_V +
                        (run.num_eig_D - 1 if run.free_knots else 0) +
                        (run.num_eig_V - 1 if run.free_knots else 0) +
                        ((7 if run.clusters else 5) if run.source_file is None else 2)
                    )
                if run.use_scaling:
                    ndim += 1 + run.run_data.vuv_signals_norm_combined.shape[0]
                
                if run.method in ('spline', 'linterp') and run.free_knots:
                    if run.method == 'spline':
                        cdim = (run.num_eig_D - run.spline_k_D - 1) + (run.num_eig_V - run.spline_k_V - 1)
                    else:
                        cdim = (run.num_eig_D - 1 - 1) + (run.num_eig_V - 1 - 1)
                else:
                    cdim = 0
                
                super(MAPProblem, self).__init__(ndim)#, 0, 1, cdim, cdim, 0)
                
                bounds = scipy.asarray(self.run.get_prior().bounds[:])
                
                self.set_bounds(bounds[:, 0], bounds[:, 1])
        
        def _objfun_impl(self, x):
            acquire_working_dir()
            try:
                out = self.run.compute_ln_prob(
                    x,
                    sign=-1
                )
            except:
                warnings.warn(
                    "Unhandled exception. Error is: %s: %s. "
                    "Params are: %s" % (
                        sys.exc_info()[0],
                        sys.exc_info()[1],
                        x
                    )
                )
                out = scipy.inf
            finally:
                release_working_dir()
            # Some optimizers don't like infinite values, apparently:
            if scipy.isinf(out):
                out = sys.float_info.max
            else:
                out = float(out)
            return (out,)
    
        # def _compute_constraints_impl(self, x):
        #     """Compute the constraints on the knots.
        #
        #     I assume PyGMO will respect the bounds in any case.
        #
        #     I also assume PyGMO will treat a constraint as satisfied if it is >= 0.
        #     """
        #     eig_D, eig_V, knots_D, knots_V, hp_D, hp_mu_D, hp_V, param_scaling, param_source = self.run.split_params(x)
        #
        #     return scipy.concatenate((scipy.diff(knots_D), scipy.diff(knots_V)))

class _ComputeLnProbWrapper(object):
    """Wrapper to support parallel execution of STRAHL runs.
    
    This is needed since instance methods are not pickleable.
    
    Parameters
    ----------
    run : :py:class:`Run` instance
        The :py:class:`Run` to wrap.
    make_dir : bool, optional
        If True, a new STRAHL directory is acquired and released for each call.
        Default is False (run in current directory).
    for_min : bool, optional
        If True, the function is wrapped in the way it needs to be for a
        minimization: only -1 times the log-posterior is returned, independent
        of the value of `return_blob`.
    denormalize : bool, optional
        If True, a normalization from [lb, ub] to [0, 1] is removed. Default is
        False (don't adjust parameters).
    """
    def __init__(self, run, make_dir=False, for_min=False, denormalize=False):
        self.run = run
        self.make_dir = make_dir
        self.for_min = for_min
        self.denormalize = denormalize
    
    def __call__(self, params, **kwargs):
        if self.denormalize:
            bounds = scipy.asarray(self.run.get_prior().bounds[:], dtype=float)
            lb = bounds[:, 0]
            ub = bounds[:, 1]
            params = [x * (u - l) + l for x, u, l in zip(params, ub, lb)]
        try:
            if self.make_dir:
                acquire_working_dir()
            out = self.run.DV2ln_prob(
                params,
                sign=(-1 if self.for_min else 1),
                **kwargs
            )
        except:
            warnings.warn(
                "Unhandled exception. Error is: %s: %s. "
                "Params are: %s" % (
                    sys.exc_info()[0],
                    sys.exc_info()[1],
                    params
                )
            )
            if self.for_min:
                out = scipy.inf
            else:
                # if kwargs.get('return_blob', False):
                #     if kwargs.get('light_blob', False):
                #         out = (-scipy.inf)
                #     else:
                #         out = (-scipy.inf, (-scipy.inf, None, None, None, ''))
                # else:
                out = -scipy.inf
        finally:
            if self.make_dir:
                release_working_dir()
        return out

class _UGradEval(object):
    """Wrapper object for evaluating :py:meth:`Run.u2ln_prob` in parallel.
    """
    def __init__(self, run, sign, kwargs):
        self.run = run
        self.sign = sign
        self.kwargs = kwargs
    
    def __call__(self, p):
        return self.run.u2ln_prob(p, sign=self.sign, **self.kwargs)

class _OptimizeEval(object):
    """Wrapper class to allow parallel execution of random starts when optimizing the parameters.
    
    Parameters
    ----------
    run : :py:class:`Run`
        The :py:class:`Run` instance to wrap.
    thresh : float, optional
        If True, a test run of the starting 
    """
    def __init__(self, run, thresh=None):
        self.run = run
        # Get the bounds into the correct format for scipy.optimize.minimize:
        b = self.run.get_prior().bounds[:]
        self.bounds = [list(v) for v in b]
        for v in self.bounds:
            if scipy.isinf(v[0]):
                v[0] = None
            if scipy.isinf(v[1]):
                v[1] = None
        self.thresh = thresh
    
    def __call__(self, params):
        """Run the optimizer starting at the given params.
        
        All exceptions are caught and reported.
        
        Returns a tuple of (`u_opt`, `f_opt`, `return_code`, `num_strahl_calls`).
        If it fails, returns a tuple of (None, None, `sys.exc_info()`, `num_strahl_calls`).
        """
        global NUM_STRAHL_CALLS
        
        try:
            if self.thresh is not None:
                l = self.run.DV2ln_prob(params, sign=-1)
                if scipy.isinf(l) or scipy.isnan(l) or l > self.thresh:
                    warnings.warn("Bad start, skipping! lp=%.3g" % (l,))
                    return None
                else:
                    print("Good start: lp=%.3g" % (l,))
            # out = scipy.optimize.minimize(
            #     self.run.u2ln_prob,
            #     self.run.get_prior().elementwise_cdf(params),
            #     args=(None, -1, True),
            #     jac=True,
            #     method='TNC',
            #     bounds=[(0, 1),] * len(params),
            #     options={
            #         'disp': True,
            #         'maxfun': 50000,
            #         'maxiter': 50000,
            #         # 'maxls': 50, # Doesn't seem to be supported. WTF?
            #         'maxcor': 50
            #     }
            # )
            NUM_STRAHL_CALLS = 0
            # out = scipy.optimize.fmin_l_bfgs_b(
            #     self.run.u2ln_prob,
            #     self.run.get_prior().elementwise_cdf(params),
            #     args=(None, -1, True),
            #     bounds=[(0, 1),] * len(params),
            #     iprint=50,
            #     maxfun=50000
            # )
            opt = nlopt.opt(nlopt.LN_SBPLX, len(params))
            opt.set_max_objective(self.run.u2ln_prob)
            opt.set_lower_bounds([0.0,] * opt.get_dimension())
            opt.set_upper_bounds([1.0,] * opt.get_dimension())
            opt.set_ftol_abs(1.0)
            # opt.set_maxeval(40000)#(100000)
            opt.set_maxtime(3600 * 12)
            uopt = opt.optimize(self.run.get_prior().elementwise_cdf(params))
            out = (uopt, opt.last_optimum_value(), opt.last_optimize_result(), NUM_STRAHL_CALLS)
            print("Done. Made %d calls to STRAHL." % (NUM_STRAHL_CALLS,))
            return out
        except:
            warnings.warn(
                "Minimizer failed, skipping sample. Error is: %s: %s."
                % (
                    sys.exc_info()[0],
                    sys.exc_info()[1]
                )
            )
            return (None, None, sys.exc_info(), NUM_STRAHL_CALLS)

class _ComputeProfileWrapper(object):
    """Wrapper to enable evaluation of D, V profiles in parallel.
    """
    def __init__(self, run):
        self.run = run
    
    def __call__(self, params):
        return self.run.eval_DV(params)

class _CallGPWrapper(object):
    """Wrapper to enable use of GaussianProcess instances with emcee Samplers.
    
    Enforces the bounds when called to prevent runaway extrapolations.
    """
    def __init__(self, gp):
        self.gp = gp
        # Capture the X limits:
        self.X_min = gp.X.min(axis=0)
        self.X_max = gp.X.max(axis=0)
    
    def __call__(self, X, **kwargs):
        X = scipy.asarray(X)
        if (X < self.X_min).any() or (X > self.X_max).any():
            return -scipy.inf
        else:
            return self.gp.predict(X, **kwargs)

def send_email(subject, body, impath):
    """Send an email with the given subject, body and image attachment.
    
    Parameters
    ----------
    subject : str
        Subject of the email.
    body : str
        Body of the email.
    impath : str
        Path to the image to attach to the email.
    """
    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = "markchil@mit.edu"
        msg['To'] = "markchil@mit.edu"
        msg.attach(MIMEText(body))
        for im in impath:
            with open(im, "rb") as f:
                msg.attach(
                    MIMEApplication(
                        f.read(),
                        Content_Disposition='attachment; filename="%s"' % (os.path.basename(im),)
                    )
                )
        
        s = smtplib.SMTP("mail1.psfc.mit.edu")
        s.sendmail("markchil@mit.edu", ["markchil@mit.edu"], msg.as_string())
        s.quit()
    except:
        warnings.warn("Message not sent!")

def eval_profile(x, k, eig, n, params=None, mu=None):
    """Evaluate the profile.
    
    Note that you must externally exponentiate the D profile to ensure
    positivity.
    
    Parameters
    ----------
    x : array of float
        The points to evaluate the profile at.
    k : :py:class:`gptools.Kernel` instance
        The covariance kernel to use.
    eig : array of float
        The eigenvalues to use when drawing the sample.
    n : int
        The derivative order to set to 0 at the origin. If `n`=0 then the
        value is manually forced to zero to avoid numerical issues. The sign
        of the eigenvectors is also chosen based on `n`: `n`=0 uses the left
        slope constraint, `n`=1 uses the left concavity constraint.
    params : array of float, optional
        The values for the (free) hyperparameters of `k`. If provided, the
        hyperparameters of `k` are first updated. Otherwise, `k` is used as-
        is (i.e., it assumes the hyperparameters were set elsewhere).
    """
    if params is not None:
        if mu is None:
            k.set_hyperparams(params)
        else:
            k.set_hyperparams(params[:k.num_free_params])
            mu.set_hyperparams(params[k.num_free_params:])
    if eig.ndim == 1:
        eig = scipy.atleast_2d(eig).T
    gp = gptools.GaussianProcess(k, mu=mu)
    gp.add_data(0, 0, n=n)
    y = gp.draw_sample(
        x,
        method='eig',
        num_eig=len(eig),
        rand_vars=eig,
        modify_sign='left concavity' if n == 1 else 'left slope'
    ).ravel()
    
    if n == 0:
        y[0] = 0
    
    return y

def source_function(t, t_start, t_rise, n_rise, t_fall, n_fall, t_cluster=0.0, h_cluster=0.0):
    """Defines a model form to approximate the shape of the source function.
    
    Consists of an exponential rise, followed by an exponential decay and,
    optionally, a constant tail to approximate clusters.
    
    The cluster period is optional, so you can either treat this as a
    5-parameter function or a 7-parameter function.
    
    The function is set to have a peak value of 1.0.
    
    Parameters
    ----------
    t : array of float
        The time values to evaluate the source at.
    t_start : float
        The time the source starts at.
    t_rise : float
        The length of the rise portion.
    n_rise : float
        The number of e-folding times to put in the rise portion.
    t_fall : float
        The length of the fall portion.
    n_fall : float
        The number of e-folding times to put in the fall portion.
    t_cluster : float, optional
        The length of the constant period. Default is 0.0.
    h_cluster : float, optional
        The height of the constant period. Default is 0.0.
    """
    s = scipy.atleast_1d(scipy.zeros_like(t))
    tau_rise = t_rise / n_rise
    tau_fall = t_fall / n_fall
    
    rise_idx = (t >= t_start) & (t < t_start + t_rise)
    s[rise_idx] = 1.0 - scipy.exp(-(t[rise_idx] - t_start) / tau_rise)
    
    fall_idx = (t >= t_start + t_rise) & (t < t_start + t_rise + t_fall)
    s[fall_idx] = scipy.exp(-(t[fall_idx] - t_start - t_rise) / tau_fall)
    
    s[(t >= t_start + t_rise + t_fall) & (t < t_start + t_rise + t_fall + t_cluster)] = h_cluster
    
    return s

def interp_max(x, y, err_y=None, s_guess=0.2, s_max=10.0, l_guess=0.005, fixed_l=False, debug_plots=False):
    """Compute the maximum value of the smoothed data.
    
    Estimates the uncertainty using Gaussian process regression and returns the
    mean and uncertainty.
    
    Parameters
    ----------
    x : array of float
        Abscissa of data to be interpolated.
    y : array of float
        Data to be interpolated.
    err_y : array of float, optional
        Uncertainty in `y`. If absent, the data are interpolated.
    s_guess : float, optional
        Initial guess for the signal variance. Default is 0.2.
    s_max : float, optional
        Maximum value for the signal variance. Default is 10.0
    l_guess : float, optional
        Initial guess for the covariance length scale. Default is 0.03.
    fixed_l : bool, optional
        Set to True to hold the covariance length scale fixed during the MAP
        estimate. This helps mitigate the effect of bad points. Default is True.
    debug_plots : bool, optional
        Set to True to plot the data, the smoothed curve (with uncertainty) and
        the location of the peak value.
    """
    grid = scipy.linspace(max(0, x.min()), min(0.08, x.max()), 1000)
    hp = (
        gptools.UniformJointPrior([(0, s_max),]) *
        gptools.GammaJointPriorAlt([l_guess,], [0.1,])
    )
    k = gptools.SquaredExponentialKernel(
        # param_bounds=[(0, s_max), (0, 2.0)],
        hyperprior=hp,
        initial_params=[s_guess, l_guess],
        fixed_params=[False, fixed_l]
    )
    gp = gptools.GaussianProcess(k, X=x, y=y, err_y=err_y)
    gp.optimize_hyperparameters(verbose=True, random_starts=100)
    m_gp, s_gp = gp.predict(grid)
    i = m_gp.argmax()
    
    if debug_plots:
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        a.errorbar(x, y, yerr=err_y, fmt='.', color='b')
        a.plot(grid, m_gp, color='g')
        a.fill_between(grid, m_gp - s_gp, m_gp + s_gp, color='g', alpha=0.5)
        a.axvline(grid[i])
    
    return (m_gp[i], s_gp[i])

# def remove_all(l):
#     """Remove all of the various hierarchical objects matplotlib spits out.
#     """
#     # TODO: This is a hack -- store the objects better!
#     for v in l:
#         try:
#             for vv in v:
#                 try:
#                     for vvv in vv:
#                         vvv.remove()
#                 except TypeError:
#                     vv.remove()
#         except TypeError:
#             v.remove()

# def remove_all(v):
#     """Remove all of the various hierarchical objects matplotlib spits out.
#     """
#     # TODO: This is a hack -- store the objects better!
#     # A list needs an argument for remove to work, so the correct exception is
#     # TypeError.
#     try:
#         print(type(v))
#         v.remove()
#     except TypeError:
#         for vv in v:
#             remove_all(vv)
#     except Exception as e:
#         import pdb
#         pdb.set_trace()

def remove_all(v):
    """Yet another recursive remover, because matplotlib is stupid.
    """
    try:
        for vv in v:
            remove_all(vv)
    except TypeError:
        v.remove()

def write_Ca_16_ADF15(
        path='atomdat/adf15/ca/pue#ca16.dat',
        Te=[5e1, 1e2, 2e2, 5e2, 7.5e2, 1e3, 1.5e3, 2e3, 4e3, 7e3, 1e4, 2e4],
        ne=[1e12, 1e13, 2e13, 5e13, 1e14, 2e14, 5e14, 1e15, 2e15]
    ):
    """Write an ADF15-formatted file for the 19.3nm Ca 16+ line.
    
    Computes the photon emissivity coefficients as a function of temperature and
    density using the expression John Rice found for me.
    
    TODO: GET CITATION!
    TODO: Verify against Ar rates!
    
    Parameters
    ----------
    path : str, optional
        The path to write the file to. Default is
        'atomdat/adf15/ca/pue#ca16.dat'.
    Te : array of float, optional
        Temperatures to evaluate the model at in eV. Defaults to the values used
        in pue#ca17.dat.
    ne : array of float, optional
        Densities to evaluate the model at in cm^-3. Defaults to the values used
        in pue#ca17.dat.
    """
    # Only one transition available:
    NSEL = 1
    TEXT = 'CA+16 EMISSIVITY COEFFTS.'
    # Convert to angstroms:
    WLNG = CA_16_LINES[0] * 10
    NDENS = len(ne)
    NTE = len(Te)
    FILMEM = 'none'
    TYPE = 'EXCIT'
    INDM = 'T'
    ISEL = 1
    
    s = (
        "{NSEL: >5d}  /{TEXT:s}/\n"
        "{WLNG: >8.3f} A{NDENS: >4d}{NTE: >4d} /FILMEM = {FILMEM: <8s}/TYPE = {TYPE: <8s} /INDM = {INDM:s}/ISEL ={ISEL: >5d}\n".format(
            NSEL=NSEL,
            TEXT=TEXT,
            WLNG=WLNG,
            NDENS=NDENS,
            NTE=NTE,
            FILMEM=FILMEM,
            TYPE=TYPE,
            INDM=INDM,
            ISEL=ISEL
        )
    )
    ne_str = ['{: >9.2e}'.format(i) for i in ne]
    ct = 0
    while ne_str:
        s += ne_str.pop(0)
        ct += 1
        if ct == 8:
            s += '\n'
            ct = 0
    if ct != 0:
        s += '\n'
    Te_str = ['{: >9.2e}'.format(i) for i in Te]
    ct = 0
    while Te_str:
        s += Te_str.pop(0)
        ct += 1
        if ct == 8:
            s += '\n'
            ct = 0
    if ct != 0:
        s += '\n'
    
    # Information from John Rice:
    ne = scipy.asarray(ne, dtype=float)
    Te = scipy.asarray(Te, dtype=float)
    fij = 0.17
    Eij = 64.3
    y = Eij / Te
    A = 0.6
    D = 0.28
    gbar = A + D * (scipy.log((y + 1) / y) - 0.4 / (y + 1)**2.0)
    PEC = 1.57e-5 / (scipy.sqrt(Te) * Eij) * fij * gbar * scipy.exp(-y)
    
    PEC_str = ['{: >9.2e}'.format(i) for i in PEC]
    PEC_fmt = ''
    ct = 0
    while PEC_str:
        PEC_fmt += PEC_str.pop(0)
        ct += 1
        if ct == 8:
            PEC_fmt += '\n'
            ct = 0
    if ct != 0:
        PEC_fmt += '\n'
    s += PEC_fmt * NDENS
    
    with open(path, 'w') as f:
        f.write(s)

class HirexPlotFrame(tk.Frame):
    """Frame to hold the plot with the HiReX-SR time-series data.
    """
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.f = Figure()
        self.suptitle = self.f.suptitle('')
        self.a = self.f.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.f, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='NESW')
        
        # Need to put the toolbar in its own frame, since it automatically calls
        # pack on itself, but I am using grid.
        self.toolbar_frame = tk.Frame(self)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar_frame.grid(row=1, column=0, sticky='EW')
        
        self.canvas.mpl_connect(
            'button_press_event',
            lambda event: self.canvas._tkcanvas.focus_set()
        )
        self.canvas.mpl_connect('key_press_event', self.on_key_event)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.a.set_xlabel('$t$ [s]')
        self.a.set_ylabel('HiReX-SR signal [AU]')
        # TODO: Get a more clever way to handle ylim!
        self.a.set_ylim(0, 1)
        
        self.l = []
        self.l_flagged = []
    
    def on_key_event(self, evt):
        """Tie keys to the toolbar.
        """
        key_press_handler(evt, self.canvas, self.toolbar)

class HirexWindow(tk.Tk):
    """GUI to flag bad HiReX-SR points.
    
    Parameters
    ----------
    data : :py:class:`RunData` instance
        The :py:class:`RunData` object holding the information to be processed.
    """
    def __init__(self, data, ar=False):
        print(
            "Type the indices of the bad points into the text box and press "
            "ENTER to flag them. Use the arrow keys to move between channels. "
            "When done, close the window to continue with the analysis."
        )
        tk.Tk.__init__(self)
        
        self.data = data
        self.ar = ar
        
        self.wm_title("HiReX-SR inspector")
        
        self.plot_frame = HirexPlotFrame(self)
        self.plot_frame.grid(row=0, column=0, sticky='NESW')
        
        if self.ar:
            self.signal = data.ar_signal
            self.time = data.ar_time
            self.uncertainty = data.ar_uncertainty
            self.flagged = data.ar_flagged
        else:
            self.signal = data.hirex_signal
            self.time = data.hirex_time
            self.uncertainty = data.hirex_uncertainty
            self.flagged = data.hirex_flagged
        
        self.idx_slider = tk.Scale(
            master=self,
            from_=0,
            to=self.signal.shape[1] - 1,
            command=self.update_slider,
            orient=tk.HORIZONTAL
        )
        self.idx_slider.grid(row=1, column=0, sticky='NESW')
        
        self.flagged_box = tk.Entry(self)
        self.flagged_box.grid(row=2, column=0, sticky='NESW')
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.bind("<Left>", self.on_arrow)
        self.bind("<Right>", self.on_arrow)
        self.bind("<Return>", self.process_flagged)
        # self.bind("<Enter>", self.process_flagged)
        self.bind("<KP_Enter>", self.process_flagged)
    
    def destroy(self):
        self.process_flagged()
        tk.Tk.destroy(self)
    
    def on_arrow(self, evt):
        """Handle arrow keys to move slider.
        """
        if evt.keysym == 'Right':
            self.process_flagged()
            self.idx_slider.set(
                min(self.idx_slider.get() + 1, self.signal.shape[1] - 1)
            )
        elif evt.keysym == 'Left':
            self.process_flagged()
            self.idx_slider.set(
                max(self.idx_slider.get() - 1, 0)
            )
    
    def process_flagged(self, evt=None):
        """Process the flagged points which have been entered into the text box.
        """
        flagged = re.findall(
            LIST_REGEX,
            self.flagged_box.get()
        )
        flagged = scipy.asarray([int(i) for i in flagged], dtype=int)
        
        idx = self.idx_slider.get()
        self.flagged[:, idx] = False
        self.flagged[flagged, idx] = True
        
        remove_all(self.plot_frame.l_flagged)
        self.plot_frame.l_flagged = []
        
        self.plot_frame.l_flagged.append(
            self.plot_frame.a.plot(
                self.time[flagged],
                self.signal[flagged, idx],
                'rx',
                markersize=12
            )
        )
        
        self.plot_frame.canvas.draw()
    
    def update_slider(self, new_idx):
        """Update the slider to the new index.
        """
        # Remove the old lines:
        remove_all(self.plot_frame.l)
        self.plot_frame.l = []
        
        self.plot_frame.l.append(
            self.plot_frame.a.errorbar(
                self.time,
                self.signal[:, new_idx],
                yerr=self.uncertainty[:, new_idx],
                fmt='.',
                color='b'
            )
        )
        for i, x, y in zip(
                xrange(0, self.signal.shape[0]),
                self.time,
                self.signal[:, new_idx]
            ):
            self.plot_frame.l.append(
                self.plot_frame.a.text(x, y, str(i))
            )
        
        # Insert the flagged points into the textbox:
        self.flagged_box.delete(0, tk.END)
        self.flagged_box.insert(
            0,
            ', '.join(map(str, scipy.where(self.flagged[:, new_idx])[0]))
        )
        
        self.process_flagged()
        
        # Called by process_flagged:
        # self.plot_frame.canvas.draw()

class VuvPlotFrame(tk.Frame):
    """Frame to hold the plots with the XEUS data.
    """
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
        # Store the lines that change when updating the time:
        self.l_time = []
        
        # Store the lines that change when updating the wavelength:
        self.l_lam = []
        
        # Store the lines that change when updating the XEUS line:
        self.l_final = []
        
        self.f = Figure()
        self.suptitle = self.f.suptitle('')
        gs = mplgs.GridSpec(2, 2)
        self.a_im = self.f.add_subplot(gs[0, 0])
        self.a_spec = self.f.add_subplot(gs[1, 0])
        self.a_time = self.f.add_subplot(gs[0, 1])
        self.a_final = self.f.add_subplot(gs[1, 1])
        self.canvas = FigureCanvasTkAgg(self.f, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='NESW')
        
        # Need to put the toolbar in its own frame, since it automatically calls
        # pack on itself, but I am using grid.
        self.toolbar_frame = tk.Frame(self)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar_frame.grid(row=1, column=0, sticky='EW')
        
        self.canvas.mpl_connect(
            'button_press_event',
            lambda event: self.canvas._tkcanvas.focus_set()
        )
        self.canvas.mpl_connect('key_press_event', self.on_key_event)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Just plot the image now since it doesn't change:
        LAM, T = scipy.meshgrid(
            self.master.data.vuv_lam[self.master.system],
            self.master.data.vuv_time[self.master.system]
        )
        self.im = self.a_im.pcolormesh(
            LAM,
            T,
            self.master.data.vuv_signal[self.master.system],
            cmap='gray'
        )
        xlim = self.a_im.get_xlim()
        for x, i, c in zip(
                CA_17_LINES + CA_16_LINES,
                range(0, len(CA_17_LINES) + len(CA_16_LINES)),
                ['r'] * len(CA_17_LINES) + ['c'] * len(CA_16_LINES)
            ):
            self.a_im.axvline(x, linestyle='--', color=c)
            self.a_spec.axvline(x, linestyle='--', color=c)
            self.a_im.text(
                x,
                self.master.data.vuv_time[self.master.system].min(),
                str(i)
            )
            self.a_spec.text(x, 0, str(i))
        
        self.a_im.set_xlim(xlim)
        self.a_spec.set_xlim(xlim)
        
        self.a_im.set_xlabel(r'$\lambda$ [nm]')
        self.a_im.set_ylabel('$t$ [s]')
        
        self.a_spec.set_xlabel(r'$\lambda$ [nm]')
        self.a_spec.set_ylabel('raw signal [AU]')
        
        self.a_time.set_xlabel('$t$ [s]')
        self.a_time.set_ylabel('raw signal [AU]')
        
        self.a_final.set_xlabel('$t$ [s]')
        self.a_final.set_ylabel('processed signal [AU]')
    
    def on_key_event(self, evt):
        """Tie keys to the toolbar.
        """
        key_press_handler(evt, self.canvas, self.toolbar)
    
    def on_click(self, evt):
        """Move the cursors with a click in any given axis.
        
        Only does so if the widgetlock is not locked.
        """
        if not self.canvas.widgetlock.locked():
            if evt.inaxes == self.a_im:
                # Update both lam and t:
                lam_idx = profiletools.get_nearest_idx(
                    evt.xdata,
                    self.master.data.vuv_lam[self.master.system]
                )
                self.master.slider_frame.lam_slider.set(lam_idx)
                
                t_idx = profiletools.get_nearest_idx(
                    evt.ydata,
                    self.master.data.vuv_time[self.master.system]
                )
                self.master.slider_frame.t_slider.set(t_idx)
            elif evt.inaxes == self.a_spec:
                # Only update lam:
                lam_idx = profiletools.get_nearest_idx(
                    evt.xdata,
                    self.master.data.vuv_lam[self.master.system]
                )
                self.master.slider_frame.lam_slider.set(lam_idx)
            elif evt.inaxes == self.a_time:
                # Only update t:
                t_idx = profiletools.get_nearest_idx(
                    evt.xdata,
                    self.master.data.vuv_time[self.master.system]
                )
                self.master.slider_frame.t_slider.set(t_idx)

class VuvSliderFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
        self.t_idx = None
        self.lam_idx = None
        self.max_val = None
        
        self.t_slider = tk.Scale(
            master=self,
            from_=0,
            to=len(self.master.data.vuv_time[self.master.system]) - 1,
            command=self.master.update_t,
            orient=tk.HORIZONTAL,
            label='t'
        )
        self.t_slider.grid(row=0, column=0)
        
        self.lam_slider = tk.Scale(
            master=self,
            from_=0,
            to=len(self.master.data.vuv_lam[self.master.system]) - 1,
            command=self.master.update_lam,
            orient=tk.HORIZONTAL,
            label=u'\u03bb'
        )
        self.lam_slider.grid(row=0, column=1)
        
        self.max_val_slider = tk.Scale(
            master=self,
            from_=0,
            to=self.master.data.vuv_signal[self.master.system].max(),
            command=self.master.update_max_val,
            orient=tk.HORIZONTAL,
            label='max =',
            resolution=0.01
        )
        self.max_val_slider.set(self.master.data.vuv_signal[self.master.system].max())
        self.max_val_slider.grid(row=0, column=2)


class VuvWindow(tk.Tk):
    def __init__(self, data, system):
        tk.Tk.__init__(self)
        
        self.data = data
        self.system = system
        
        self.wm_title(system + " inspector")
        
        self.plot_frame = VuvPlotFrame(self)
        self.plot_frame.grid(row=0, column=0, sticky='NESW')
        
        self.slider_frame = VuvSliderFrame(self)
        self.slider_frame.grid(row=1, column=0, sticky='NESW')
        
        self.line_frame = VuvLineFrame(self)
        self.line_frame.grid(row=0, column=1, rowspan=2, sticky='NESW')
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.bind("<Left>", self.on_arrow)
        self.bind("<Right>", self.on_arrow)
        self.bind("<Up>", self.on_arrow)
        self.bind("<Down>", self.on_arrow)
    
    def on_arrow(self, evt):
        """Handle arrow keys to move slider.
        """
        if evt.keysym == 'Right':
            self.slider_frame.lam_slider.set(
                min(
                    self.slider_frame.lam_slider.get() + 1,
                    len(self.data.vuv_lam[self.system]) - 1
                )
            )
        elif evt.keysym == 'Left':
            self.slider_frame.lam_slider.set(
                max(self.slider_frame.lam_slider.get() - 1, 0)
            )
        elif evt.keysym == 'Up':
            self.slider_frame.t_slider.set(
                min(
                    self.slider_frame.t_slider.get() + 1,
                    len(self.data.vuv_time[self.system]) - 1
                )
            )
        elif evt.keysym == 'Down':
            self.slider_frame.t_slider.set(
                max(self.slider_frame.t_slider.get() - 1, 0)
            )
    
    def update_t(self, t_idx):
        """Update the time slice plotted.
        """
        # Cast to int, because Tkinter is inexplicably giving me str (!?)
        t_idx = int(t_idx)
        # Need to check this, since apparently getting cute with setting the
        # label creates an infinite recursion...
        if t_idx != self.slider_frame.t_idx:
            self.slider_frame.t_idx = t_idx
            self.slider_frame.t_slider.config(
                label="t = %.3fs" % (self.data.vuv_time[self.system][t_idx],)
            )
            remove_all(self.plot_frame.l_time)
            self.plot_frame.l_time = []
            self.plot_frame.l_time.append(
                self.plot_frame.a_spec.plot(
                    self.data.vuv_lam[self.system],
                    self.data.vuv_signal[self.system][t_idx, :],
                    'k'
                )
            )
            self.plot_frame.l_time.append(
                self.plot_frame.a_time.axvline(
                    self.data.vuv_time[self.system][t_idx],
                    color='b'
                )
            )
            self.plot_frame.l_time.append(
                self.plot_frame.a_im.axhline(
                    self.data.vuv_time[self.system][t_idx],
                    color='b'
                )
            )
            # self.plot_frame.a_spec.relim()
            # self.plot_frame.a_spec.autoscale_view(scalex=False)
            self.plot_frame.canvas.draw()
    
    def update_lam(self, lam_idx):
        """Update the wavelength slice plotted.
        """
        lam_idx = int(lam_idx)
        if lam_idx != self.slider_frame.lam_idx:
            self.slider_frame.lam_idx = lam_idx
            self.slider_frame.lam_slider.config(
                label=u"\u03bb = %.3fnm" % (self.data.vuv_lam[self.system][lam_idx],)
            )
            remove_all(self.plot_frame.l_lam)
            self.plot_frame.l_lam = []
            self.plot_frame.l_lam.append(
                self.plot_frame.a_time.plot(
                    self.data.vuv_time[self.system],
                    self.data.vuv_signal[self.system][:, lam_idx],
                    'k'
                )
            )
            self.plot_frame.l_lam.append(
                self.plot_frame.a_spec.axvline(
                    self.data.vuv_lam[self.system][lam_idx],
                    color='g'
                )
            )
            self.plot_frame.l_lam.append(
                self.plot_frame.a_im.axvline(
                    self.data.vuv_lam[self.system][lam_idx],
                    color='g'
                )
            )
            # self.plot_frame.a_time.relim()
            # self.plot_frame.a_time.autoscale_view(scalex=False)
            self.plot_frame.canvas.draw()
    
    def update_max_val(self, max_val):
        """Update the maximum value on the image plot.
        """
        max_val = float(max_val)
        if max_val != self.slider_frame.max_val:
            self.slider_frame.max_val = max_val
            self.plot_frame.im.set_clim(vmax=max_val)
            self.plot_frame.canvas.draw()

class VuvLineFrame(tk.Frame):
    """Frame that holds the controls to setup VUV line information.
    """
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
        # Keep track of the selected idx separately, since tkinter is stupid
        # about it (loses state when using tab to move between text boxes):
        self.idx = None
        
        self.listbox_label = tk.Label(self, text="defined lines:", anchor=tk.SW)
        self.listbox_label.grid(row=0, column=0, columnspan=2, sticky='NESW')
        
        self.listbox = tk.Listbox(self)
        self.listbox.grid(row=1, column=0, columnspan=2, sticky='NESW')
        self.listbox.bind('<<ListboxSelect>>', self.on_select)
        
        self.add_button = tk.Button(self, text="+", command=self.add_line)
        self.add_button.grid(row=2, column=0, sticky='NESW')
        
        self.remove_button = tk.Button(self, text="-", command=self.remove_line)
        self.remove_button.grid(row=2, column=1, sticky='NESW')
        
        self.included_lines_label = tk.Label(self, text="included lines:", anchor=tk.SW)
        self.included_lines_label.grid(row=3, column=0, columnspan=2, sticky='NESW')
        
        self.included_lines_box = tk.Entry(self)
        self.included_lines_box.grid(row=4, column=0, columnspan=2, sticky='NESW')
        
        self.lam_lb_label = tk.Label(self, text=u"\u03bb min (nm):", anchor=tk.SW)
        self.lam_lb_label.grid(row=5, column=0, sticky='NESW')
        
        self.lam_lb_box = tk.Entry(self)
        self.lam_lb_box.grid(row=6, column=0, sticky='NESW')
        
        self.lam_ub_label = tk.Label(self, text=u"\u03bb max (nm):", anchor=tk.SW)
        self.lam_ub_label.grid(row=5, column=1, sticky='NESW')
        
        self.lam_ub_box = tk.Entry(self)
        self.lam_ub_box.grid(row=6, column=1, sticky='NESW')
        
        self.t_lb_label = tk.Label(self, text="baseline start (s):", anchor=tk.SW)
        self.t_lb_label.grid(row=7, column=0, sticky='NESW')
        
        self.t_lb_box = tk.Entry(self)
        self.t_lb_box.grid(row=8, column=0, sticky='NESW')
        
        self.t_ub_label = tk.Label(self, text="baseline end (s):", anchor=tk.SW)
        self.t_ub_label.grid(row=7, column=1, sticky='NESW')
        
        self.t_ub_box = tk.Entry(self)
        self.t_ub_box.grid(row=8, column=1, sticky='NESW')
        
        self.apply_button = tk.Button(self, text="apply", command=self.on_apply)
        self.apply_button.grid(row=9, column=0, columnspan=2, sticky='NESW')
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Add the existing VuvLine instances to the GUI:
        if self.master.data.vuv_lines[self.master.system]:
            for l in self.master.data.vuv_lines[self.master.system]:
                self.listbox.insert(tk.END, ', '.join(map(str, l.diagnostic_lines)))
        else:
            self.add_line()
    
    def on_select(self, event):
        """Handle selection of a new line.
        """
        # TODO: This should save the current state into the selected line
        
        try:
            self.idx = self.listbox.curselection()[0]
        except IndexError:
            self.idx = None
        
        if self.idx is not None:
            self.included_lines_box.delete(0, tk.END)
            if self.master.data.vuv_lines[self.master.system][self.idx].diagnostic_lines is not None:
                self.included_lines_box.insert(
                    0,
                    ', '.join(
                        map(
                            str,
                            self.master.data.vuv_lines[self.master.system][self.idx].diagnostic_lines
                        )
                    )
                )
            
            self.lam_lb_box.delete(0, tk.END)
            if self.master.data.vuv_lines[self.master.system][self.idx].lam_lb is not None:
                self.lam_lb_box.insert(
                    0,
                    self.master.data.vuv_lines[self.master.system][self.idx].lam_lb
                )
            
            self.lam_ub_box.delete(0, tk.END)
            if self.master.data.vuv_lines[self.master.system][self.idx].lam_ub is not None:
                self.lam_ub_box.insert(
                    0,
                    self.master.data.vuv_lines[self.master.system][self.idx].lam_ub
                )
            
            if self.master.data.vuv_lines[self.master.system][self.idx].t_lb is not None:
                self.t_lb_box.delete(0, tk.END)
                self.t_lb_box.insert(
                    0,
                    self.master.data.vuv_lines[self.master.system][self.idx].t_lb
                )
            
            if self.master.data.vuv_lines[self.master.system][self.idx].t_ub is not None:
                self.t_ub_box.delete(0, tk.END)
                self.t_ub_box.insert(
                    0,
                    self.master.data.vuv_lines[self.master.system][self.idx].t_ub
                )
            
            remove_all(self.master.plot_frame.l_final)
            self.master.plot_frame.l_final = []
            if self.master.data.vuv_lines[self.master.system][self.idx].signal is not None:
                self.master.plot_frame.l_final.append(
                    self.master.plot_frame.a_final.plot(
                        self.master.data.vuv_time[self.master.system],
                        self.master.data.vuv_lines[self.master.system][self.idx].signal,
                        'k'
                    )
                )
            self.master.plot_frame.canvas.draw()
    
    def add_line(self):
        """Add a new (empty) line to the listbox.
        """
        self.master.data.vuv_lines[self.master.system].append(VuvLine(self.master.system))
        self.listbox.insert(tk.END, "unassigned")
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(tk.END)
        self.on_select(None)
    
    def remove_line(self):
        """Remove the currently-selected line from the listbox.
        """
        if self.idx is not None:
            self.master.data.vuv_lines[self.master.system].pop(self.idx)
            self.listbox.delete(self.idx)
            self.included_lines_box.delete(0, tk.END)
            self.lam_lb_box.delete(0, tk.END)
            self.lam_ub_box.delete(0, tk.END)
            # Don't clear the time boxes, since we will usually want the same
            # time window for baseline subtraction.
            # self.t_lb_box.delete(0, tk.END)
            # self.t_ub_box.delete(0, tk.END)
            self.idx = None
    
    def on_apply(self):
        """Apply the current settings and update the plot.
        """
        if self.idx is None:
            print("Please select a line to apply!")
            self.bell()
            return
        
        included_lines = re.findall(LIST_REGEX, self.included_lines_box.get())
        if len(included_lines) == 0:
            print("No lines to include!")
            self.bell()
            return
        try:
            included_lines = [int(l) for l in included_lines]
        except ValueError:
            print("Invalid entry in included lines!")
            self.bell()
            return
        
        try:
            lam_lb = float(self.lam_lb_box.get())
        except ValueError:
            print("Invalid lower bound for wavelength!")
            self.bell()
            return
        
        try:
            lam_ub = float(self.lam_ub_box.get())
        except ValueError:
            print("Invalid upper bound for wavelength!")
            self.bell()
            return
        
        try:
            t_lb = float(self.t_lb_box.get())
        except ValueError:
            print("Invalid baseline start!")
            self.bell()
            return
        
        try:
            t_ub = float(self.t_ub_box.get())
        except ValueError:
            print("Invalid baseline end!")
            self.bell()
            return
        
        xl = self.master.data.vuv_lines[self.master.system][self.idx]
        xl.diagnostic_lines = included_lines
        xl.lam_lb = lam_lb
        xl.lam_ub = lam_ub
        xl.t_lb = t_lb
        xl.t_ub = t_ub
        
        xl.process_bounds(self.master.data)
        
        self.listbox.delete(self.idx)
        self.listbox.insert(self.idx, ', '.join(map(str, included_lines)))
        
        remove_all(self.master.plot_frame.l_final)
        self.master.plot_frame.l_final = []
        self.master.plot_frame.l_final.append(
            self.master.plot_frame.a_final.plot(
                self.master.data.vuv_time[self.master.system],
                xl.signal,
                'k'
            )
        )
        self.master.plot_frame.canvas.draw()

class VuvLine(object):
    """Class to store information on a single VUV diagnostic line.
    
    The line may encapsulate more than one "diagnostic line" from the STRAHL
    output in case these lines overlap too much.
    
    Assumes you set the relevant attributes externally, then call
    :py:meth:`process_bounds`.
    
    Attributes
    ----------
    diagnostic_lines : list of int
        List of the indices of the lines included in the spectral region of the
        line.
    lam_lb : float
        Lower bound of wavelength to include (nm).
    lam_ub : float
        Upper bound of wavelength to include (nm).
    t_lb : float
        Lower bound of time to use for baseline subtraction.
    t_ub : float
        Upper bound of time to use for baseline subtraction.
    signal : array, (`N`,)
        The `N` timepoints of the combined, baseline-subtracted signal.
    """
    def __init__(self, system, diagnostic_lines=None, lam_lb=None, lam_ub=None, t_lb=None, t_ub=None):
        self.system = system
        self.diagnostic_lines = diagnostic_lines
        self.lam_lb = lam_lb
        self.lam_ub = lam_ub
        self.t_lb = t_lb
        self.t_ub = t_ub
        
        self.signal = None
        self.uncertainty = None
    
    def process_bounds(self, data):
        # Find the indices in the data:
        lam_lb_idx, lam_ub_idx = profiletools.get_nearest_idx(
            [self.lam_lb, self.lam_ub],
            data.vuv_lam[self.system]
        )
        t_lb_idx, t_ub_idx = profiletools.get_nearest_idx(
            [self.t_lb, self.t_ub],
            data.vuv_time[self.system]
        )
        
        # Form combined spectrum:
        # The indices are reversed for lambda vs. index:
        self.signal = data.vuv_signal[self.system][:, lam_ub_idx:lam_lb_idx + 1].sum(axis=1)
        
        # Perform the baseline subtraction:
        self.signal -= self.signal[t_lb_idx:t_ub_idx + 1].mean()
        
        # Compute the propagated uncertainty:
        self.uncertainty = (data.vuv_uncertainty[self.system][:, lam_ub_idx:lam_lb_idx + 1]**2).sum(axis=1)
        self.uncertainty += (self.uncertainty[t_lb_idx:t_ub_idx + 1]**2).sum() / (t_ub_idx - t_lb_idx + 1)**2
        self.uncertainty = scipy.sqrt(self.uncertainty)

class XtomoPlotFrame(tk.Frame):
    """Frame to hold the plot with the XTOMO time-series data.
    """
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.f = Figure()
        self.suptitle = self.f.suptitle('')
        self.a = self.f.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.f, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='NESW')
        
        # Need to put the toolbar in its own frame, since it automatically calls
        # pack on itself, but I am using grid.
        self.toolbar_frame = tk.Frame(self)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar_frame.grid(row=1, column=0, sticky='EW')
        
        self.canvas.mpl_connect(
            'button_press_event',
            lambda event: self.canvas._tkcanvas.focus_set()
        )
        self.canvas.mpl_connect('key_press_event', self.on_key_event)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.a.set_xlabel('$t$ [s]')
        self.a.set_ylabel('XTOMO signal [AU]')
        
        # Produce the plots here so all of the variables are in place:
        # self.l_raw, = self.a.plot(
        #     self.master.data.xtomo_t[self.master.system],
        #     self.master.data.xtomo_sig[self.master.system][0],
        #     'b',
        #     label='raw'
        # )
        # self.l_smoothed, = self.a.plot(
        #     self.master.data.xtomo_t[self.master.system],
        #     scipy.convolve(
        #         self.master.data.xtomo_sig[self.master.system][0],
        #         scipy.ones(10) / 10.0,
        #         mode='same'
        #     ),
        #     'g',
        #     label='smoothed'
        # )
        # Just put dummy values here, and call "apply" first thing:
        # self.l_bsub, = self.a.plot(
        #     self.master.data.xtomo_t[self.master.system],
        #     self.master.data.xtomo_sig[self.master.system][0],
        #     'm',
        #     label='baseline-subtracted'
        # )
        self.l_bsub_smoothed, = self.a.plot(
            self.master.data.xtomo_t[int(self.master.sys_state.get())][::100],
            scipy.convolve(
                self.master.data.xtomo_sig[int(self.master.sys_state.get())][0],
                scipy.ones(10) / 10.0,
                mode='same'
            )[::100],
            'k',
            label='baseline-subtracted, smoothed'
        )
        self.l_inj_time = self.a.axvline(
            self.master.data.settings.injections[0].t_inj,
            color='r',
            label='injection time'
        )
        self.span_inj_window = self.a.axvspan(
            self.master.data.settings.injections[0].t_start,
            self.master.data.settings.injections[0].t_stop,
            color='r',
            alpha=0.2
        )
        
        self.a.legend(loc='best')
    
    def on_key_event(self, evt):
        """Tie keys to the toolbar.
        """
        key_press_handler(evt, self.canvas, self.toolbar)

class XtomoWindow(tk.Tk):
    """GUI to set bad channels and baseline subtraction ranges for XTOMO data.
    """
    def __init__(self, data):
        print(
            "Enter baseline subtraction ranges as '(lb1, ub1), (lb2, ub2)'. "
            "Press enter to apply the baseline subtraction and boxcar smoothing. "
            "Use the right/left arrows to change channels and the up/down arrows "
            "to change systems. The baseline subtraction is the same for all "
            "channels/systems across a given injection. "
            "Close the window when done to continue the analysis."
        )
        tk.Tk.__init__(self)
        
        self.data = data
        # self.system = system
        
        self.current_inj = 0
        
        self.wm_title("XTOMO inspector")
        
        # Set these up first, since self.plot_frame needs them:
        self.sys_s = [
            str(k) for k in self.data.xtomo_sig.keys()
            if self.data.xtomo_sig[k] is not None
        ]
        self.sys_state = tk.StringVar(self)
        self.sys_state.set(self.sys_s[0])
        
        # Now set everything else up in sequence:
        self.plot_frame = XtomoPlotFrame(self)
        self.plot_frame.grid(row=0, column=0, sticky='NESW', rowspan=8)
        
        self.sys_label = tk.Label(self, text='system:')
        self.sys_label.grid(row=0, column=1, sticky='SE')
        
        self.sys_menu = tk.OptionMenu(
            self,
            self.sys_state,
            *self.sys_s,
            command=self.update_sys
        )
        self.sys_menu.grid(row=0, column=2, sticky='SW')
        
        self.chan_label = tk.Label(self, text='channel:')
        self.chan_label.grid(row=1, column=1, sticky='SE')
        
        self.chan_state = tk.StringVar(self)
        self.chan_state.set("0")
        # Put the trace on the variable not the menu so we can change the
        # options later:
        self.chan_state.trace('w', self.update_channel)
        
        self.channel_s = [str(v) for v in range(0, self.data.xtomo_sig[int(self.sys_state.get())].shape[0])]
        self.chan_menu = tk.OptionMenu(
            self,
            self.chan_state,
            *self.channel_s
        )
        self.chan_menu.grid(row=1, column=2, sticky='SW')
        
        self.inj_label = tk.Label(self, text='injection:')
        self.inj_label.grid(row=2, column=1, sticky='NSE')
        
        self.inj_state = tk.StringVar(self)
        self.inj_state.set("0")
        
        self.inj_s = [str(v) for v in range(0, len(self.data.settings.injections))]
        self.inj_menu = tk.OptionMenu(
            self,
            self.inj_state,
            *self.inj_s,
            command=self.update_inj
        )
        self.inj_menu.grid(row=2, column=2, sticky='NSW')
        
        self.bad_state = tk.IntVar(self)
        self.bad_state.set(
            int(not self.data.xtomo_channel_mask[int(self.sys_state.get())][int(self.chan_state.get())])
        )
        self.bad_check = tk.Checkbutton(
            self,
            text="bad channel",
            variable=self.bad_state,
            command=self.toggle_bad
        )
        self.bad_check.grid(row=3, column=1, sticky='NW', columnspan=2)
        
        self.boxcar_label = tk.Label(self, text='boxcar points:')
        self.boxcar_label.grid(row=4, column=1, sticky='NE')
        
        self.boxcar_spin = tk.Spinbox(
            self,
            from_=1,
            to=100001,
            command=self.apply,
            increment=2.0
        )
        self.boxcar_spin.grid(row=4, column=2, sticky='NW')
        
        self.baseline_ranges_label = tk.Label(
            self,
            text="baseline subtraction range(s):"
        )
        self.baseline_ranges_label.grid(row=5, column=1, sticky='NW', columnspan=2)
        
        self.baseline_ranges_box = tk.Entry(self)
        self.baseline_ranges_box.grid(row=6, column=1, sticky='NEW', columnspan=2)
        self.baseline_ranges_box.delete(0, tk.END)
        self.baseline_ranges_box.insert(
            0,
            str(self.data.xtomo_baseline_ranges[int(self.inj_state.get())])[1:-1]
        )
        
        self.apply_button = tk.Button(self, text="apply", command=self.apply)
        self.apply_button.grid(row=7, column=1, sticky='NW', columnspan=2)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.bind("<Left>", self.on_arrow)
        self.bind("<Right>", self.on_arrow)
        self.bind("<Up>", self.on_arrow)
        self.bind("<Down>", self.on_arrow)
        self.bind("<Return>", self.apply)
        self.bind("<KP_Enter>", self.apply)
        
        self.apply(plot_inj_time=True)
    
    def destroy(self):
        self.apply(plot_bsub=False, plot_inj_time=False)
        tk.Tk.destroy(self)
        # TODO: Sometimes this hangs when matplotlib windows are still open. For
        # now I will just instruct the user to close them, but a more permanent
        # solution would be nice.
        print(
            "XtomoWindow.destroy complete. You may need to close all open "
            "matplotlib windows if the terminal is hanging."
        )
    
    def update_sys(self, new_idx=None):
        """Update the system displayed, keeping the injection constant.
        
        The channel will revert to the first channel of the new system.
        
        This function should:
            - Reset the options in the channel menu to reflect the new system.
            - Replot the data.
            - Retrieve the correct value for the "bad channel" check box and set
              it accordingly.
        """
        self.channel_s = [str(v) for v in range(0, self.data.xtomo_sig[int(self.sys_state.get())].shape[0])]
        self.chan_menu['menu'].delete(0, tk.END)
        for c in self.channel_s:
            self.chan_menu['menu'].add_command(
                label=c,
                command=tk._setit(self.chan_state, c)
            )
        self.chan_state.set(self.channel_s[0])
    
    def update_channel(self, *args):
        """Update the channel displayed, keeping the injection constant.
        
        This function should:
            - Replot the data (modifying in place if possible). This means four
              curves to plot:
                - Raw data
                - Smoothed data
                - Baseline-subtracted data
                - Smoothed baseline-subtracted data
            - Retrieve the correct value for the "bad channel" check box and set
              it accordingly.
        """
        self.apply(plot_inj_time=False)
        self.bad_state.set(
            not self.data.xtomo_channel_mask[int(self.sys_state.get())][int(self.chan_state.get())]
        )
    
    def update_inj(self, new_idx=None):
        """Update the injection displayed, keeping the channel constant.
        
        This function should:
            - Store the baseline subtraction range(s) for the previous
              injection.
            - Load the baseline subtraction range(s) for the new injection.
            - Update the plot with the new baseline subtraction range. This
              means two curves to plot:
                - Baseline-subtracted data
                - Smoothed baseline-subtracted data
            - Update the vertical bar on the plot to show the injection
              location.
        """
        self.data.xtomo_baseline_ranges[self.current_inj] = eval(
            '[' + self.baseline_ranges_box.get() + ']'
        )
        self.current_inj = int(self.inj_state.get())
        self.baseline_ranges_box.delete(0, tk.END)
        self.baseline_ranges_box.insert(
            0,
            str(self.data.xtomo_baseline_ranges[int(self.inj_state.get())])[1:-1]
        )
        
        self.apply(plot_inj_time=True)
    
    def toggle_bad(self):
        """Update the flagging of a bad channel.
        
        This function should set the state to the appropriate value. By doing
        this as soon as the button is clicked, we avoid having to handle
        anything with this when changing state.
        """
        self.data.xtomo_channel_mask[int(self.sys_state.get())][int(self.chan_state.get())] = not bool(self.bad_state.get())
    
    def apply(self, evt=None, plot_bsub=True, plot_inj_time=False):
        """Apply the selected boxcar smoothing and baseline subtraction.
        
        This function should:
            - Store the baseline subtraction range(s) for the current injection.
            - Update the plot with the new baseline subtraction range and boxcar
              smoothing. This means two curves to plot:
                - Baseline-subtracted data
                - Smoothed baseline-subtracted data
        """
        print("Applying settings...")
        
        self.data.xtomo_baseline_ranges[int(self.inj_state.get())] = eval(
            '[' + self.baseline_ranges_box.get() + ']'
        )
        
        # if plot_raw:
        #     self.plot_frame.l_raw.set_ydata(
        #         self.data.xtomo_sig[int(self.sys_state.get())][int(self.chan_state.get())]
        #     )
        #     self.plot_frame.l_smoothed.set_ydata(
        #         scipy.convolve(
        #             self.data.xtomo_sig[int(self.sys_state.get())][int(self.chan_state.get())],
        #             scipy.ones(int(self.boxcar_spin.get())) / float(self.boxcar_spin.get()),
        #             mode='same'
        #         )
        #     )
        
        if plot_bsub:
            # TODO: This will need to be pulled out into a separate function!
            bsub_idxs = []
            for r in self.data.xtomo_baseline_ranges[int(self.inj_state.get())]:
                lb_idx, ub_idx = profiletools.get_nearest_idx(
                    r,
                    self.data.xtomo_t[int(self.sys_state.get())]
                )
                bsub_idxs.extend(range(lb_idx, ub_idx + 1))
            # Reduce to just the unique values:
            bsub_idxs = list(set(bsub_idxs))
            bsub = scipy.mean(self.data.xtomo_sig[int(self.sys_state.get())][int(self.chan_state.get())][bsub_idxs])
            
            # self.plot_frame.l_bsub.set_ydata(
            #     self.data.xtomo_sig[int(self.sys_state.get())][int(self.chan_state.get())] - bsub
            # )
            self.plot_frame.l_bsub_smoothed.set_ydata(
                scipy.convolve(
                    self.data.xtomo_sig[int(self.sys_state.get())][int(self.chan_state.get())],
                    scipy.ones(int(self.boxcar_spin.get())) / float(self.boxcar_spin.get()),
                    mode='same'
                )[::100] - bsub
            )
        
        if plot_inj_time:
            self.plot_frame.l_inj_time.set_xdata(
                [self.data.settings.injections[int(self.inj_state.get())].t_inj,] * 2
            )
            xy = self.plot_frame.span_inj_window.get_xy()
            xy[[0, 1, 4], 0] = self.data.settings.injections[int(self.inj_state.get())].t_start
            xy[[2, 3], 0] = self.data.settings.injections[int(self.inj_state.get())].t_stop
            self.plot_frame.span_inj_window.set_xy(xy)
        
        if plot_bsub or plot_inj_time:
            self.plot_frame.a.relim()
            self.plot_frame.a.autoscale_view()
            self.plot_frame.f.canvas.draw()
        
        print("done!")
    
    def on_arrow(self, evt):
        """Handle arrow key events by updating the relevant slider.
        
        This function should:
            - Use right/left arrows to change channels
            - Use up/down arrows to change system
        """
        if evt.keysym == 'Right':
            if int(self.chan_state.get()) < int(self.channel_s[-1]):
                self.chan_state.set(str(int(self.chan_state.get()) + 1))
                # self.update_channel()
            else:
                self.bell()
        elif evt.keysym == 'Left':
            if int(self.chan_state.get()) > int(self.channel_s[0]):
                self.chan_state.set(str(int(self.chan_state.get()) - 1))
                # self.update_channel()
            else:
                self.bell()
        # TODO: This is hard-coded to assume we only ever use (1, 3, 5). This
        # should be fixed.
        elif evt.keysym == 'Up':
            if int(self.sys_state.get()) < int(self.sys_s[-1]):
                self.sys_state.set(str(int(self.sys_state.get()) + 2))
                self.update_sys()
            else:
                self.bell()
        elif evt.keysym == 'Down':
            if int(self.sys_state.get()) > int(self.sys_s[0]):
                self.sys_state.set(str(int(self.sys_state.get()) - 2))
                self.update_sys()
            else:
                self.bell()

def read_ADF15(path, debug_plots=[], order=1):
    """Read photon emissivity coefficients from an ADF15 file.
    
    Returns a dictionary whose keys are the wavelengths of the lines in
    angstroms. The value is an interp2d instance that will evaluate the PEC at
    a desired dens, temp.
    
    Parameter `order` lets you change the order of interpolation -- use 1
    (linear) to speed things up, higher values for more accuracy.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    header = lines.pop(0)
    # Get the expected number of lines by reading the header:
    num_lines = int(header.split()[0])
    pec_dict = {}
    for i in xrange(0, num_lines):
        # Get the wavelength, number of densities and number of temperatures
        # from the first line of the entry:
        l = lines.pop(0)
        header = l.split()
        try:
            lam = float(header[0])
        except ValueError:
            # These lines appear to occur when lam has more digits than the
            # allowed field width. We don't care about these lines, so we will
            # just ditch them.
            warnings.warn(
                "Bad line, ISEL=%d, lam=%s" % (i + 1, header[0]),
                RuntimeWarning
            )
            lam = None
        if 'excit' not in l.lower():
            warnings.warn(
                "Throwing out non-excitation line, ISEL=%d, lam=%s" % (i + 1, header[0]),
                RuntimeWarning
            )
            lam = None
        num_den = int(header[2])
        num_temp = int(header[3])
        # Get the densities:
        dens = []
        while len(dens) < num_den:
            dens += [float(v) for v in lines.pop(0).split()]
        dens = scipy.asarray(dens)
        # Get the temperatures:
        temp = []
        while len(temp) < num_temp:
            temp += [float(v) for v in lines.pop(0).split()]
        temp = scipy.asarray(temp)
        # Get the PEC's:
        PEC = []
        while len(PEC) < num_den:
            PEC.append([])
            while len(PEC[-1]) < num_temp:
                PEC[-1] += [float(v) for v in lines.pop(0).split()]
        PEC = scipy.asarray(PEC)
        if lam is not None:
            if lam not in pec_dict:
                pec_dict[lam] = []
            pec_dict[lam].append(
                scipy.interpolate.RectBivariateSpline(
                    scipy.log10(dens),
                    scipy.log10(temp),
                    PEC,
                    kx=order,
                    ky=order
                )
            )
            # {'dens': dens, 'temp': temp, 'PEC': PEC}
            if lam in debug_plots:
                ne_eval = scipy.linspace(dens.min(), dens.max(), 100)
                Te_eval = scipy.linspace(temp.min(), temp.max(), 100)
                NE, TE = scipy.meshgrid(ne_eval, Te_eval)
                PEC_eval = pec_dict[lam][-1].ev(scipy.log10(NE), scipy.log10(TE))
                f = plt.figure()
                a = f.add_subplot(111, projection='3d')
                # a.set_xscale('log')
                # a.set_yscale('log')
                a.plot_surface(NE, TE, PEC_eval, alpha=0.5)
                DENS, TEMP = scipy.meshgrid(dens, temp)
                a.scatter(DENS.ravel(), TEMP.ravel(), PEC.T.ravel(), color='r')
                a.set_xlabel('$n_e$ [cm$^{-3}$]')
                a.set_ylabel('$T_e$ [eV]')
                a.set_zlabel('PEC')
                f.suptitle(str(lam))
    
    return pec_dict

def read_atomdat(path):
    """Read the Ca.atomdat file to get out the diagnostic lines specification.
    
    Returns ordered arrays of the charge state, center wavelength (in angstroms)
    and half-width of the window to use (in angstroms).
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    l = lines.pop(0)
    while l[0:2] != 'cd':
        l = lines.pop(0)
    # ditch the specification:
    lines.pop(0)
    # empty line:
    lines.pop(0)
    # header for number of lines:
    lines.pop(0)
    # Now get the number of lines:
    num_lines = int(lines.pop(0).strip())
    # empty line:
    lines.pop(0)
    # Header:
    lines.pop(0)
    # Now read the line specifications:
    charges = []
    CWL = []
    HW = []
    for i in xrange(0, num_lines):
        data = lines.pop(0).split()
        charges.append(float(data[0]))
        CWL.append(float(data[1]))
        HW.append(float(data[2]))
    
    return (charges, CWL, HW)

def compute_emiss(pec_dict, cw, hw, ne, nZ, Te, no_ne=False):
    """Compute the emission summed over all lines in a given window.
    
    This is very approximate -- it just adds up the photons per second for the
    included lines as computed directly from the PECs.
    
    Parameters
    ----------
    pec_dict : dictionary
        The photon emission coefficient dictionary as returned by
        :py:func:`read_ADF15` for the desired charge state.
    cw : array of float
        The center wavelengths of the bins to use, in angstroms.
    hw : array of float
        The half-widths of the bins to use, in angstroms.
    ne : array of float
        The electron density on the grid, in cm^3.
    nZ : array of float
        The density of the selected charge state on the grid, in cm^3.
    Te : array of float
        The electron temperature on the grid, in eV.
    no_ne : bool, optional
        If True, the PEC is taken to not depend on density. Default is False.
    """
    lb = cw - hw
    ub = cw + hw
    wl = scipy.asarray(pec_dict.keys())
    included = wl[(wl >= lb) & (wl <= ub)]
    emiss = scipy.zeros_like(ne)
    for lam in included:
        # Need to loop over all lines having the same lam:
        for p in pec_dict[lam]:
            if no_ne:
                emiss += 1.986449e-15 / lam * ne * nZ * p(scipy.log10(Te))
            else:
                emiss += 1.986449e-15 / lam * ne * nZ * p.ev(
                    scipy.log10(ne),
                    scipy.log10(Te)
                )
    # Sometimes there are interpolation issues with the PECs:
    emiss[emiss < 0] = 0.0
    return emiss

def flush_blobs(sampler, burn):
    """Zeros out all blobs up to (but not including) the one at burn.
    """
    for i_step in xrange(0, burn):
        b_chains = sampler.blobs[i_step]
        for i_chain in xrange(0, len(b_chains)):
            b_chains[i_chain] = (-scipy.inf, None, None, None, 'cleared')

class _InterpBrightWrapper(object):
    def __init__(self, t, num_s, num_v):
        self.t = t
        self.num_s = num_s
        self.num_v = num_v
    
    def __call__(self, params):
        s, v, t_, dt_s, dt_v = params
        sbright_interp = scipy.zeros((len(self.t), self.num_s))
        vbright_interp = scipy.zeros((len(self.t), self.num_v))
        
        postinj_s = (self.t >= dt_s)
        for j in xrange(0, s.shape[1]):
            sbright_interp[postinj_s, j] = scipy.interpolate.InterpolatedUnivariateSpline(
                t_ + dt_s,
                s[:, j]
            )(self.t[postinj_s])
        postinj_v = (self.t >= dt_v)
        for j in xrange(0, v.shape[1]):
            vbright_interp[postinj_v, j] = scipy.interpolate.InterpolatedUnivariateSpline(
                t_ + dt_v,
                v[:, j]
            )(self.t[postinj_v])
        
        return (sbright_interp, vbright_interp)

class HirexVuvFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
        self.f = Figure()
        self.suptitle = self.f.suptitle("")
        
        self.canvas = FigureCanvasTkAgg(self.f, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='NESW')
        
        dum, self.a_H, self.a_V = self.master.r.run_data.plot_data(f=self.f)
        
        # Make dummy lines to modify data in:
        self.l_H = []
        self.l_V = []
        for k, a in enumerate(self.a_H):
            l, = a.plot(
                self.master.time_vec - self.master.r.time_1,
                scipy.zeros_like(self.master.time_vec)
            )
            self.l_H.append(l)
        for k, a in enumerate(self.a_V):
            l, = a.plot(
                self.master.time_vec - self.master.r.time_1,
                scipy.zeros_like(self.master.time_vec)
            )
            self.l_V.append(l)
        
        # Need to put the toolbar in its own frame, since it automatically calls
        # pack on itself, but I am using grid.
        self.toolbar_frame = tk.Frame(self)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar_frame.grid(row=1, column=0, sticky='EW')
        
        self.canvas.mpl_connect(
            'button_press_event',
            lambda event: self.canvas._tkcanvas.focus_set()
        )
        self.canvas.mpl_connect('key_press_event', self.on_key_event)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
    
    def on_key_event(self, evt):
        """Tie keys to the toolbar.
        """
        key_press_handler(evt, self.canvas, self.toolbar)

class DVFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
        self.f = Figure(figsize=(2, 2))
        
        self.a_D = self.f.add_subplot(2, 1, 1)
        self.a_D.set_title('$D$ [m$^2$/s]')
        self.a_V = self.f.add_subplot(2, 1, 2)
        self.a_V.set_title('$V$ [m/s]')
        self.a_V.set_xlabel('$r/a$')
        
        self.l_D, = self.a_D.plot(
            self.master.r.roa_grid_DV,
            scipy.zeros_like(self.master.r.roa_grid_DV)
        )
        self.l_V, = self.a_V.plot(
            self.master.r.roa_grid_DV,
            scipy.zeros_like(self.master.r.roa_grid_DV)
        )
        
        self.canvas = FigureCanvasTkAgg(self.f, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='NESW')
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

class ParameterFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
        self.labels = []
        self.boxes = []
        
        row = 0
        
        for l, b in zip(self.master.r.get_labels(), self.master.r.get_prior().bounds[:]):
            self.labels.append(tk.Label(self, text=l.translate(None, '$\\')))
            self.labels[-1].grid(row=row, column=0, sticky='NSE')
            self.boxes.append(
                tk.Spinbox(
                    self,
                    from_=b[0],
                    to=b[1],
                    command=self.master.apply,
                    increment=max((b[1] - b[0]) / 100.0, 0.0001),
                )
            )
            self.boxes[-1].grid(row=row, column=1, sticky='NSW')
            row += 1

class BoxcarFrame(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
        self.boxcar_label = tk.Label(self, text='XTOMO boxcar points:')
        self.boxcar_label.grid(row=0, column=0, sticky='NSE')
        
        self.boxcar_spin = tk.Spinbox(
            self,
            from_=1,
            to=100001,
            command=self.master.apply,
            increment=2.0
        )
        self.boxcar_spin.grid(row=0, column=1, sticky='NSW')

class XTOMOExplorerPlotFrame(tk.Frame):
    def __init__(self, system, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        
        self.system = system
        
        self.f = Figure()
        self.suptitle = self.f.suptitle("")
        
        self.canvas = FigureCanvasTkAgg(self.f, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='NESW')
        
        dum, self.a_X = self.master.master.r.run_data.plot_xtomo(
            self.system,
            norm=True,
            f=self.f,
            boxcar=1001
        )
        
        # Make dummy lines to modify data in:
        self.l_X = []
        for k, a in enumerate(self.a_X):
            l, = a.plot(
                self.master.master.time_vec - self.master.master.r.time_1,
                scipy.zeros_like(self.master.master.time_vec),
                'g'
            )
            self.l_X.append(l)
        
        # Need to put the toolbar in its own frame, since it automatically calls
        # pack on itself, but I am using grid.
        self.toolbar_frame = tk.Frame(self)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.toolbar_frame.grid(row=1, column=0, sticky='EW')
        
        self.canvas.mpl_connect(
            'button_press_event',
            lambda event: self.canvas._tkcanvas.focus_set()
        )
        self.canvas.mpl_connect('key_press_event', self.on_key_event)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
    
    def on_key_event(self, evt):
        """Tie keys to the toolbar.
        """
        key_press_handler(evt, self.canvas, self.toolbar)

class XTOMOExplorerWindow(tk.Toplevel):
    def __init__(self, *args, **kwargs):
        tk.Toplevel.__init__(self, *args, **kwargs)
        
        self.XTOMO_frames = [
            XTOMOExplorerPlotFrame(k, self) for k in self.master.r.run_data.xtomo_sig.keys()
            if self.master.r.run_data.xtomo_sig[k] is not None
        ]
        for k, f in enumerate(self.XTOMO_frames):
            f.grid(row=0, column=k, sticky='NESW')
        
        self.grid_rowconfigure(0, weight=1)
        for k in range(0, len(self.XTOMO_frames)):
            self.grid_columnconfigure(k, weight=1)

class ParameterExplorer(tk.Tk):
    def __init__(self, r):
        tk.Tk.__init__(self)
        self.r = r
        
        # Do a dummy STRAHL run to figure out the length of the time vector:
        params = self.r.get_prior().random_draw()
        cs_den, sqrtpsinorm, time, ne, Te = self.r.DV2cs_den(params)
        self.time_vec = time
        
        self.wm_title("Parameter Explorer")
        
        self.hirex_vuv_frame = HirexVuvFrame(self)
        self.hirex_vuv_frame.grid(row=0, column=0, sticky='NESW', rowspan=3)
        
        self.DV_frame = DVFrame(self)
        self.DV_frame.grid(row=0, column=1, sticky='NESW')
        
        self.parameter_frame = ParameterFrame(self)
        self.parameter_frame.grid(row=1, column=1, sticky='NESW')
        
        # self.boxcar_frame = BoxcarFrame(self)
        # self.boxcar_frame.grid(row=2, column=1, sticky='NESW')
        
        self.apply_button = tk.Button(self, text='apply', command=self.apply)
        self.apply_button.grid(row=2, column=1, sticky='NESW')
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.bind("<Return>", self.apply)
        self.bind("<KP_Enter>", self.apply)
        
        self.XTOMO_window = XTOMOExplorerWindow(self)
        
        
    def apply(self, evt=None):
        print("begin apply...")
        params = [float(b.get()) for b in self.parameter_frame.boxes]
        D, V = self.r.eval_DV(params)
        self.DV_frame.l_D.set_ydata(D)
        self.DV_frame.l_V.set_ydata(V)
        self.DV_frame.a_D.relim()
        self.DV_frame.a_V.relim()
        self.DV_frame.a_D.autoscale_view()
        self.DV_frame.a_V.autoscale_view()
        self.DV_frame.f.canvas.draw()
        try:
            cs_den, sqrtpsinorm, time, ne, Te = self.r.DV2cs_den(params)
        except TypeError:
            print('fail!')
            return
        dlines = self.r.cs_den2dlines(params, cs_den, sqrtpsinorm, time, ne, Te)
        sbright, vbright, xtomobright = self.r.dlines2sig(params, dlines, time)
        lp = self.r.sig2ln_prob(params, sbright, vbright, xtomobright, time)
        self.hirex_vuv_frame.suptitle.set_text("%.3e" % (lp,))
        
        eig_D, eig_V, knots_D, knots_V, hp_D, hp_mu_D, hp_V, param_scaling, param_source = self.r.split_params(params)
        
        time = time - self.r.time_1
        time_s = time + param_source[0]
        time_v = time + param_source[1]
        time_xtomo = time + param_source[2]
        
        for k, l in enumerate(self.hirex_vuv_frame.l_H):
            l.set_ydata(sbright[:, k])
            l.set_xdata(time_s)
        for k, l in enumerate(self.hirex_vuv_frame.l_V):
            l.set_ydata(vbright[:, k])
            l.set_xdata(time_v)
        for frame in self.XTOMO_window.XTOMO_frames:
            for k, l in enumerate(frame.l_X):
                l.set_ydata(xtomobright[frame.system][:, k])
                l.set_xdata(time_xtomo)
            frame.f.canvas.draw()
        self.hirex_vuv_frame.f.canvas.draw()
        print("apply done!")
