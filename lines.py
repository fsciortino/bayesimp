from __future__ import division

import scipy
import scipy.io
from scipy.constants import h, c, e
import scipy.special
import scipy.interpolate
import periodictable
import re
import warnings
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.widgets as mplw
import matplotlib.gridspec as mplgs
import cPickle as pkl

# Regular expression to match the header used for each line:
HEADER_PATTERN = r'^[ \t]*([0-9]+)[ \t]+([0-9]+)[ \t]+([0-9.]+)[ \t]+([0-9]+)[ \t]*(.*)\n$'

# Mapping between sin*.dat file names and atomic number:
SIN_NAME_MAP = {
    'sinar.dat': 18,
    'sinti.dat': 22,
    'sinsc.dat': 21,
    'sinca.dat': 20,
    'sinfl.dat': 9,
    'sinnit.dat': 7
}

# List of H-like line types:
H_LIKE_TYPES = [9, 17, 18, 19, 20, 21, 22, 23, 24]

element_names = [el.name for el in periodictable.elements]

# NOTE: For Z != 7, num_params[9] should only be 7, and the array should be
# padded with NaN.
num_params = [None, 7, 12, 2, 6, 4, 1, 6, 18, 8, 2, 12, 6, 9, 7, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9]

class Line(object):
    """Class to keep track of the parameters of a given spectral line.
    
    In addition to storing the parameters below, the energy of the line `E` is
    stored in keV.
    
    Parameters
    ----------
    Z : int
        The nuclear charge of the atom.
    q : int
        The charge state of the ion.
    lam : float
        The wavelength of the line, in nm.
    data_type : int
        The type of line the parameters `p` correspond to.
    comment : str
        The comment from the data file.
    p : array of float, optional
        The parameters of the line. The length of this array is set by
        `data_type`. If not present, the array is initialized with all zeros.
    """
    def __init__(self, Z, q, lam, data_type, comment, p=None):
        self.Z = Z
        self.q = q
        self.lam = lam
        if lam == 0:
            self.E = scipy.nan
        else:
            self.E = h * c / (lam * 1e-9 * e * 1e3)
        self.data_type = data_type
        self.comment = comment
        if p is None:
            self.p = scipy.zeros(num_params[data_type])
        else:
            self.p = scipy.asarray(p, dtype=float)
    
    def __repr__(self):
        """Make exact representation of the :py:class:`Line` instance.
        """
        return "Line({Z:d}, {q:d}, {lam:f}, {data_type:d}, '{comment:s}', p={p:s})".format(
            Z=self.Z,
            q=self.q,
            lam=self.lam,
            data_type=self.data_type,
            comment=self.comment,
            p=self.p.__repr__()
        )
    
    def __str__(self):
        """Make human-readable representation of the :py:class:`Line` instance.
        """
        return (
            "Line with:\n"
            "\tZ={Z:d}\n"
            "\tq={q:d}\n"
            "\tlambda={lam:.3f}nm\n"
            "\tE={E:.3f}keV\n"
            "\ttype={data_type:d}\n"
            "\tcomment={comment:s}\n"
            "\tp={p}".format(
                Z=self.Z,
                q=self.q,
                lam=self.lam,
                E=self.E,
                data_type=self.data_type,
                comment=self.comment,
                p=self.p
            )
        )

def read_atdata(path='atdata.dat'):
    """Read the atomic data used for computing spectral lines.
    
    Reads the data from the file originally from
    /usr/local/cmod/codes/spectroscopy/mist/atdata.dat.
    
    Returns a dictionary where the keys are the atomic numbers and the values
    are lists of :py:class:`Line` instances, one for each line of the element.
    
    In line with the previous IDL code, lines.pro, this program uses only the
    first entry for every element and ignores the extra lines at the bottom of
    the file.
    
    Parameters
    ----------
    path : str, optional
        The path to the file to read the atomic data from. Default is
        'atdata.dat' (i.e., in the local directory).
    """
    atdata = {}
    # Get all of the data into memory:
    with open(path, 'r') as f:
        lines = f.readlines()
    # Find where the data start:
    i = 0
    while i < len(lines) - 1:
        while i < len(lines) and lines[i].strip().lower() not in element_names:
            i += 1
        if i == len(lines):
            break
        # Parse all of the lines as they appear in the file:
        Z = periodictable.elements.name(lines[i].strip().lower()).number
        i = i + 1
        if Z in atdata:
            warnings.warn(
                "{els:s} (Z={el:d}) already read! Ignoring second set of data at "
                "i={i:d}!".format(els=periodictable.elements[Z].name, el=Z, i=i),
                RuntimeWarning
            )
            continue
        atdata[Z] = []
        n_lines = int(lines[i].strip())
        for n in range(n_lines):
            # Consume any blank/comment lines, just in case:
            while lines[i].split()[0] != str(Z):
                i += 1
            # Parse the header line:
            hdr = re.split(HEADER_PATTERN, lines[i])
            assert int(hdr[1]) == Z
            assert len(hdr) == 7
            q = int(hdr[2]) - 1
            lam = float(hdr[3]) / 10.0
            data_type = int(hdr[4])
            comment = hdr[5].strip()
            atdata[Z].append(Line(Z, q, lam, data_type, comment))
            i += 1
            # j is the counter for how many parameters have been read:
            j = 0
            if data_type == 9 and Z != 7:
                np = 7
                atdata[Z][-1].p[-1] = scipy.nan
            else:
                np = num_params[data_type]
            while j < np:
                p = [float(s) for s in lines[i].split()]
                atdata[Z][-1].p[j:j + len(p)] = p
                i += 1
                j += len(p)
    return atdata

def read_sindat(sindir=''):
    """Reads all of the files matching `sin*.dat` in directory `root`.
    
    Returns a dictionary where the keys are the atomic numbers and the values
    are dictionaries containing the data. The keys of the data dictionaries are:
    'sq', 'sr', 'ss', 'st' and 'terange'.
    
    These appear to determine the Te-dependence of the satellite lines.
    
    Parameters
    ----------
    sindir : str, optional
        The directory to look for the `sin*.dat` files in. Default is to look in
        the local directory.
    """
    sin_files = glob.glob(os.path.join(sindir, 'sin*.dat'))
    sin_dat = {}
    for p in sin_files:
        try:
            # This structure is necessary since the sin*.dat files don't use the
            # standard atomic symbols.
            Z = SIN_NAME_MAP[os.path.basename(p)]
        except KeyError:
            warnings.warn(
                "Unknown sin*.dat file '{p:s}', skipped!".format(p=p),
                RuntimeWarning
            )
            continue
        name = re.split(r'sin(.+)\.dat', p)[1]
        f = scipy.io.readsav(p)
        sin_dat[Z] = {
            'sq': scipy.asarray(f['sq' + name], dtype=float),
            'sr': scipy.asarray(f['sr' + name], dtype=float),
            'ss': scipy.asarray(f['ss' + name], dtype=float),
            'st': scipy.asarray(f['st' + name], dtype=float),
            'terange': scipy.asarray(f['terange'], dtype=float)
        }
    return sin_dat

def ZYFF(Te, EIJ):
    """Computes `ZY` and `FF`, used in other functions.
    
    If `EIJ` is a scalar, the output has the same shape as `Te`. If `EIJ` is an
    array, the output has shape `EIJ.shape` + `Te.shape`. This should keep the
    output broadcastable with `Te`.
    
    Parameters
    ----------
    Te : array of float
        Electron temperature. Shape is arbitrary.
    EIJ : scalar float or array of float
        Energy difference.
    """
    # Expand the dimensions of EIJ to produce the desired output shape:
    Te = scipy.asarray(Te, dtype=float)
    EIJ = scipy.asarray(EIJ, dtype=float)
    for n in xrange(Te.ndim):
        EIJ = scipy.expand_dims(EIJ, axis=-1)
    
    ZY = EIJ / (1e3 * Te)
    
    FF = scipy.zeros_like(ZY)
    mask = (ZY >= 1.5)
    FF[mask] = scipy.log((ZY[mask] + 1) / ZY[mask]) - (0.36 + 0.03 * scipy.sqrt(ZY[mask] + 0.01)) / (ZY[mask] + 1)**2
    mask = ~mask
    FF[mask] = scipy.log((ZY[mask] + 1) / ZY[mask]) - (0.36 + 0.03 / scipy.sqrt(ZY[mask] + 0.01)) / (ZY[mask] + 1)**2
    
    return ZY, FF

def SSS(Te, p):
    """Computes S values for the six He-like lines (W, X, Y, Z, F, O).
    
    Parameters
    ----------
    Te : array
        Electron temperature. Shape can be arbitrary.
    p : array, (6,)
        Parameters to use.
    """
    EIJ, A, B, C, D, E = p
    ZY, FF = ZYFF(Te, EIJ)
    OMEG = A + (B * ZY - C * ZY**2 + D * ZY**3 + E) * FF + (C + D) * ZY - D * ZY**2
    
    return 8e-8 * OMEG * 10.2 / EIJ * scipy.exp(-ZY) / scipy.sqrt(1e3 * Te)

def ALPHAZ(Te, EIJ, S2, S3, S6, S4):
    """Computes alpha for the Z line.
    
    Parameters
    ----------
    Te : array of float
        Electron temperature. Shape must match the S* parameters, but can
        otherwise be arbitrary.
    EIJ : scalar float
        Transition energy (0th parameter from the line data).
    S2, S3, S6, S4 : arrays of float
        S values for various lines.
    """
    S4 = S4.copy()
    S4[S4 <= 0.0] = 1e-30
    return 1.0 + 0.4 * (S2 + S3 + S6) / S4 * scipy.exp(-0.213 * EIJ / (1e3 * Te))

def SSSDPR(Te, Z, E1, E2, A, B):
    """Computes collisional coupling (into state).
    
    Parameters
    ----------
    Te : array of float
        Electron temperature. Shape can be arbitrary.
    Z : scalar int
        Nuclear charge.
    E1, E2, A, B : scalar floats
        Parameters from atdata.dat.
    """
    ZY, FF = ZYFF(Te, E2 - E1)
    return 9.28e-7 / (Z - 1.0)**2 / scipy.sqrt(1e3 * Te) * (A + B * FF) * scipy.exp(-ZY)

def SSSDPRO(Te, C, E1, E2, SSS):
    """Computes collisional coupling (out of state).
    
    Parameters
    ----------
    Te : array of float
        Electron temperature. Shape can be arbitrary.
    C, E1, E2 : scalar floats
        Parameters from atdata.dat.
    SSS : array of float
        Intermediate result, must match shape of `Te`.
    """
    return C * SSS * scipy.exp((E2 - E1) / (1e3 * Te))

def SSSLI(Te, CHI, C):
    """Computes the inner shell ionization of Li-like.
    
    Parameters
    ----------
    Te : array of float
        Electron temperature. Shape can be arbitrary.
    CHI, C : scalar float
        Parameters.
    """
    ZY, FF = ZYFF(Te, CHI)
    return 3e-6 * C * scipy.exp(-ZY) * FF / ZY / (1e3 * Te)**1.5

def RADREC(Te, Z, p):
    """Computes the radiative recombination term.
    
    Parameters
    ----------
    Te : array of float
        Electron temperature. Shape can be arbitrary.
    Z : scalar int
        Nuclear charge.
    p : array of float, (6,)
        Parameters to use.
    """
    C1, C2, C3, C4, C5, C6 = p
    return 1e-14 * (
        C1 * (Z - 1)**C2 / (1e3 * Te)**C3 + C4 * (Z - 1)**C5 / (1e3 * Te)**C6
    )

def compute_SXR(cs_den, ne, Te, atdata, sindat, filter_trans, PEC, E_thresh=1.0, **kwargs):
    """Compute the soft x-ray emission due to the calcium injection.
    
    Assumes Z=20. Returns an array with shape (`n_time`, `n_space`).
    
    Parameters
    ----------
    cs_den : array of float, (`n_time`, Z+1, `n_space`)
        The charge state densities (as returned by STRAHL). Units are cm^-3.
    ne : array of float, (`n_space`,) or (`n_time`, `n_space`)
        The electron densities, either a stationary profile, or profiles as a
        function of time. Units are cm^-3.
    Te : array of float, (`n_space`,) or (`n_time`, `n_space`)
        The electron temperatures, either a stationary profile, or profiles as a
        function of time. Units are keV.
    atdata : dict
        The atomic physics data, as read by :py:func:`read_atdata`.
    sindat : dict
        Data from the sin*.dat files.
    filter_trans : array of float, (`n_lines`,)
        Filter transmissions for each of the lines that will be returned by
        :py:func:`compute_lines`.
    PEC : dict, optional
        Dictionary of photon emissivity coefficients as collected by
        :py:class:`~bayesimp.Run`. This should have keys which are the charge
        states and values which are dictionaries which have keys which are
        wavelengths and values which are interp1d objects.
    E_thresh : float, optional
        The energy threshold below which lines are thrown out. Default is 1.0 keV.
        FOR NOW THIS IS ONLY APPLIED TO THE PEC CALCUATION.
    **kwargs : extra arguments, optional
        Extra arguments are passed to :py:func:`compute_lines`.
    """
    if 'He_source' not in kwargs:
        kwargs['He_source'] = 'PEC'
    
    em, E = compute_lines(
        20,
        cs_den,
        ne,
        Te,
        atdata=atdata,
        sindat=sindat,
        PEC=PEC,
        E_thresh=E_thresh,
        full_return=False,
        **kwargs
    )
    # Convert to number of photons actually getting through filter, then to
    # W/cm^3, then sum into total power:
    em = (em * filter_trans[None, :, None] * E[None, :, None] * e * 1e3).sum(axis=1)
    
    # Sometimes there are numerical issues:
    em[em < 0.0] = 0.0
    
    return em

def compute_lines(
        Z,
        cs_den,
        ne,
        Te,
        atdata=None,
        path=None,
        sindat=None,
        sindir=None,
        PEC=None,
        E_thresh=0.0,
        He_source='PEC',
        full_return=False
    ):
    """Compute the spectral lines, using the same algorithm as the IDL program lines.pro.
    
    The output `em` is an array of shape (`n_time`, `n_lines`, `n_space`).
    
    Returns a tuple of (`em`, `lam`, `E`, `q`, `comment`), where the variables
    are as follows:
    * `em` is the emissivities in photons/s/cm^3, and has shape
      (`n_time`, `n_lines`, `n_space`).
    * `lam` is the wavelengths of each line in nm, and has shape (`n_lines`,).
    * `E` is the energy of each line in keV, and has shape (`n_lines`,).
    * `q` is the charge state of each line, and has shape (`n_lines`,).
    * `comment` is the comment from each line, and has shape (`n_lines`,).
    
    Parameters
    ----------
    Z : int
        The atomic number of the element to compute the lines for.
    cs_den : array, (`n_time`, `n_cs`, `n_space`)
        The charge state densities (as returned by STRAHL). Units are cm^-3.
    ne : array, (`n_space`,) or (`n_time`, `n_space`)
        The electron densities, either a stationary profile, or profiles as a
        function of time. Units are cm^-3.
    Te : array, (`n_space`,) or (`n_time`, `n_space`)
        The electron temperatures, either a stationary profile, or profiles as a
        function of time. Units are keV.
    atdata : dict, optional
        The atomic physics data, as read by :py:func:`read_atdata`. If `None`,
        the atomic physics data are read from `path`. Default is `None` (read
        from file).
    path : str, optional
        The path to read the atomics physics data from, only used if `atdata` is
        `None`. If `None`, the default path defined in :py:func:`read_atdata` is
        used. Default is `None` (use :py:func:`read_atdata` default).
    sindat : dict, optional
        Data from the sin*.dat files. If `None`, the data are read from `root`
        using :py:func:`read_sindat`. Default is `None` (read from files).
    sindir : str, optional
        The directory to look for the `sin*.dat` files in. If `None`, the
        default path defined in :py:func:`read_sindat` is used. Default is `None`
        (use :py:func:`read_sindat` default).
    PEC : dict, optional
        Dictionary of photon emissivity coefficients as collected by
        :py:class:`~bayesimp.Run`. This should have keys which are the charge
        states and values which are dictionaries which have keys which are
        wavelengths and values which are interp1d objects.
    E_thresh : float, optional
        The energy threshold below which lines are thrown out. Default is 0.0.
        FOR NOW THIS IS ONLY APPLIED TO THE PEC CALCUATION.
    He_source : {'PEC', 'lines', 'both'}, optional
        Source for data on He-like line emission. Can come from the ADAS ADF15
        PEC files ('PEC'), the approach used in lines.pro ('lines'), or both
        sets of lines can be included ('both'). Default is to use both sets of
        lines.
    full_output : bool, optional
        If True, all values described above are returned. If False, only `em`
        and `E` are returned. Default is False (only return `em`, `E`).
    """
    # Convert the densities to cm^-3:
    # cs_den = cs_den * 1e-6
    # ne = ne * 1e-6
    
    # Make sure the atomic physics data are loaded:
    if atdata is None:
        if path is None:
            atdata = read_atdata()
        else:
            atdata = read_atdata(path=path)
    try:
        atdata = atdata[Z]
    except KeyError:
        raise ValueError("No atomic physics data for Z={Z:d}!".format(Z=Z))
    # Get the additional data from the sin*.dat files:
    if sindat is None:
        if sindir is None:
            sindat = read_sindat()
        else:
            sindat = read_sindat(sindir=sindir)
    try:
        sindat = sindat[Z]
    except KeyError:
        sindat = None
    
    # Pull out the He-, Li- and H-like charge states:
    n_H = cs_den[:, -2, :]
    if Z > 1:
        n_He = cs_den[:, -3, :]
    if Z > 2:
        n_Li = cs_den[:, -4, :]
    
    # Figure out the shape of the PEC results so we don't have to do so many
    # concatenations:
    PEC_len = 0
    for qv, P in PEC.iteritems():
        for lv, pv in P.iteritems():
            for pec_obj in pv:
                E_val = h * c / (lv / 10.0 * 1e-9 * e * 1e3)
                if E_val >= E_thresh:
                    PEC_len += 1
    
    em = scipy.zeros((cs_den.shape[0], len(atdata) + PEC_len, cs_den.shape[2]))
    
    # Set up the return values:
    lam = [l.lam for l in atdata]
    E = [l.E for l in atdata]
    q = [l.q for l in atdata]
    comment = [l.comment for l in atdata]
    
    # The He-like lines need to be handled specially, since they all enter into
    # the calculation of each other:
    # This approach lets the He-like lines appear anywhere in the sequence in
    # atdata.dat, as long as they are ordered.
    if He_source in ('lines', 'both'):
        line_types = scipy.asarray([ld.data_type for ld in atdata])
        He_like_lines, = scipy.where(line_types == 8)
        # Enforce the condition that the lines be in order in the file:
        He_like_lines.sort()
        if len(He_like_lines) > 0:
            S1 = SSS(Te, atdata[He_like_lines[0]].p[0:6])
            S2 = SSS(Te, atdata[He_like_lines[1]].p[0:6])
            S3 = SSS(Te, atdata[He_like_lines[2]].p[0:6])
            S4 = SSS(Te, atdata[He_like_lines[3]].p[0:6])
            S5 = SSS(Te, atdata[He_like_lines[4]].p[0:6])
            S6 = SSS(Te, atdata[He_like_lines[5]].p[0:6])
            
            SPR1 = S1 * atdata[He_like_lines[0]].p[6]
            SPR2 = S2 * atdata[He_like_lines[1]].p[6]
            SPR3 = S3 * atdata[He_like_lines[2]].p[6]
            SPR4 = S4 * ALPHAZ(Te, atdata[He_like_lines[3]].p[0], S2, S3, S6, S4)
            SPR5 = S5 * atdata[He_like_lines[4]].p[6]
            SPR6 = S6 * atdata[He_like_lines[5]].p[6]
            
            SMP1P = SSSDPR(
                Te,
                Z,
                atdata[He_like_lines[4]].p[0],
                atdata[He_like_lines[0]].p[0],
                atdata[He_like_lines[0]].p[7],
                atdata[He_like_lines[0]].p[8]
            )
            SM2 = SSSDPR(
                Te,
                Z,
                atdata[He_like_lines[3]].p[0],
                atdata[He_like_lines[1]].p[0],
                atdata[He_like_lines[1]].p[7],
                atdata[He_like_lines[1]].p[8]
            )
            SM1 = SSSDPR(
                Te,
                Z,
                atdata[He_like_lines[3]].p[0],
                atdata[He_like_lines[2]].p[0],
                atdata[He_like_lines[2]].p[7],
                atdata[He_like_lines[2]].p[8]
            )
            SM0 = SSSDPR(
                Te,
                Z,
                atdata[He_like_lines[3]].p[0],
                atdata[He_like_lines[5]].p[0],
                atdata[He_like_lines[5]].p[7],
                atdata[He_like_lines[5]].p[8]
            )
            
            S1PMP = SSSDPRO(
                Te,
                0.333,
                atdata[He_like_lines[4]].p[0],
                atdata[He_like_lines[0]].p[0],
                SMP1P
            )
            S2M = SSSDPRO(
                Te,
                0.6,
                atdata[He_like_lines[3]].p[0],
                atdata[He_like_lines[1]].p[0],
                SM2
            )
            S1M = SSSDPRO(
                Te,
                1.0,
                atdata[He_like_lines[3]].p[0],
                atdata[He_like_lines[2]].p[0],
                SM1
            )
            S0M = SSSDPRO(
                Te,
                3.0,
                atdata[He_like_lines[3]].p[0],
                atdata[He_like_lines[5]].p[0],
                SM0
            )
            
            SLIF = SSSLI(Te, atdata[He_like_lines[4]].p[9], 0.5)
            SLIZ = SSSLI(Te, atdata[He_like_lines[3]].p[9], 1.5)
            
            ALPHRRW = RADREC(Te, Z, atdata[He_like_lines[0]].p[10:16])
            ALPHRRX = RADREC(Te, Z, atdata[He_like_lines[1]].p[10:16])
            ALPHRRY = RADREC(Te, Z, atdata[He_like_lines[2]].p[10:16])
            ALPHRRZ = RADREC(Te, Z, atdata[He_like_lines[3]].p[10:16])
            ALPHRRF = RADREC(Te, Z, atdata[He_like_lines[4]].p[10:16])
            ALPHRRO = RADREC(Te, Z, atdata[He_like_lines[5]].p[10:16])
            
            T1DR = scipy.exp(-6.80 * (Z + 0.5)**2 / (1e3 * Te))
            T2DR = scipy.exp(-8.77 * Z**2 / (1e3 * Te))
            T3DR = scipy.exp(-10.2 * Z**2 / (1e3 * Te))
            T0DR = 5.17e-14 * Z**4 / (1e3 * Te)**1.5
            
            C1 = 12.0 / (1.0 + 6.0e-6 * Z**4)
            C2 = 18.0 / (1.0 + 3.0e-5 * Z**4)
            C3 = 69.0 / (1.0 + 5.0e-3 * Z**3)
            ALPHDRW = T0DR * (C1 * T1DR + C2 * T2DR + C3 * T3DR)
            
            C1 = 1.9
            C2 = 54.0 / (1.0 + 1.9e-4 * Z**4)
            C3 = (
                380.0 / (1.0 + 5.0e-3 * Z**3) * 2.0 * (Z - 1)**0.6 /
                (1e3 * Te)**0.3 / (1.0 + 2.0 * (Z - 1)**0.6 / (1e3 * Te)**0.3)
            )
            ALPHDRX = T0DR * 5.0 / 9.0 * (C1 * T1DR + C2 * T2DR + C3 * T3DR)
            ALPHDRY = T0DR * 3.0 / 9.0 * (C1 * T1DR + C2 * T2DR + C3 * T3DR)
            ALPHDRO = T0DR * 1.0 / 9.0 * (C1 * T1DR + C2 * T2DR + C3 * T3DR)
            
            C1 = 3.0 / (1.0 + 3.0e-6 * Z**4)
            C2 = 0.5 / (1.0 + 2.2e-5 * Z**4)
            C3 = 6.3 / (1.0 + 5.0e-3 * Z**3)
            ALPHDRF = T0DR * (C1 * T1DR + C2 * T2DR + C3 * T3DR)
            
            C1 = 9.0 / (1.0 + 7.0e-5 * Z**4)
            C2 = 27.0 / (1.0 + 8.0e-5 * Z**4)
            C3 = 380.0 / (1.0 + 5.0e-3 * Z**3) / (1.0 + 2.0 * (Z - 1)**0.6 / (1e3 * Te)**0.3)
            ALPHDRZ = T0DR * (C1 * T1DR + C2 * T2DR + C3 * T3DR)
            
            ALPHW = ALPHRRW + ALPHDRW
            ALPHX = ALPHRRX + ALPHDRX
            ALPHY = ALPHRRY + ALPHDRY
            ALPHZ = ALPHRRZ + ALPHDRZ
            ALPHF = ALPHRRF + ALPHDRF
            ALPHO = ALPHRRO + ALPHDRO
            
            # NOTE: This only computes the W, X, Y, Z lines, even though we in
            # principle have data for F and O, too. I suspect this is because the
            # other lines are treated as negligible, but it still seems to be a
            # strange oversight.
            
            # Calculation for W line:
            NA1 = (n_Li * SLIF + n_He * SPR5 + n_H * ALPHF) / (atdata[He_like_lines[4]].p[16] + ne * SMP1P)
            NA2 = (n_He * SPR1 + n_H * ALPHW) / (ne * SMP1P)
            NA3 = (atdata[He_like_lines[0]].p[16] + ne * S1PMP) / (ne * SMP1P)
            NA4 = (atdata[He_like_lines[0]].p[17] + ne * S1PMP) / (atdata[He_like_lines[4]].p[16] + ne * SMP1P)
            NW = atdata[He_like_lines[0]].p[16] * ne * (NA1 + NA2) / (NA3 - NA4)
            em[:, He_like_lines[0], :] = NW
            
            # Calculation for Z line:
            NA1 = n_Li * SLIZ
            NA2 = n_He * (
                SPR4 + SPR6 + SPR3 / (
                    1.0 + atdata[He_like_lines[2]].p[16] / (
                        atdata[He_like_lines[2]].p[17] + ne * S1M
                    )
                )
            )
            NA3 = n_He * SPR2 / (
                1.0 + atdata[He_like_lines[1]].p[16] / (
                    atdata[He_like_lines[1]].p[17] + ne * S2M
                )
            )
            NA4 = n_H * (
                ALPHZ + ALPHO + ALPHY / (
                    1.0 + atdata[He_like_lines[2]].p[16] / (
                        atdata[He_like_lines[2]].p[17] + ne * S1M
                    )
                )
            )
            NA5 = n_H * ALPHX / (
                1.0 + atdata[He_like_lines[1]].p[16] / (
                    atdata[He_like_lines[1]].p[17] + ne * S2M
                )
            )
            NA6 = ne / atdata[He_like_lines[3]].p[16] * SM2 / (
                1.0 + (
                    atdata[He_like_lines[1]].p[17] + ne * S2M
                ) / atdata[He_like_lines[1]].p[16]
            )
            NA7 = ne / atdata[He_like_lines[3]].p[16] * SM1 / (
                1.0 + (
                    atdata[He_like_lines[2]].p[17] + ne * S1M
                ) / atdata[He_like_lines[2]].p[16]
            )
            NZ = ne * (NA1 + NA2 + NA3 + NA4 + NA5) / (1.0 + NA6 + NA7)
            em[:, He_like_lines[3], :] = NZ
            
            # Calculation for X line:
            NA1 = n_He * SPR2 + n_H * ALPHX
            NA2 = 1.0 + (
                atdata[He_like_lines[1]].p[17] + ne * S2M
            ) / atdata[He_like_lines[1]].p[16]
            NA3 = ne * SM2 / NA2
            NX = ne * NA1 / NA2 + NA3 * NZ / atdata[He_like_lines[3]].p[16]
            em[:, He_like_lines[1], :] = NX
            
            # Calculation for Y line:
            NA1 = n_He * SPR3 + n_H * ALPHY
            NA2 = 1.0 + (atdata[He_like_lines[2]].p[17] + ne * S1M) / atdata[He_like_lines[2]].p[16]
            NA3 = ne * SM1 / NA2
            NY = ne * NA1 / NA2 + NA3 * NZ / atdata[He_like_lines[3]].p[16]
            em[:, He_like_lines[2], :] = NY
    
    for line_idx, line_data in zip(range(len(atdata)), atdata):
        if line_data.data_type == 8:
            # This case was handled above, but this saves having to filter it
            # out at the start of the loop.
            pass
        elif line_data.data_type == 9:
            # This case is handled in a far more complicated manner in
            # lines.pro, but I am going to use a far simpler approximation for
            # the time being. This will probably need to change to get F right.
            ZY, FF = ZYFF(Te, line_data.p[0])
            gg = line_data.p[3] + (
                line_data.p[4] * ZY - line_data.p[5] * ZY**2 + line_data.p[6]
            ) * FF + line_data.p[5] * ZY
            # This assumes ne is in cm^-3...
            SHY = 1.58e-5 * 1.03 / scipy.sqrt(1e3 * Te) / line_data.p[0] * line_data.p[1] * gg * scipy.exp(-ZY)
            em[:, line_idx, :] = SHY * n_H * ne
        elif line_data.data_type == 10 and He_source in ('lines', 'both'):
            # Handle He-like/H-like satellites:
            tfact = scipy.exp(-line_data.p[0] / (1e3 * Te)) / (1e3 * Te)**1.5 * ne
            # This does the same thing as the if-statements in lines.pro.
            v = Z - line_data.q - 1
            if v in (0, 1, 2):
                em[:, line_idx, :] = 1.65e-9 * tfact * line_data.p[1] * cs_den[:, -v - 2, :]
            else:
                warnings.warn("Unknown satellite lines, skipping!", RuntimeWarning)
            # This handles all of the Z cases for which there are only q, r, s,
            # t lines present. The others are not implemented at this time.
            if (sindat is not None) and (line_data.q == (Z - 2)):
                if line_data.comment in ('/q line', '/r line', '/s line', '/t line'):
                    try:
                        em[:, line_idx, :] += 10**(
                            scipy.interpolate.InterpolatedUnivariateSpline(
                                sindat['terange'],
                                scipy.log10(sindat['s' + line_data.comment[1]])
                            )(1e3 * Te)
                        ) * ne * cs_den[:, Z - 3, :]
                    except KeyError:
                        pass
        # None of these line types are in the spectral range I need, so it isn't
        # worth the time to implement them at present.
        elif line_data.data_type == 11:
            # TODO!
            pass
        elif line_data.data_type == 12:
            # TODO!
            pass
        elif line_data.data_type == 13:
            # TODO!
            pass
        elif line_data.data_type == 14:
            # TODO!
            pass
        elif line_data.data_type == 15:
            # TODO!
            pass
        elif line_data.data_type == 16:
            # TODO!
            pass
        else:
            warnings.warn(
                "Unsupported line type {lt:d}, skipping line and leaving set to "
                "zero!".format(lt=line_data.data_type),
                RuntimeWarning
            )
    
    if PEC is not None and He_source in ('PEC', 'both'):
        k = len(atdata)
        for qv, P in PEC.iteritems():
            for lv, pv in P.iteritems():
                for pec_obj in pv:
                    E_val = h * c / (lv / 10.0 * 1e-9 * e * 1e3)
                    if E_val >= E_thresh:
                        lam = lam + [lv / 10.0,]
                        E = E + [E_val,]
                        q = q + [qv,]
                        comment = comment + [str(lv),]
                        # em has shape [ntimes, nlines, nspace]
                        em[:, k, :] = ne * cs_den[:, qv, :] * pec_obj.ev(
                            scipy.log10(ne),
                            scipy.log10(1e3 * Te)
                        )
                        k += 1
                        # em = scipy.concatenate((em, emv[:, None, :]), axis=1)
    
    # Convert to ph/s/m^3:
    # em = em * 1e6
    
    # This simplifies later calculations:
    E = scipy.asarray(E)
    
    if full_return:
        return em, lam, E, q, comment
    else:
        return em, E

def fij_sum(nu, nl):
    """Compute the summed oscillator strength for the given transition.
    
    This applies to the hydrogen-like isoelectronic sequence. `nl` is the lower
    level, `nu` is the upper level. The oscillator strengths have been summed
    over all angular momenta.
    
    This uses the analytic expression given in Menzel and Pekeris MNRAS 1935.
    
    This is probably not useful for our application, since it includes
    presumably non-radiative transitions. (Why do these have non-zero oscillator
    strengths?)
    
    Parameters
    ----------
    nu : int
        Upper level. Must satisfy `nu` > `nl`.
    nl : int
        Lower level.
    """
    # if nl >= nu:
    #     raise ValueError("Lower level cannot be higher than upper level!")
    nl = nl * 1.0
    nu = nu * 1.0
    return (
        32.0 / 3.0 * nu**4.0 * nl**2.0 *
        (nu - nl)**(2.0 * nu + 2.0 * nl - 4.0) /
        (nu + nl)**(2.0 * nu + 2.0 * nl + 3.0) *
        (
            (
                scipy.special.hyp2f1(
                    -1.0 * nu + 1.0,
                    -1.0 * nl,
                    1.0,
                    -4.0 * nu * nl / (nu - nl)**2.0
                )
            )**2.0 -
            (
                scipy.special.hyp2f1(
                    -1.0 * nl + 1.0,
                    -1.0 * nu,
                    1.0,
                    -4.0 * nu * nl / (nu - nl)**2.0
                )
            )**2.0
        )
    )

def plot_lines(em, E, q, comment, t, r):
    """Make a plot of emissivity as a function of energy with each line labelled.
    
    Provides sliders to select space and time point displayed.
    
    Parameters
    ----------
    em : array of float, (`n_time`, `n_lines`, `n_space`)
        Emissivities to plot, in photons/s/cm^3.
    E : array of float, (`n_lines`,)
        Energies of the lines, in keV.
    q : array of int, (`n_lines`,)
        Charge states of the lines.
    comments : array of str, (`n_lines`,)
        Comments describing each line, used for formulate the labels.
    t : array of float, (`n_time`,)
        Time points, in s.
    r : array of float, (`n_space`,)
        Space points, in r/a.
    """
    f = plt.figure()
    gs = mplgs.GridSpec(2, 2, height_ratios=[8, 1])
    a = f.add_subplot(gs[0, :])
    a_s_space = f.add_subplot(gs[1, 0])
    a_s_time = f.add_subplot(gs[1, 1])
    
    title = f.suptitle("")
    
    def update(val):
        """Update the slice shown.
        """
        i_space = int(s_space.val)
        i_time = int(s_time.val)
        title.set_text(
            "r={space:.2f}m, t={time:.2f}".format(
                space=r[i_space],
                time=t[i_time]
            )
        )
        
        a.clear()
        a.set_xlabel("$E$ [keV]")
        a.set_ylabel(r"$\epsilon$ [photons/s/cm$^3$]")
        
        for em_v, E_v, q_v, c_v in zip(em[i_time, :, i_space], E, q, comment):
            a.plot([E_v, E_v], [0, em_v])
            a.text(E_v, em_v, "$q={q:d}$, ${c:s}$".format(q=q_v, c=c_v))
        
        f.canvas.draw()
    
    s_space = mplw.Slider(
        a_s_space,
        'space index',
        0,
        len(r) - 1,
        valinit=0,
        valfmt='%d'
    )
    s_space.on_changed(update)
    
    s_time = mplw.Slider(
        a_s_time,
        'time index',
        0,
        len(t) - 1,
        valinit=0,
        valfmt='%d'
    )
    s_time.on_changed(update)
    
    update(0)
    
    def arrow_respond(evt):
        """Respond to arrow keys.
        """
        if evt.key == 'right':
            s_time.set_val(min(s_time.val + 1, s_time.valmax))
        elif evt.key == 'left':
            s_time.set_val(max(s_time.val - 1, s_time.valmin))
        elif evt.key == 'up':
            s_space.set_val(min(s_space.val + 1, s_space.valmax))
        elif evt.key == 'down':
            s_space.set_val(max(s_space.val - 1, s_space.valmin))
    
    f.canvas.mpl_connect('key_press_event', arrow_respond)
    
    return f

def test(path='lines_test_dat.sav', PEC=None, E_thresh=1.0, He_source='PEC'):
    sf = scipy.io.readsav(path)
    r = scipy.asarray(sf['rad'], dtype=float) / 100.0
    q_IDL = scipy.asarray(sf['emhead'][:, 1], dtype=int) - 1
    lam_IDL = scipy.asarray(sf['emhead'][:, 2], dtype=float) / 10.0
    E_IDL = h * c / (lam_IDL * 1e-9 * e * 1e3)
    ne = scipy.asarray(sf['nelec'], dtype=float) # already in cm^-3
    Te = scipy.asarray(sf['te'], dtype=float) # already in keV
    em_IDL = scipy.swapaxes(scipy.asarray(sf['lemist'], dtype=float), 1, 2) * 1e6 # convert to ph/s/m^3
    cs_den = scipy.swapaxes(scipy.asarray(sf['denst'], dtype=float), 1, 2) # already in cm^-3
    
    atdata_IDL = read_atdata(path='atdata.dat.original')[20]
    comment_IDL = [l.comment for l in atdata_IDL]
    # comment_IDL = [str(typ) for typ in sf['emhead'][:, 3]]
    
    em, lam, E, q, comment = compute_lines(
        20,
        cs_den,
        ne,
        Te,
        PEC=PEC,
        E_thresh=E_thresh,
        He_source=He_source,
        full_return=True
    )
    t = range(0, em.shape[0])
    
    plot_lines(em, E, q, comment, t, r)
    
    plot_lines(em_IDL, E_IDL, q_IDL, comment_IDL, t, r)
    
    atdata = read_atdata()
    sindat = read_sindat()
    filter_trans = read_filter_file('Be_filter_50_um.dat')(E)
    with open('Be_50_um_{src:s}.pkl'.format(src=He_source), 'wb') as f:
        pkl.dump(filter_trans, f)
    emiss = compute_SXR(cs_den, ne, Te, atdata, sindat, filter_trans, PEC, He_source=He_source)
    
    f = plt.figure()
    a_ne = f.add_subplot(3, 1, 1)
    a_ne.plot(r, ne)
    a_ne.set_ylabel('$n_e$ [cm^-3]')
    a_Te = f.add_subplot(3, 1, 2)
    a_Te.plot(r, Te)
    a_Te.set_ylabel('$T_e$ [keV]')
    a_emiss = f.add_subplot(3, 1, 3)
    a_emiss.plot(r, emiss[0, :])
    a_emiss.set_ylabel('$\epsilon$ [W/m^3]')
    a_emiss.set_xlabel('$r$ [m]')
    
    return em, E, q, comment, emiss

def read_filter_file(path, plot=False, title=None, figsize=None):
    """Read a filter file, optionally plotting the transmission curve.
    
    The file should have 2 header rows and be formatted with the energy (in eV)
    in the first column and the transmission in the second column. The file
    should be whitespace-delimited. This is the format used by
    http://henke.lbl.gov/optical_constants/filter2.html
    
    Returns a :py:class:`scipy.interpolate.InterpolatedUnivariateSpline`
    instance which takes as an argument the energy in keV and returns the
    transmission.
    
    Parameters
    ----------
    path : str
        The path to the filter file.
    plot : bool, optional
        If True, the filter curve will be plotted. Default is False (do not plot
        the filter curve).
    """
    E, T = scipy.loadtxt(path, skiprows=2, unpack=True)
    E = E / 1e3
    if plot:
        f = plt.figure(figsize=figsize)
        a = f.add_subplot(1, 1, 1)
        a.plot(E, T)
        a.set_xlabel("$E$ [keV]")
        a.set_ylabel("transmission, $T$")
        a.set_title(path if title is None else title)
    return scipy.interpolate.InterpolatedUnivariateSpline(E, T, ext='const')
        