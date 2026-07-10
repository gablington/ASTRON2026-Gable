import numpy as np
import pypulse as pyp
from numpy.polynomial.legendre import legval
from glob import glob
import pypulse.utils as u
from matplotlib import pyplot as plt
import seaborn as sns
import astropy.units as units
from tqdm import tqdm
from IFD_class import IFD
from LIFD_class import LIFD

# # Compatibility shim for pypulse with newer NumPy versions.
# # pypulse still calls np.trapz(), which was removed in recent NumPy releases.
# if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
#     np.trapz = np.trapezoid

# # pypulse still calls np.in1d(), which was removed in newer NumPy releases.
# # np.isin() is the supported replacement.
# if not hasattr(np, "in1d") and hasattr(np, "isin"):
#     np.in1d = np.isin



def get_FD_delay(FD_params, freqs_GHz):
    """
    Returns a function that provides the timing delays as a function of observing frequency
    """
    FD_params = FD_params[::-1]  # We need to invert the order because np.polyval uses p[0]*x**(N-1)+p[1]*x**(N-2)+...
    FD_params = np.concatenate((FD_params, [0]))  # https://numpy.org/doc/stable/reference/generated/numpy.polyval.html
    FD_func = lambda nu: 1e6 * np.polyval(FD_params, np.log(nu))  # Function that gives the timing delay
    FD_delays = FD_func(freqs_GHz)  * units.us  # nu in GHz, returns values in microseconds
    return FD_delays

def get_IFD_values(coeffs, lambdas_ns):
    '''
    coeffs:     coefficients of the IFD polynomial
    lambdas:    inverse frequencies, in GHz^-1
    '''

    return np.polynomial.polynomial.polyval(x=lambdas_ns, c=coeffs) * units.second

def calculate_snr(files, template_profile):
    """Calculate the average S/N for a given DM value"""
    snr_values = np.zeros(len(files))
    for j, file in tqdm(enumerate(files)):

        ar = pyp.Archive(file, prepare=False, verbose=False)
        ar.dedisperse(wcfreq=True)
        ar.pscrunch()
        ar.center()
        ar.fscrunch()
        ar.tscrunch()

        if np.any(np.isnan(ar.getData())):  # If the pulse is empty
            snr_values[j] = 0.0
        else:
            snr_values[j] = ar.fitPulses(template_profile, nums=[5])[0]

    return snr_values


def load_template_profile(psr_name, template_dir):
    """Load the template profile for a given pulsar."""
    template_file = glob(f"{template_dir}{psr_name}.*.GUPPI.15y.x.sum.sm")[0]
    return pyp.Archive(template_file).getSinglePulses()


def main():
    """Main execution function for DM optimization using golden section search."""
    # Set up parameters and files
    psr_name: str = "B1937+21"

    ar_files = glob(f"../DM_calculations/{psr_name}/ff_files/*.ff")  # FF files

    # We need to find which of the FF files has the maximum S/N
    if psr_name == "B1937+21":
        max_snr: int = 180
    elif psr_name == "J1012+5307":
        max_snr: int = 215
    elif psr_name == "J1022+1001":
        max_snr: int = 6
    elif psr_name == "J1744-1134":
        max_snr: int = 114
    elif psr_name == "J1713+0747":
        max_snr: int = 278
    elif psr_name == "J2145-0750":
        max_snr: int = 128
    elif psr_name == "J1909-3744":
        max_snr: int = 354
    elif psr_name == "J1918-0642":
        max_snr: int = 110
    elif psr_name == "J1643-1224":
        max_snr: int = 20
    elif psr_name == "J1024-0719":
        max_snr: int = 109
    elif psr_name == "J2302+4442":
        max_snr: int = 116
    else:
        max_snr = None

    sns.set_style("ticks")
    sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.5})

    fig, axs = plt.subplots(nrows=2, ncols=1, height_ratios=[1, 3], figsize=(8, 10), sharex=True, gridspec_kw={'wspace':0, 'hspace':0})
    colors = plt.colormaps['Paired'](range(8))[[3, 1, 7]]
#    axs[0].set_title(f"{psr_name}")
    axs[0].set_ylabel("Intensity")

    # Load the template
    template_dir = "../NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/template/"
    template_profile = load_template_profile(psr_name, template_dir)

    # If we haven't figured out which FF file has the maximum S/N yet, then calculate it
    if max_snr is None:
        snr_values = calculate_snr(ar_files, template_profile)   # Calculate the S/N values of all FF files
        print("Maximum S/N file = ", np.argmax(snr_values), " with S/N = ", np.max(snr_values), "")
        max_snr = np.argmax(snr_values)                          # Save the index of the FF file with the maximum S/N

    # Load and prepare the FF file
    ar = pyp.Archive(ar_files[max_snr], prepare=False, verbose=False)
    nbins = ar.getNbin()
    ar.dedisperse(wcfreq=True)
    ar.pscrunch()
    ar.tscrunch()
    ar.center()

    # Convert the period to microseconds
    period_us = (ar.getPeriod() * units.s).to(units.us)

    # Make the waterfall plot
#    axs[1].set_xlabel("Pulse Phase [bins]")
    axs[1].set_xlabel("Time ($\mu$s)")
    unit = u.unitchanger(ar.getFrequencyUnit())
    axs[1].set_ylabel("Frequency (%s)" % unit)
    ax2 = axs[1].twinx()

    # Plot the waterfall
    freq_Mhz = ar.getAxis('F', edges=False) * units.MHz
    data = ar.getData(setnan=0.0)

    # Make sure that freq_Mhz[0] is always the highest frequency and freq_Mhz[-1] is always the lowest.
    if freq_Mhz[0] < freq_Mhz[-1]:
        data = np.flipud(data)
        freq_Mhz = freq_Mhz[::-1]

    # extent[2] is the bottom of the axis, extent[3] is the top
    extent = [0, period_us.value, freq_Mhz[0].value, freq_Mhz[-1].value]
    freq_hz = freq_Mhz.to(units.Hz)
    freq_Ghz = freq_Mhz.to(units.GHz)

    y2min, y2max = 0, ar.getNchan()
#    x2min, x2max = 0, ar.getNsubint()
    axs[1].imshow(data, origin='lower', interpolation='nearest', aspect='auto', cmap='gray', extent=extent, zorder=0    )

    ax2.set_ylim(y2min, y2max)
    ax2.set_ylabel("Channel Number")

    # FD curve
    FD_coeffs = np.load(f"../results/{psr_name}/FD_coeffs.npy")
    FD_delays = get_FD_delay(FD_coeffs, freq_Ghz.value)

    # IFD curve
    IFD_coeffs = np.load(f"../results/{psr_name}/IFD_coeffs.npy")
    lambdas_ns = IFD.get_lambda_ns_from_freq(freq_Ghz)  # Inverse frequencies
    IFD_delays = get_IFD_values(IFD_coeffs, lambdas_ns).to(units.us)

    # LIFD curve
    lambdas = LIFD.get_lambda_sec_from_freq(freq_hz)  # Inverse frequencies
    lmin = lambdas.min()  # lambda_min
    lmax = lambdas.max()  # lambda_max
    x_vals = np.sort(LIFD.map_lambda_to_unit(lambdas, lmin, lmax))  # fixed mapping set in setup()

    # Get the delays calculated by the LIFD polynomial
    LIFD_coeffs = np.load(f"../results/{psr_name}/LIFD_coeffs.npy")
    LIFD_delays = legval(x_vals, LIFD_coeffs) * 1e6 * units.us         # Delays in microseconds

    mid_idx = len(freq_Mhz) // 2

    FD_delays_middle = FD_delays + (period_us / 2.0 - FD_delays[mid_idx])
    IFD_delays_middle = IFD_delays + (period_us / 2.0 - IFD_delays[mid_idx])
    LIFD_delays_middle = LIFD_delays + (period_us / 2.0 - LIFD_delays[mid_idx])

    if psr_name == "B1937+21":
        window_factor: int = 8
        IFD_delays_middle = IFD_delays_middle[::-1]
        LIFD_delays_middle = LIFD_delays_middle[::-1]
    elif psr_name == "J1012+5307":
        window_factor: int = 2
        IFD_delays_middle = IFD_delays_middle[::-1]
    elif psr_name == "J1022+1001":
        window_factor: int = 8
        FD_delays_middle = FD_delays_middle[::-1]
    elif psr_name == "J1024-0719":
        window_factor: int = 8
    elif psr_name == "J1643-1224":
        window_factor: int = 8
    elif psr_name == "J2145-0750":
        window_factor: int = 32


    axs[1].plot(FD_delays_middle, freq_Mhz, color=colors[2], lw=4, zorder=2, label="FD delay")
    axs[1].plot(IFD_delays_middle, freq_Mhz, color=colors[1], lw=4, zorder=2, label="IFD delay")
    axs[1].plot(LIFD_delays_middle, freq_Mhz, color=colors[0], lw=4, zorder=2, label="LIFD delay")

    # Just some plotting details
    axs[1].legend(loc='best')
    axs[1].set_xlim(period_us.value / 2.0 - period_us.value/window_factor , period_us.value / 2.0 + period_us.value/window_factor)
    axs[1].axvline(period_us.value / 2.0, color="black", linestyle=":", zorder=1)
    axs[0].axvline(period_us.value / 2.0, color="black", linestyle=":", zorder=1)

    # Integrated pulse profile on the top of the plot
    ar.fscrunch()
    axs[0].plot(np.linspace(0, period_us.value, nbins), ar.getData(), color="black")
    plt.tight_layout()
    plt.savefig(f"../figures/{psr_name}_waterfall.pdf")
    plt.show()

if __name__ == "__main__":
    main()