import numpy as np
import pypulse as pyp
from astropy.time import Time
from pint.models import get_model
from scipy.optimize import minimize_scalar
from glob import glob
from tqdm import tqdm
import math
import sys

# Constants
INVERSE_PHI = (math.sqrt(5) - 1) / 2  # Golden ratio inverse: 1 / phi
DEFAULT_TOLERANCE = 1e-4
DM_SEARCH_RANGE_PERCENT = 10.0  # Percentage of nominal DM to search


def calculate_snr(files, dm_value, Rcvr_800_template, Rcvr1_2_template):
    """Calculate the average S/N for a given DM value"""
    snr_values = np.zeros(len(files))
    for j, file in tqdm(enumerate(files)):

        ar = pyp.Archive(file, prepare=False, verbose=False)
#        print(Time(ar.getMJD(), format='mjd').iso, ar.getMJD(), ar.getDM())
        ar.dedisperse(DM=dm_value, wcfreq=True)
        ar.pscrunch()
        ar.center()
        ar.fscrunch()
        ar.tscrunch()

        if np.any(np.isnan(ar.getData())):  # If the pulse is empty
            snr_values[j] = 0.0
        else:
            # The template we use to calculate the S/N will change depending on the frontend used for this particular observation
            if ar.getFrontend() == "Rcvr_800":
                snr_values[j] = ar.fitPulses(Rcvr_800_template, nums=[5])[0]
            elif ar.getFrontend() == "Rcvr1_2":
                snr_values[j] = ar.fitPulses(Rcvr1_2_template, nums=[5])[0]
            else:
                print(f"Unknown frontend: {ar.getFrontend()}")
                sys.exit(1)

    average_snr = np.mean(snr_values)
    return average_snr


def golden_section_search(snr_func, lower_bound, upper_bound, tolerance=DEFAULT_TOLERANCE):
    """
    Perform golden section search to find the maximum of a unimodal function.

    Args:
        function: Function to maximize (should take single argument)
        lower_bound: Lower bound of search interval
        upper_bound: Upper bound of search interval
        tolerance: Convergence tolerance for the search interval width

    Returns:
        tuple: (optimal_value, max_objective_value, num_iterations)
    """
    a = lower_bound
    b = upper_bound
    iteration = 1

    while b - a > tolerance:
        print(f"Iteration {iteration}. Searching in the interval [{a:.4f}, {b:.4f}]")

        # Calculate interior points using golden ratio
        c = b - (b - a) * INVERSE_PHI  # Point closer to b
        d = a + (b - a) * INVERSE_PHI  # Point closer to a

        fc = snr_func(c)
        fd = snr_func(d)

        # Evaluate function at interior points
        if fc > fd:
            b = d
        else:
            a = c

        iteration += 1

    optimal_dm = (a + b) / 2
    optimal_snr = snr_func(optimal_dm)

    return optimal_dm, optimal_snr, iteration - 1


def load_template_profile(psr_name, rcvr, template_dir):
    """Load the template profile for a given pulsar."""
    template_file = glob(f"{template_dir}{psr_name}.{rcvr}*.GUPPI.15y.x.sum.sm")

    # Check that only one file was found
    if len(template_file) == 1:
        template_file = template_file[0]
    elif len(template_file) > 1:
        print(f"Multiple template files found for {psr_name}.{rcvr}")
        sys.exit(1)
    elif len(template_file) == 0:
        print(f"No template file found for {psr_name}.{rcvr}")
        sys.exit(1)

    return pyp.Archive(template_file).getSinglePulses()


def main(psr_name, method):
    """Main execution function for DM optimization using golden section search."""
    # Set up parameters and files
#    psr_name = "J1909-3744"
    ar_files = glob(f"./{psr_name}/ff_files/*.ff")

    if len(ar_files) == 0:
        print(f"No FF files found for {psr_name}")
        sys.exit(1)

    # Load the template
    template_dir = "/home/svsosafiscella/PycharmProjects/LIFD_modeling/NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/template/"
    Rcvr_800_template = load_template_profile(psr_name, 'Rcvr_800', template_dir)
    Rcvr1_2_template = load_template_profile(psr_name, "Rcvr1_2", template_dir)

    # Calculate the nominal DM and search range
    DM_0_from_ar = pyp.Archive(ar_files[0], prepare=True).getDM()
    print(f"Nominal DM from AR = {DM_0_from_ar:.4f}")
    parfile: str = glob(f"../NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/par/{psr_name}_PINT_*.nb.par")[0]
    DM_0 = get_model(parfile).DM.value
    print(f"Nominal DM from parfile = {DM_0:.4f}")
    print(f"Nominal DM = {DM_0:.4f}")

    delta_dm = DM_0 / 100.0 * DM_SEARCH_RANGE_PERCENT
    lower_bound = DM_0 - delta_dm
    upper_bound = DM_0 + delta_dm
    print(f"Search range: [{lower_bound:.4f}, {upper_bound:.4f}]")

    # Create objective function for golden section search
    if method == 'golden':
        def snr_func(dm_value):
            return calculate_snr(ar_files, dm_value, Rcvr_800_template, Rcvr1_2_template)
    elif method == 'brent':
        def snr_func(dm_value):
            return -calculate_snr(ar_files, dm_value, Rcvr_800_template, Rcvr1_2_template)

    # Perform DM search
    if method == "golden":
        final_dm, final_snr, num_iterations = golden_section_search(snr_func, lower_bound, upper_bound)
    elif method =='brent':
        result = minimize_scalar(snr_func, bounds=(lower_bound, upper_bound), method="bounded", options={"xatol": 1e-4})
        final_dm = result.x
        final_snr = -result.fun
#        num_iterations = result.nfev

    print(f"DM = {final_dm}")
    print(f"SNR = {final_snr}")

    np.save(f"../results/{psr_name}/new_DM.npy", final_dm)


if __name__ == "__main__":
#    psr_name: str = "B1937+21"
#    psr_name = "J0610-2100"
#    psr_name: str = "J1012+5307"
    psr_name: str = "J1022+1001"
#    psr_name: str = "J1024-0719"
#    psr_name: str = "J1643-1224"
#    psr_name: str = "J1713+0747"
#    psr_name: str = "J1744-1134"
#    psr_name: str = "J1909-3744"
#    psr_name = "J1918-0642"
#    psr_name = "J2145-0750"
#    psr_name: str = "J2302+4442"
    method: str = 'brent'
    main(psr_name, method)