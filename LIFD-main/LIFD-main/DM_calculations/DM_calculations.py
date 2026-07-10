import numpy as np
import pandas as pd
import plotly.express as px
import pypulse as pyp
from glob import glob
from tqdm import tqdm
import os
import sys


def setup_dm_analysis(psr_name: str):
    """Set up parameters and files."""
    output_file = f"./{psr_name}/{psr_name}_DM_results.csv"
    archive_files = glob(f"./{psr_name}/ff_files/*.ff")

    initial_dm = pyp.Archive(archive_files[0], prepare=False).getDM()
    delta_dm = initial_dm / 100.0 * 5.0  # 5% of the nominal DM
    dm_range = np.linspace(initial_dm - delta_dm, initial_dm + delta_dm, 100)

    # Load the template
    template_dir = "/home/svsosafiscella/PycharmProjects/NANOGrav15yr_PulsarTiming_v2.0.1/narrowband/template/"
    template_file = glob(f"{template_dir}{psr_name}.*.GUPPI.15y.x.sum.sm")[0]
    template_profile = pyp.Archive(template_file).getSinglePulses()

    # Calculate the nominal DM
    DM_0 = pyp.Archive(archive_files[0]).getDM()

    return output_file, archive_files, dm_range, template_profile, DM_0



def process_single_file(file: str, dm_range: np.ndarray, template_profile) -> np.ndarray:
    """Process a single archive file across all DM values and return S/N ratios."""
    signal_to_noise = np.zeros(len(dm_range))

    for i, dm_value in enumerate(dm_range):
        ar = pyp.Archive(file, prepare=False, verbose=False)
        ar.dedisperse(DM=dm_value, wcfreq=True)
        ar.pscrunch()
        ar.center()
        ar.fscrunch()
        ar.tscrunch()

        # Calculate the S/N
        signal_to_noise[i] = ar.fitPulses(template_profile, nums=[5])[0]

    return signal_to_noise


def calculate_snr_values(files: list, dm_range: np.ndarray,
                        template_profile, output_file: str) -> pd.DataFrame:
    """Calculate S/N values for all DM values across all archive files."""
    results = np.zeros((len(dm_range), len(files)))

    for j, archive_file in tqdm(enumerate(files)):
        results[:, j] = process_single_file(archive_file, dm_range, template_profile)

    results_df = pd.DataFrame(results, columns=files, index=dm_range)
    results_df.to_csv(output_file)
    return results_df


def main():

    #PSR_name = "J2145-0750"
    #PSR_name = "J1909-3744"

    output_file, archive_files, dm_range, template_profile, DM_0 = setup_dm_analysis(PSR_name)

    if os.path.exists(output_file):
        results_df = pd.read_csv(output_file, index_col=0)
    else:
        results_df = calculate_snr_values(archive_files, dm_range, template_profile, output_file)

    average_results = results_df.mean(axis=1).reset_index()
    average_results.columns = ["DM", "snr"]
    fig = px.line(average_results, x="DM", y="snr", title=f"Nominal DM = {DM_0}")
    fig.show()
    fig.write_image(f"{PSR_name}_SNR_curve.png")


if __name__ == "__main__":
    main()
