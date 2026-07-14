"""
epoch_dm_timeseries.by, by @sophiasosafiscella
IMPORTANT: If the par file already has a NANOGrav-style DMX component (DMX_0001, DMXR1_0001, ... piecewise fit jointly with everything else), that
is doing a *different* thing than what's implemented here. Before running approach B, either:
    (a) remove the existing DispersionDMX component and keep a
        constant DM (model.remove_component("DispersionDMX")), or
    (b) freeze all its DMX_* values at their fit solution and treat them as
        fixed input, then fit only the top-level DM parameter epoch by epoch.
Mixing "fit DM directly" with "DMX bins still free" is wrong!
"""

import numpy as np
import astropy.units as u
import pint
from pint import DMconst
from pint.toa import TOAs
from pint.models.timing_model import TimingModel
from pint.fitter import Fitter
from pint import utils as pu
import copy
import sys

# ---------------------------------------------------------------------------
# Epoch definition
# ---------------------------------------------------------------------------
def group_into_epochs(toas: TOAs, gap_days: float = 1.0):
    """
    Group TOAs into epochs by finding gaps in sorted MJD larger than gap_days. Return a list of boolean masks (one per epoch), each aligned
    to the original toas order.
    """
    mjds = toas.get_mjds().value
    order = np.argsort(mjds)
    sorted_mjds = mjds[order]

    breaks = np.where(np.diff(sorted_mjds) > gap_days)[0]
    boundaries = np.concatenate(([-1], breaks, [len(sorted_mjds) - 1]))

    masks = []
    for i in range(len(boundaries) - 1):
        lo = sorted_mjds[boundaries[i] + 1]
        hi = sorted_mjds[boundaries[i + 1]]
        mask = (mjds >= lo - 1e-6) & (mjds <= hi + 1e-6)
        masks.append(mask)
    return masks

def per_epoch_DMX_binning(model: TimingModel, toas: TOAs, gap_days: float = 0.25):
    """
    Create one DispersionDMX window for each observing epoch. Observing epochs are identified as contiguous groups of TOAs separated by more than gap_days. Each epoch receives its own free DMX parameter.

    Returns
    -------
    model : TimingModel with the new DispersionDMX component attached
    """

    # Get the new windows
    epoch_masks = group_into_epochs(toas, gap_days)

    # Create a new DMX model based on those windows
    from pint.models.timing_model import Component
    dmx_class = Component.component_types["DispersionDMX"]
    dmx_comp = dmx_class()

    all_mjds = toas.get_mjds()
    covered = np.zeros(len(toas), dtype=bool)

    for ii, epoch in enumerate(epoch_masks):
        mjds = all_mjds[epoch]  # TOAs inside this window
        covered |= epoch

        if len(mjds) == 0:
            continue

        # A TOA that lands exactly on the edge of a window can occasionally fall outside due to floating-point
        # comparisons so I'm adding a delta epsilon
        epsilon = 1e-6 * u.d
        DMXR1 = mjds.min() - epsilon
        DMXR2 = mjds.max() + epsilon

        if ii == 0:
            dmx_comp.DMX_0001.value = 0.0
            dmx_comp.DMX_0001.frozen = False
            dmx_comp.DMXR1_0001.value = DMXR1.value
            dmx_comp.DMXR2_0001.value = DMXR2.value
        else:
            # Add the DMX parameter value
            dmx_par = pint.models.parameter.prefixParameter(
                parameter_type="float",
                name="DMX_{:04d}".format(ii + 1),
                value=0.0,
                units=u.pc / u.cm ** 3,
                frozen=False,
                tcb2tdb_scale_factor=DMconst,  # <- the fix
            )
            dmx_comp.add_param(dmx_par, setup=True)

            # Add the left edge of the DMX window
            dmxr1_par = pint.models.parameter.prefixParameter(
                parameter_type="mjd",
                name="DMXR1_{:04d}".format(ii + 1),
                value=DMXR1.value,
                units=u.d,
                tcb2tdb_scale_factor=u.Quantity(1),  # <- the fix
            )
            dmx_comp.add_param(dmxr1_par, setup=True)

            # Add the right edge of the DMX window
            dmxr2_par = pint.models.parameter.prefixParameter(
                parameter_type="mjd",
                name="DMXR2_{:04d}".format(ii + 1),
                value=DMXR2.value,
                units=u.d,
                tcb2tdb_scale_factor=u.Quantity(1),  # <- the fix
            )
            dmx_comp.add_param(dmxr2_par, setup=True)

    dmx_comp.validate()
    assert covered.all()

    # Remove the old DMX model and add the new component
    if "DispersionDMX" in model.components:  # Remove DMX component from the model
        model.remove_component("DispersionDMX")
    model.add_component(dmx_comp, validate=True)

    return model