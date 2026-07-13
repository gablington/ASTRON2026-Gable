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


def dmx_ranges_fixed(toas: TOAs, divide_freq=1000.0 * u.MHz, binwidth=15.0 * u.d,
                     verbose=False):
    from pint.models.timing_model import Component

    MJDs = toas.get_mjds()
    freqs = toas.table["freq"].quantity

    DMXs = []
    prevbinR2 = MJDs[0] - 0.001 * u.d
    while np.any(MJDs > prevbinR2):
        startMJD = MJDs[MJDs > prevbinR2][0]
        binidx = np.logical_and(MJDs > prevbinR2, MJDs <= startMJD + binwidth)
        if not np.any(binidx):
            break
        binMJDs = MJDs[binidx]
        binfreqs = freqs[binidx]
        loMJDs = binMJDs[binfreqs < divide_freq]
        hiMJDs = binMJDs[binfreqs >= divide_freq]
        if np.any(binfreqs < divide_freq) and np.any(binfreqs > divide_freq):
            DMXs.append(pu.dmxrange(list(loMJDs), list(hiMJDs)))
        prevbinR2 = binMJDs.max()

    if verbose:
        print("Good DMX ranges (N below/above divide_freq):")
        for DMX in DMXs:
            DMX.sum_print()

    mask = np.zeros_like(MJDs.value, dtype=bool)
    for DMX in DMXs:
        mask[np.logical_and(MJDs >= DMX.min, MJDs <= DMX.max)] = True

    dmx_class = Component.component_types["DispersionDMX"]
    dmx_comp = dmx_class()
    for ii, DMX in enumerate(DMXs):
        if ii == 0:
            dmx_comp.DMX_0001.value = 0.0
            dmx_comp.DMX_0001.frozen = False
            dmx_comp.DMXR1_0001.value = DMX.min.value
            dmx_comp.DMXR2_0001.value = DMX.max.value
        else:
            dmx_par = pint.models.parameter.prefixParameter(
                parameter_type="float",
                name="DMX_{:04d}".format(ii + 1),
                value=0.0,
                units=u.pc / u.cm ** 3,
                frozen=False,
                tcb2tdb_scale_factor=DMconst,  # <- the fix
            )
            dmx_comp.add_param(dmx_par, setup=True)

            dmxr1_par = pint.models.parameter.prefixParameter(
                parameter_type="mjd",
                name="DMXR1_{:04d}".format(ii + 1),
                value=DMX.min.value,
                units=u.d,
                tcb2tdb_scale_factor=u.Quantity(1),  # <- the fix
            )
            dmx_comp.add_param(dmxr1_par, setup=True)

            dmxr2_par = pint.models.parameter.prefixParameter(
                parameter_type="mjd",
                name="DMXR2_{:04d}".format(ii + 1),
                value=DMX.max.value,
                units=u.d,
                tcb2tdb_scale_factor=u.Quantity(1),  # <- the fix
            )
            dmx_comp.add_param(dmxr2_par, setup=True)

    dmx_comp.validate()
    return mask, dmx_comp


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


# ---------------------------------------------------------------------------
# Approach A: PINT fit using epoch-sized DMX bins
# ---------------------------------------------------------------------------
def joint_dmx_epoch_fit(model: TimingModel, toas: TOAs, binwidth=0.5 * u.d,
                         divide_freq=1000 * u.MHz):
    """
    Bin TOAs into epoch-sized DMX windows and fit everything (DMX offsets + all other free parameters) simultaneously in one GLS fit. This is the same as NANOGrav's standard DMX methodology,
    just with bins narrow enough that each one is a single observing epoch rather than a multi-day window.

    IMPORTANT! Binwidth should be a bit longer than your longest intra-epoch TOA span (e.g. a few hours) but much shorter than the gap between epochs, so each
    bin encloses exactly one session's multi-frequency TOAs (I learned this the hard way!)
    """
    mask, dmx_component = dmx_ranges_fixed(toas, divide_freq=divide_freq,
                                            binwidth=binwidth, verbose=False)


    if not np.all(mask):
#        n_dropped = np.sum(~mask)
#        print(f"[joint_dmx_epoch_fit] {n_dropped} TOAs fell outside any DMX bin and will be excluded from this fit.")
        toas = toas[mask]

    if "DispersionDMX" in model.components:  # Remove DMX component from the model
        model.remove_component("DispersionDMX")
    model.add_component(dmx_component, validate=True)


    # FREEZE ALL THE NON-DMX PARAMETER

    fitter = Fitter.auto(toas, model)  # Set up the fitter
    fitter.fit_toas()  # Do the fit

    dmx_result = pu.dmxparse(fitter, save=False)
    return fitter, dmx_result


# ---------------------------------------------------------------------------
# Approach B: per-epoch fit
# ---------------------------------------------------------------------------
def sequential_epoch_dm(model: TimingModel, toas: TOAs, epoch_masks,
                         n_outer_iter: int = 3, verbose: bool = True):
    """
    For each epoch freeze everything except DM, fit DM using only that epoch's TOAs against the current global model. After looping over all
    epochs, rebuild a piecewise-constant DM(t) correction from the fitted values, refit the *non-DM* parameters against the whole dataset with
    that correction applied, and repeat.
    """
    base_free_params = [p for p in model.free_params]
    results = None

    for outer in range(n_outer_iter):
        results = []

        for i, mask in enumerate(epoch_masks):
            sub = toas[mask]
            freqs = sub.table["freq"].quantity.to_value(u.MHz)

#            if len(np.unique(np.round(freqs, 1))) < 2:
                #if verbose:
                 #   print(f"[epoch {i}] skipped: only one frequency ({freqs[0]:.0f} MHz), can't constrain DM.")
                #continue

            # FREEZE EVERYTHING BUT THE DM
            epoch_model = model  # share model object; we only touch .frozen
            for p in epoch_model.free_params:
                getattr(epoch_model, p).frozen = True
            epoch_model.DM.frozen = False

            # FIT THE DM
            f = Fitter.auto(sub, epoch_model)
            f.fit_toas(maxiter=1)

 #           try:
 #               f = Fitter.auto(sub, epoch_model)
#                f.fit_toas()

#            except Exception as e:
#                if verbose:
#                    print(f"[epoch {i}] fit failed: {e}")
#                continue

            mjd_center = sub.get_mjds().value.mean()
            results.append({
                "mjd": mjd_center,
                "dm": f.model.DM.quantity,
                "dm_err": f.model.DM.uncertainty,
                "n_toas": len(sub),
                "freq_span_MHz": (freqs.max() - freqs.min()),
            })

            # adopt this epoch's DM back into the shared model so later iterations can see it too
            model.DM.quantity = f.model.DM.quantity
            model.DM.uncertainty = f.model.DM.uncertainty

        # refit non-DM parameters against the whole dataset, WITH DM frozen at the converged value trend
        for p in base_free_params:
            getattr(model, p).frozen = (p == "DM")
        model.DM.frozen = True

        try:
            full_fitter = Fitter.auto(toas, model)
            full_fitter.fit_toas()
            model = full_fitter.model
        except Exception as e:
            if verbose:
                print(f"[outer iter {outer}] global refit failed: {e}")

        if verbose:
            print(f"outer iteration {outer + 1}/{n_outer_iter}: {len(results)} epochs fit successfully")

    return results, model


# ---------------------------------------------------------------------------
# This is how we run things
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     from pint.models import get_model_and_toas

#     parfile =
#     timfile =
#     model, toas = get_model_and_toas(parfile, timfile)

#     # Approach A: joint fit with epoch-sized DMX bins
#     model_a = model.copy()
#     fitter_a, dmx_a = joint_dmx_epoch_fit(model_a, toas.copy(), binwidth=0.5 * u.d)
#     print("Approach A: mean DM =", dmx_a["mean_dmx"], "+/-", dmx_a["avg_dm_err"])

#     # # Approach B: sequential per-epoch
#     # model_b = model.copy()
#     # if "DispersionDMX" in model_b.components:
#     #     model_b.remove_component("DispersionDMX")

#     # epoch_masks = group_into_epochs(toas, gap_days=1.0)
#     # print(f"Found {len(epoch_masks)} epochs")

#     # results_b, model_b = sequential_epoch_dm(model_b, toas, epoch_masks, n_outer_iter=3)

#     # for r in results_b:
#     #     print(f"MJD {r['mjd']:.2f}: DM = {r['dm']:.6f} +/- {r['dm_err']:.6f}  ({r['n_toas']} TOAs, {r['freq_span_MHz']:.0f} MHz span)")