from os import PathLike
from typing import TypeAlias
from warnings import warn

import jax.numpy as jnp
import jetto_tools
from jetto_tools.classes import jsp_to_xarray, jst_to_xarray
from torax.constants import CONSTANTS

FileName: TypeAlias = str | bytes | PathLike


def config(
    jsp_path: FileName,
    jst_path: FileName,
    jset_path: FileName | None = None,
    eqdsk_path: FileName | None = None,
) -> dict:
    """Create a TORAX configuration dictionary from JETTO files.

    Parameters
    ----------
    jsp_path : str | bytes | PathLike
        Path to a JSP file.
    jst_path : str | bytes | PathLike
        Path to a JST file.
    jset_path : str | bytes | PathLike | None, optional
        Path to a JSET file, defaults to None.
    eqdsk_path : str | bytes | PathLike | None, optional
        Path to an EQDSK file, defaults to None.

    Returns
    -------
    dict
        A TORAX configuration dictionary.
    """
    #################
    # 1. Load files #
    #################
    # JSP and JST are required
    jsp = jsp_to_xarray(jetto_tools.binary.read_binary_file(jsp_path))
    jst = jst_to_xarray(jetto_tools.binary.read_binary_file(jst_path))

    # JSET is optional
    if jset_path is not None:
        with open(jset_path) as f:
            # Load the JSET file
            jset = jetto_tools.jset.JSET(f.read())
            # Convert to arrays rather than strings
            jset.collapse_all_arrays()
    else:
        jset = None

    #############################
    # 2. Initialise config dict #
    #############################
    torax_config = {
        "runtime_params": {
            "plasma_composition": {},
            "profile_conditions": {},
            "numerics": {},
        },
        "geometry": {},
        "sources": {},
        "transport": {},
        "stepper": {},
        "time_step_calculator": {},
    }

    ###################
    # 3. Set numerics #
    ###################
    numerics = torax_config["runtime_params"]["numerics"]
    # Reference density [m^-3] - used as a normalising factor for numerical convenience
    numerics["nref"] = 1e19
    # Equations to solve
    numerics["ion_heat_eq"] = True
    numerics["el_heat_eq"] = True
    numerics["current_eq"] = True
    numerics["dens_eq"] = True
    # Make TORAX time start from 0
    time_offset = jst.time.values[0]
    time = jst.time.values - time_offset
    numerics["t_initial"] = 0
    numerics["t_final"] = time[-1]

    #############################
    # 4. Set plasma composition #
    #############################
    plasma_composition = torax_config["runtime_params"]["plasma_composition"]

    # Main species charge
    ## Only support hydrogen plasmas
    plasma_composition["Zi"] = 1.0

    if jset is not None:
        # Main species mass
        species_mass = jnp.array(jset.get("EquationsPanel.ionDens.mass", jnp.nan))
        species_fraction = jnp.array(
            jset.get("EquationsPanel.ionDens.fraction", jnp.nan)
        )
        plasma_composition["Ai"] = jnp.sum(species_mass * species_fraction)

        # Effective charge
        plasma_composition["Zeff"] = jset.get("SancoICZPanel.ConstantZeff", None)
        if plasma_composition["Zeff"] is None:
            warn("Zeff not set to constant in JSET; Zeff not set.")
            del plasma_composition["Zeff"]

        # Impurity species charge
        ## We average over all the impurities
        Zimp = 0
        n_impurity_species = 0
        for i in range(6):
            if jset[f"ImpOptionPanel.impuritySelect[{i}]"]:
                Zimp += jset[f"ImpOptionPanel.impurityCharge[{i}]"]
                n_impurity_species += 1
        Zimp /= n_impurity_species
        plasma_composition["Zimp"] = Zimp

    else:
        warn("JSET not loaded; Ai, Zeff, Zimp not set.")

    #############################
    # 5. Set profile conditions #
    #############################
    profile_conditions = torax_config["runtime_params"]["profile_conditions"]
    rho_norm = jsp.XRHO.values[0]

    # Plasma current [MA]
    # Note: JETTO current is -ve
    profile_conditions["Ip"] = (time, -jst.CUR.values / 1e6)

    # Temperature [keV]
    ## Initial or prescribed profiles
    ## Note: if evolving the temperature profiles, only the initial value will be used
    profile_conditions["Te"] = (time, rho_norm, jsp.TE.values / 1e3)
    profile_conditions["Ti"] = (time, rho_norm, jsp.TI.values / 1e3)
    ## Boundary conditions
    profile_conditions["Te_bound_right"] = (time, jst.TEBO.values / 1e3)
    profile_conditions["Ti_bound_right"] = (time, jst.TIBO.values / 1e3)
    ## Pedestal
    profile_conditions["Teped"] = (time, jst.TEBA.values / 1e3)
    profile_conditions["Tiped"] = (time, jst.TIBA.values / 1e3)

    # Density [nref m^-3]
    ## Initial or prescribed profiles
    ## Note: if evolving the density profiles, only the initial value will be used
    profile_conditions["ne"] = (time, rho_norm, jsp.NE.values / numerics["nref"])
    ## Boundary conditions
    profile_conditions["ne_bound_right"] = (time, jst.NEBO.values / numerics["nref"])
    ## Pedestal
    profile_conditions["neped"] = (time, jst.NEBA.values / numerics["nref"])
    ## nbar = line averaged density
    profile_conditions["normalize_to_nbar"] = True
    profile_conditions["nbar"] = (time, jst.NEL.values / numerics["nref"])
    profile_conditions["ne_is_fGW"] = False

    #  Additional pedestal settings
    profile_conditions["set_pedestal"] = True
    profile_conditions["Ped_top"] = (time, jst.ROBA.values)

    ###################
    # 5. Set geometry #
    ###################
    if eqdsk_path is not None:
        torax_config["geometry"] = {
            "geometry_type": "EQDSK",
            "geometry_file": eqdsk_path,
            "Ip_from_parameters": True,
        }
    else:
        warn("No EQDSK file provided; geometry not set.")

    ##################
    # 6. Set sources #
    ##################
    sources = torax_config["sources"]

    # Internal plasma heat sources and sinks
    ## Ohmic and Qei will always be set
    sources["ohmic_heat_source"] = {}  # default
    sources["qei_source"] = {}  # default

    ## Fusion power
    if jset is not None:
        if jset.get("FusionPanel.select", False):
            sources["fusion_heat_source"] = {}  # default
    else:
        warn("JSET not loaded; fusion heat source not set.")

    ## Bremsstrahlung
    if jset is not None:
        if jset.get("RadiationAddPanel.bremsstrahlung", False):
            sources["bremsstrahlung_heat_sink"] = {}  # default
    else:
        warn("JSET not loaded; Bremsstrahlung heat not set.")

    ## TODO: Add Prad

    # Current sources
    ## Bootstrap (Sauter model)
    if jset is not None:
        if jset.get("CurrentPanel.selBootstrap", False):
            sources["j_bootstrap"] = {
                "mode": "model_based",
                "bootstrap_mult": jset["CurrentPanel.bootstrapCoeff"],
            }
    else:
        warn("JSET not loaded; bootstrap current not set.")

    # Density sources
    ## Pellet (Continuous)
    if jset is not None:
        # Pellet is set in a somewhat convoluted way in JSET
        sources["pellet_source"] = {
            "mode": "formula_based",
            "pellet_deposition_location": jset.extras["SPCEN"].as_dict()[None],
            "pellet_width": jset.extras["SPWID"].as_dict()[None],
            "S_pellet_tot": (time, jst.SPEL.values),
        }
    else:
        warn("JSET not loaded; pellet source not set.")

    # Electron-cyclotron Heating
    warn("ECHCD not yet implemented in TORAX; none will be set.")

    ####################
    # 7. Set transport #
    ####################
    transport = torax_config["transport"]

    if jset is not None:
        # Bohm-gyroBohm transport model
        ## Confusingly, JETTO has a hidden set of prefactors hardcoded
        if jset.get("TransportStdJettoDialog.selBohm", False) and jset.get(
            "TransportStdJettoDialog.selGyroBohm", False
        ):
            transport["transport_model"] = "bohm-gyrobohm"
            transport["bohm-gyrobohm_params"] = {
                "chi_e_bohm_coeff": jset.get("TransportStdAdvDialog.elecBohmCoeff", 1.0)
                * 2e-4,
                "chi_e_gyrobohm_coeff": jset.get(
                    "TransportStdAdvDialog.elecGBohmCoeff", 1.0
                )
                * 5e-6,
                "chi_i_bohm_coeff": jset.get("TransportStdAdvDialog.ionBohmCoeff", 1.0)
                * 2e-4,
                "chi_i_gyrobohm_coeff": jset.get(
                    "TransportStdAdvDialog.ionGBohmCoeff", 1.0
                )
                * 5e-6,
            }
            # TODO: There's a coefficient for the inward particle pinch in
            # JSET/Transport/Additional/Inward particle pinch, but it's not clear what it sets.
            warn("d_face_c1 and d_face_c2 not yet supported; using default values.")
        # TODO: QLKNN transport model
        # elif
        # TODO: CGM transport model
        # elif
        else:
            warn("No known transport model selected in JSET; none will be set.")
    else:
        warn("JSET not loaded; transport model not set.")

    #############################
    # 8. Return the config dict #
    #############################
    return torax_config


def jz_to_jdotB(jz, A, rho):
    """Convert a JETTO JZ current density to a <j.B> current density.

    Parameters
    ----------
    jz : jnp.ndarray
        JETTO JZ current density.
    A : jnp.ndarray
        Array of flux surface areas.
    rho : jnp.ndarray
        Radial coordinate (unnormalised).

    Returns
    -------
    jnp.ndarray
        <j.B> current density.
    """
    return 2 * jnp.pi * rho * jz / jnp.gradient(A, rho)
