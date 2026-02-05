from py_wake import numpy as np
from py_wake.deficit_models.gaussian import ZongGaussianDeficit
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.rotor_avg_models import CGIRotorAvg, GaussianOverlapAvgModel
from py_wake.site._site import UniformSite
from py_wake.superposition_models import LinearSum, SqrMaxSum, WeightedSum
from py_wake.turbulence_models import CrespoHernandez
from py_wake.wind_farm_models import PropagateDownwind


def get_flowmodel(wt=None, site=None, propagate=PropagateDownwind):
    if site is None:
        site = UniformSite()  # placeholder
    if wt is None:
        # add the project root to sys.path (for IEA22s)
        import os
        import sys

        project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, project_folder)
        from utils.iea22s import IEA22s

        wt = IEA22s()

    deficit_model = ZongGaussianDeficit(
        a=[0.38, 4e-3],
        deltawD=1.0 / np.sqrt(2),
        eps_coeff=0.35,
        lam=7.5,
        B=3,
        rotorAvgModel=CGIRotorAvg(21),
        groundModel=None,
        use_effective_ws=True,
        use_effective_ti=True,
    )

    turbulence_model = CrespoHernandez(
        ct2a=ct2a_mom1d,
        c=[0.73, 0.83, 0.03, -0.32],  # 10.1016/j.jweia.2023.105504  # -0.03 is misprint
        addedTurbulenceSuperpositionModel=SqrMaxSum(),
        # rotorAvgModel=CGIRotorAvg(21),
    )

    # Default values from py_wake.literature except deflection model
    # https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake/-/blob/master/py_wake/literature/gaussian_models.py?ref_type=heads
    wf_model = propagate(
        site=site,
        windTurbines=wt,
        wake_deficitModel=deficit_model,
        superpositionModel=LinearSum(),
        deflectionModel=JimenezWakeDeflection(),
        turbulenceModel=turbulence_model,
    )
    return wf_model