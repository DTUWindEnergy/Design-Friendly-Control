from py_wake.deficit_models.gaussian import ZongGaussianDeficit
from py_wake.superposition_models import WeightedSum, SqrMaxSum, LinearSum
from py_wake.rotor_avg_models import GaussianOverlapAvgModel, CGIRotorAvg
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.turbulence_models import CrespoHernandez
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.site._site import UniformSite
from py_wake.deficit_models.utils import ct2a_mom1d
import numpy as np


def get_flowmodel(wt=None, site=None):
    if site is None:
        site = UniformSite()  # placeholder
    if wt is None:
        # add the project root to sys.path (for IEA22s)
        import os, sys

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
        c=[0.73, 0.83, 0.03, -0.32],
        addedTurbulenceSuperpositionModel=SqrMaxSum(),
    )

    # Defaults from py_wake.literature except deflection model
    # https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake/-/blob/master/py_wake/literature/gaussian_models.py?ref_type=heads
    wf_model = PropagateDownwind(
        site=site,
        windTurbines=wt,
        wake_deficitModel=deficit_model,
        superpositionModel=LinearSum(),
        deflectionModel=JimenezWakeDeflection(),
        turbulenceModel=turbulence_model,
    )
    return wf_model


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    wf_model = get_flowmodel()
    D = 284
    x = np.arange(0, 5 * 11, 5)
    x = x * D

    y = np.zeros_like(x)
    sim_res = wf_model(x, y, wd=270.5, ws=7, TI=0.06, yaw=0, tilt=0)
    print(sim_res)
    sim_res.flow_map().plot_wake_map()  # this will take a long time
    plt.show()
    plt.plot(
        sim_res["x"].values.squeeze() / D,
        sim_res["WS_eff"].values.squeeze(),
        marker="x",
        linestyle="--",
        color="gray",
    )
    plt.xlabel("Distance [D]")
    plt.ylabel("WS_eff [m/s]")
    plt.legend()
    plt.tight_layout()
    plt.show()
