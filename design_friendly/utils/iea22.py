import os
import numpy as np
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from design_friendly.models import models_filepath

hsw_path = os.path.join(models_filepath, "hs2_steady_state.pwr")
pwr = np.loadtxt(hsw_path)  # @ricriv run

# wt = WindTurbine(
#     name="IEA 22 MW RWT",
#     diameter=284.0,
#     hub_height=170.0,
#     powerCtFunction=PowerCtTabular(
#         ws=pwr[:, 0],
#         power=pwr[:, 1] * 0.9542919819763047,
#         power_unit="kW",
#         ct=pwr[:, 4],
#         ws_cutin=3.0,
#         ws_cutout=25.0,
#         power_idle=0.0,
#         ct_idle=0.0,
#     ),
# )


class IEA22(WindTurbine):
    def __init__(self, method="pchip"):
        """
        Parameters
        ----------
        method : {'linear', 'pchip'}
            linear(fast) or pchip(smooth and gradient friendly) interpolation
        """
        WindTurbine.__init__(
            self,
            name="IEA22",
            diameter=284,
            hub_height=170,
            powerCtFunction=PowerCtTabular(
                ws=pwr[:, 0],
                power=pwr[:, 1] * 0.9542919819763047,
                power_unit="kW",
                ct=pwr[:, 4],
                method=method,
                ws_cutin=3.0,
                ws_cutout=25.0,
                ct_idle=0.0,
                power_idle=0.0,
            ),
        )


def main():
    wt = IEA22()
    print("Diameter", wt.diameter())
    print("Hub height", wt.hub_height())
    ws = np.arange(0, 25)
    import matplotlib.pyplot as plt

    plt.plot(ws, wt.power(ws), ".-", label="power [W]")
    c = plt.plot([], label="ct")[0].get_color()
    plt.legend()
    ax = plt.twinx()
    ax.plot(ws, wt.ct(ws), ".-", color=c)
    plt.show()
    wt.plot_power_ct()


if __name__ == "__main__":
    main()
