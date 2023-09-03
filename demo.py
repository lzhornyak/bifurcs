from pycobi import ODESystem
import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate

ode = ODESystem.from_yaml(
    "model_templates.neural_mass_models.qif.qif_sfa", auto_dir="~/auto/07p",
    node_vars={'p/qif_sfa_op/Delta': 2.0, 'p/qif_sfa_op/alpha': 1.0, 'p/qif_sfa_op/eta': 3.0},
    edge_vars=[('p/qif_sfa_op/r', 'p/qif_sfa_op/r_in', {'weight': 15.0*np.sqrt(2.0)})],
    NPR=100, NMX=30000
)

ode.plot_continuation("t", "p/qif_sfa_op/r", cont=0)
plt.show()

eta_sols, eta_cont = ode.run(
    origin=0, starting_point='EP', name='eta', bidirectional=True,
    ICP="p/qif_sfa_op/eta", RL0=-20.0, RL1=20.0, IPS=1, ILP=1, ISP=2, ISW=1, NTST=400,
    NCOL=4, IAD=3, IPLT=0, NBC=0, NINT=0, NMX=2000, NPR=10, MXBF=5, IID=2,
    ITMX=40, ITNW=40, NWTN=12, JAC=0, EPSL=1e-06, EPSU=1e-06, EPSS=1e-04,
    DS=1e-4, DSMIN=1e-8, DSMAX=5e-2, IADS=1, THL={}, THU={}, UZR={"p/qif_sfa_op/eta": 3.0}, STOP={}
)

ode.plot_continuation("p/qif_sfa_op/eta", "p/qif_sfa_op/r", cont="eta")
plt.show()

hopf_sols, hopf_cont = ode.run(
    origin=eta_cont, starting_point='HB2', name='eta_hopf',
    IPS=2, ISP=2, ISW=-1, UZR={"p/qif_sfa_op/eta": -2.0}
)

fig, ax = plt.subplots()
ode.plot_continuation("p/qif_sfa_op/eta", "p/qif_sfa_op/r", cont="eta", ax=ax)
ode.plot_continuation("p/qif_sfa_op/eta", "p/qif_sfa_op/r", cont="eta_hopf", ax=ax, ignore=["UZ", "BP"])
plt.show()

ode.run(origin=hopf_cont, starting_point="UZ1", c="ivp", name="lc")
ode.plot_continuation("t", "p/qif_sfa_op/r", cont="lc")
plt.show()

ode.close_session(clear_files=True)
