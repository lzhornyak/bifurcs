from matplotlib import pyplot as plt
from pycobi import ODESystem
from pyrates import CircuitTemplate

func_name = "vector_field"
file_name = "system_equations"
dt = 1e-3
solver = "scipy"

# # initialize circuit template
# template = CircuitTemplate.from_yaml('models.pitchfork.pitch')
#
# # update circuit template variables
# template.update_var(node_vars={'node/eqn/x': 2.0, 'node/eqn/r': 0.5})
# # template.update_var(edge_vars=kwargs.pop("edge_vars"))
#
# # generate fortran files
# _, _, params, state_vars = template.get_run_func(func_name, dft, file_name=file_name, backend="fortran",
#                                                  float_precision="float64", auto=True, vectorize=False,
#                                                  solver=solver, NPR=100, NMX=10000)

# initialize ODESystem
ode = ODESystem(auto_dir="~/auto/07p", e=file_name, c="ivp",
                params=['r'], state_vars=['x'], eq_file=file_name)

# ode = ODESystem.from_yaml(
#     "models.pitchfork.pitch", auto_dir="~/auto/07p",
#     node_vars={'node/eqn/x': 2.0, 'node/eqn/r': 0.5},
#     NPR=100, NMX=10000
# )

ode.plot_continuation("t", "x", cont=0)
plt.show()

r_sols, r_cont = ode.run(
    origin=0, starting_point='EP', name='r_cont', bidirectional=True,
    ICP="r", RL0=-2.0, RL1=2.0, IPS=1, ILP=1, ISP=1, ISW=1, NTST=20,
    NCOL=4, IAD=3, IPLT=0, NBC=0, NINT=0, NMX=100000, NPR=10, MXBF=5, IID=2,
    ITMX=8, ITNW=5, NWTN=3, JAC=0, EPSL=1e-06, EPSU=1e-06, EPSS=1e-04,
    DS=1e-4, DSMIN=1e-8, DSMAX=5e-4, IADS=1, THL={}, THU={}, STOP={}
)
ode.plot_continuation("r", "x", cont="r_cont")
plt.show()

# bp_sols, bp_cont = ode.run(origin=r_cont, starting_point='BP', name='bp_cont', IPS=1, ISP=1, ISW=-1)
fig, ax = plt.subplots()
ode.plot_continuation("r", "x", cont="r_cont", ax=ax)
# ode.plot_continuation("r", "x", cont="bp_cont", ax=ax)
plt.show()
#
# ode.close_session(clear_files=True)
