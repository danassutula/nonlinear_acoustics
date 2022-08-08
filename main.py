'''
The propagation of sound waves can be modeled by the Westervelt equation

(C0 ∇^2 - C1 d^2/dt^2 + C2 d^3/dt^3) p + C3 d^2/dt^2 p^2 + C4 s = 0,

where

C0 = c^2 * T^2
C1 = 1
C2 = δ / c^2 / T 
C3 = β / c^2 / ρ
C4 = T^2

and 

c - Speed of sound (constant) [m/s]
T - Time scaling constant (optional)
δ - Diffusivity of sound [m^2/s]
ρ - Density of medium [kg/m^3]
β - Coefficient of nonlinearity [-]

Reference paper
Karamalis, A., Wein, W., Navab, N. (2010). Fast Ultrasound Image Simulation 
Using the Westervelt Equation. In: Jiang, T., Navab, N., Pluim, J.P.W., 
Viergever, M.A. (eds) Medical Image Computing and Computer-Assisted Intervention 
- MICCAI 2010. MICCAI 2010. Lecture Notes in Computer Science, vol 6361. 
Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-15705-9_30
'''

from dolfin import *

import numpy as np
import matplotlib.pyplot as plt

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O2"

if __name__ == "__main__":

    plt.interactive(True)

    set_log_level(50)
    # set_log_active(False)

    DIM = 2

    L = 0.2 # [m]
    H = 0.2 # [m]

    space_and_time_refinement = 0
    extra_time_refinement = 0 # 0, 1, 2, ...
    line_style = ('-', '--', '-.', ':')[0]

    if 1:
        # Test
        c = 1.0 # Speed of sound [m/s]
        ρ = 1.0 # Density of medium [kg/m^3]
        δ = 0.0 # Diffusivity of sound [m^2/s]
        β = 0.0 # Coefficient of nonlinearity [-]
    else:
        # Parameters of brain tissue
        c = 1500.0 # Speed of sound [m/s]
        ρ = 1100.0 # Density of medium [kg/m^3]
        δ = 4.5e-6 * 0 # Diffusivity of sound [m^2/s]
        β = 6.0 * 1 # Coefficient of nonlinearity [-]

    # Time scaling constant: `t_real [s] = T [s] * t [-]`
    # (for example, time for a wave to do a round-trip)
    T = 2.0 * max(L, H) / c

    ne = 10 * 2**space_and_time_refinement
    nx = ne
    ny = ne

    hx = L / nx
    hy = H / ny
    he = min(hx, hy)

    element_degree = 2

    t0 = 0.0
    tn = 1.0

    # Time-step fraction (i.e. solve the discrete system at time `n+k`)
    k = 1.0

    if element_degree == 1:
        time_step_factor = k * 1.00
    elif element_degree == 2:
        time_step_factor = k * 0.40
    elif element_degree == 3:
        time_step_factor = k * 0.15
    elif element_degree == 4:
        time_step_factor = k * 0.10
    else:
        raise NotImplementedError

    dt = time_step_factor * he * (0.5 ** extra_time_refinement)
    nt = round((tn - t0) / dt)
    dt = (tn - t0) / nt
    dt_k = k * dt

    mesh = RectangleMesh(
        Point(0.0, 0.0), Point(L, H), nx, ny, diagonal='crossed')

    # facet_normal = FacetNormal(mesh) # Not used

    V = FunctionSpace(mesh, "Lagrange", element_degree)

    boundary_subdomains = \
        [
            CompiledSubDomain("near(x[1], side) && on_boundary", side=0.0), # Down
            CompiledSubDomain("near(x[0], side) && on_boundary", side=L),   # Right
            CompiledSubDomain("near(x[1], side) && on_boundary", side=H),   # Up
            CompiledSubDomain("near(x[0], side) && on_boundary", side=0.0), # Left
        ]

    # Dirichlet boundary conditions
    d2pdt2_bc = Expression('0.0', t=0, degree=0) 

    # Neumann boundary conditions
    dpdn_bc = Expression('0.0', t=0, degree=0) 

    # test_case = 0 # 'all_edges_fixed'
    # test_case = 1 # 'bottom_top_edges_fixed_and_left_right_edges_zero_neumann' 
    test_case = 2 # 'left_edge_fixed'
    # test_case = 3 # 'all_edges_free'

    all_boundary_ids = [0, 1, 2, 3]

    if test_case == 0:
        dirichlet_boundary_ids = [0, 1, 2, 3]
    elif test_case == 1:
        dirichlet_boundary_ids = [0, 2]
    elif test_case == 2:
        dirichlet_boundary_ids = [3]
        d2pdt2_bc = Expression('cos(omega*t)', t=0, omega=2*pi/tn * 2, degree=0)
    elif test_case == 3:
        dirichlet_boundary_ids = []
    else:
        raise NotImplementedError
    
    neumann_boundary_ids = list(set(all_boundary_ids).difference(dirichlet_boundary_ids))

    boundary_markers = MeshFunction('size_t', mesh, DIM-1, value=0)
    # boundary_markers.set_all(0)

    dirichlet_boundary_marker = 1
    neumann_boundary_marker = 2
    
    for i in dirichlet_boundary_ids:
        boundary_subdomains[i].mark(boundary_markers, dirichlet_boundary_marker)

    for i in neumann_boundary_ids:
        boundary_subdomains[i].mark(boundary_markers, neumann_boundary_marker)

    # Neumann boundary integration measure
    ds = ds(neumann_boundary_marker, domain=mesh, subdomain_data=boundary_markers)

    source_function_id = 0

    if source_function_id == 0:
        g = Expression('value', value=0.0, degree=0)
    elif source_function_id == 1:
        g = Expression(
            '0.25 * max * (1.0 - cos(2*pi*x[0]/L)) * (1.0 - cos(2*pi*x[1]/H))',
            L=Constant(L), H=Constant(H), max=1.0, degree=3)
    elif source_function_id == 2:
        g = Expression(
            '0.25 * max * (1.0 - cos(2*pi*x[0]/L)) * (1.0 - cos(2*pi*x[1]/H)) * sin(phase+omega*t)',
            L=Constant(L), H=Constant(H), max=1.0, phase=0, omega=2*pi/tn, t=0.0, degree=3)
    else:
        raise NotImplementedError

    def update_boundary_conditions_for_time(t):
        d2pdt2_bc.t = t
        dpdn_bc.t = t
        g.t = t

    # Initial conditions
    p_ic = Function(V, name="p|0")
    dpdt_ic = Function(V, name="dpdt|0")
    d2pdt2_ic = Function(V, name="d2pdt2|0")

    bcs = []

    bcs.append(DirichletBC(V, d2pdt2_bc, boundary_markers, dirichlet_boundary_marker))
    # fixed_dofs = np.array(list(bcs[0].get_boundary_values())) # Not used

    v = TestFunction(V)

    p = Function(V, name="p|n+1")
    dpdt = Function(V, name="dp/dt|n+1")
    d2pdt2 = TrialFunction(V) # "d2p/dt2|n+1"

    p_0 = Function(V, name="p|n")
    dpdt_0 = Function(V, name="dp/dt|n")
    d2pdt2_0 = Function(V, name="d2p/dt2|n")

    #
    # Assume constant `d3pdt3` between time `n` and time `n+1`
    # (Implies `d2pdt2` varies linearly between `n` and `n+1`)
    #

    def d3pdt3_(k, d2pdt2_1):
        return (d2pdt2_1 - d2pdt2_0) / dt

    def d2pdt2_(k, d2pdt2_1):
        return d2pdt2_0*(1-k) + d2pdt2_1*k

    def dpdt_(k, d2pdt2_1):
        return dpdt_0 + (d2pdt2_0*((1-(1-k)**2)/2) + d2pdt2_1*(k**2/2)) * dt

    def p_(k, d2pdt2_1):
        return p_0 + dpdt_0*(k*dt) + (d2pdt2_0*((3*k+(1-k)**3-1)/3) + d2pdt2_1*(k**3/3))*(dt**2/2)

    # Approximations at an intermediate time step `n+k`
    p_k = p_(k, d2pdt2_1=d2pdt2)
    dpdt_k = dpdt_(k, d2pdt2_1=d2pdt2)
    d2pdt2_k = d2pdt2_(k, d2pdt2_1=d2pdt2)
    d3pdt3_k = d3pdt3_(k, d2pdt2_1=d2pdt2)

    # Linearize $d^2/dt^2 (p^2) = 2 (p d^2p/dt^2 + (dp/dt)^2)$
    p_d2pdt2_k = p_0*d2pdt2_k + (p_k - p_0)*d2pdt2_0
    dpdt_dpdt_k = (2.0*dpdt_k - dpdt_0)*dpdt_0
    d2p2dt2_k = 2.0 * (p_d2pdt2_k + dpdt_dpdt_k)

    C0 = Constant((c*T)**2)
    C1 = Constant(1)
    C2 = Constant(δ / (c**2 * T))
    C3 = Constant(β / (c**2 * ρ))
    C4 = Constant(T**2)

    # Weak form
    F_k = (C0 * dpdn_bc * v * ds
         - C0 * inner(grad(p_k), grad(v)) * dx
         - C1 * d2pdt2_k * v * dx
         + C2 * d3pdt3_k * v * dx
         + C3 * d2p2dt2_k * v * dx
         + C4 * g * v * dx)

    a = lhs(F_k) # Bilinear form
    l = rhs(F_k) # Linear form

    # The unknown function at time step `n+1`
    d2pdt2 = Function(V, name="d2p/dt2|n+1")

    # Expressions dependent on `d2pdt2`
    dpdt_1 = dpdt_(1, d2pdt2_1=d2pdt2)
    p_1  = p_(1, d2pdt2_1=d2pdt2)
    
    # Assign initial conditions
    p.assign(p_ic)
    dpdt.assign(dpdt_ic)
    d2pdt2.assign(d2pdt2_ic)

    class Integrator:
        def __init__(self, t0, dt):
            self.t0 = t0
            self.dt = dt
            self.nt = 0

        @property
        def tn(self):
            return self.t0 + self.nt * self.dt

        def step(self):
            p_0.vector()[:] = p.vector()
            dpdt_0.vector()[:] = dpdt.vector()
            d2pdt2_0.vector()[:] = d2pdt2.vector()

            t_k = self.t0 + self.nt*self.dt + dt_k
            update_boundary_conditions_for_time(t_k)

            A, b = assemble_system(a, l, bcs)
            solve(A, d2pdt2.vector(), b)
            
            dpdt.assign(dpdt_1)
            p.assign(p_1)
            
            self.nt += 1

    integrator = Integrator(t0, dt)

    def plot_cross_section(p, yvalue=H*0.5):
        xs = np.linspace(0, L, 5*nx+1)
        ys = np.ones_like(xs) * yvalue
        ps = [p(x, y) for x, y in zip(xs, ys)]
        plt.plot(xs, ps, line_style)

    def plot_all():
        plt.figure('p')
        plot_cross_section(p, H*0.5)

        plt.figure('dp/dt')
        plot_cross_section(dpdt, H*0.5)

        plt.figure('d2p/dt2')
        plot_cross_section(d2pdt2, H*0.5)
    
    def progress_percent(i, nt, change=10):
        return int(i/nt * 100/change) * change

    def play():
        plt.close('animation')

        fig, ax = plt.subplots(3,1, num="animation", figsize = (6, 8), sharex="col", tight_layout=True)
        
        for ax_ in ax:
            ax_.clear()

        lines = []

        xs = np.linspace(0, L, 10*nx+1)
        ys = np.ones_like(xs) * (H * 0.5)

        zs = [p(x, y) for x, y in zip(xs, ys)]
        lines.append(ax[0].plot(xs, zs, '.-r'))
        
        zs = [dpdt(x, y) for x, y in zip(xs, ys)]
        lines.append(ax[1].plot(xs, zs, '.-b'))

        zs = [d2pdt2(x, y) for x, y in zip(xs, ys)]
        lines.append(ax[2].plot(xs, zs, '.-k'))

        ax[0].set_title(f"t = {0:.3e}")

        ax[0].set_ylabel('p')
        ax[1].set_ylabel('dp/dt')
        ax[2].set_ylabel('d2p/dt2')

        ax[0].set_ylim(np.array(ax[1].get_ylim())*0.5)
        ax[1].set_ylim(np.array(ax[1].get_ylim())*4)
        ax[2].set_ylim(np.array(ax[2].get_ylim())*40)

        # fig.tight_layout()
        plt.show()

        percent_old = -1

        for i in range(1, nt+1):
            integrator.step()

            percent = progress_percent(i, nt) 
            
            if percent > percent_old:
                print(f"[{percent:3}% ]")
                percent_old = percent

            if True:
                
                lines[0][0].set_ydata([p(x, y) for x, y in zip(xs, ys)])
                lines[1][0].set_ydata([dpdt(x, y) for x, y in zip(xs, ys)])
                lines[2][0].set_ydata([d2pdt2(x, y) for x, y in zip(xs, ys)])

                ti = i * dt
                ax[0].set_title(f"t = {ti:.3e}")
                plt.pause(0.01)

    play()

