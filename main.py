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

    space_time_refinement = 1
    extra_time_refinement = 0 # 0, 1, 2, ...
    line_style = ('.-', '-', '--', '-.', ':')[1]

    #
    # Usually need `extra_time_refinement > 0` when the non-linearity is big
    # 

    if 1:
        # Test
        c = 1.0 # Speed of sound [m/s]
        ρ = 1.0 # Density of medium [kg/m^3]
        δ = 0.0 # Diffusivity of sound [m^2/s]
        β = 0.0 # Coefficient of nonlinearity [-]

        # NOTE: Seems like `δ` eventually causes instabilities in 1D problems.

        if 1:
            β = 1e-4 # Scaled upto instability

    else:
        # Parameters of brain tissue
        c = 1500.0 # Speed of sound [m/s]
        ρ = 1100.0 # Density of medium [kg/m^3]
        δ = 4.5e-6 # Diffusivity of sound [m^2/s] (NOTE: An imperceivable effect)
        β = 6.0    # Coefficient of nonlinearity [-]

        if 1:
            β = 1e12 # Scaled upto instability
            # NOTE: Need `space_time_refinement = 2`

    # Time scaling constant: `t_real [s] = T [s] * t [-]`
    # (for example, time for a wave to do a round-trip)
    T = 2.0 * max(L, H) / c

    ne = 16 * (2 ** space_time_refinement)
    nx = ne
    ny = ne

    hx = L / nx
    hy = H / ny
    he = min(hx, hy)

    element_degree = 2

    t0 = 0.0
    tn = 1.0

    # Time-step fraction (i.e. solve the discrete system at time `n+k`)
    k = 1.0 # 0.5 <= k <= 1.0 (k > 1.0 promotes extra stability)

    if element_degree == 1:
        time_step_factor = k**2 * 0.80
    elif element_degree == 2:
        time_step_factor = k**2 * 0.40
    elif element_degree == 3:
        time_step_factor = k**2 * 0.20
    elif element_degree == 4:
        time_step_factor = k**2 * 0.10
    else:
        time_step_factor = k**2 * 0.05

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

    boundary_markers = MeshFunction('size_t', mesh, DIM-1)
    boundary_ids = tuple(range(len(boundary_subdomains)))

    for i in boundary_ids:
        boundary_subdomains[i].mark(boundary_markers, i)

    # Initial conditions
    p_ic = Function(V, name="p|0")
    dpdt_ic = Function(V, name="dpdt|0")
    d2pdt2_ic = Function(V, name="d2pdt2|0")

    # Dirichlet boundary conditions
    d2pdt2_bc = Expression('0.0', t=0, degree=0) 

    # Neumann boundary conditions
    dpdn_bc = Expression('0.0', t=0, degree=0) 

    # test_case = 0 # 'all_edges_fixed'
    # test_case = 1 # 'bottom_top_edges_fixed_and_left_right_edges_zero_neumann' 
    # test_case = 2 # 'left_edge_fixed'
    test_case = 3 # 'all_edges_free'

    if test_case == 0:
        dirichlet_boundary_ids = (0, 1, 2, 3)
    elif test_case == 1:
        dirichlet_boundary_ids = (0, 2)
    elif test_case == 2:
        dirichlet_boundary_ids = (3,)
    elif test_case == 3:
        dirichlet_boundary_ids = ()
    else:
        raise NotImplementedError
    
    neumann_boundary_ids = tuple(set(boundary_ids).difference(dirichlet_boundary_ids))
    ds = ds(subdomain_id=neumann_boundary_ids, domain=mesh, subdomain_data=boundary_markers) 

    ds_source = ds(subdomain_id=(3,)) 
    
    source_function_id = 1

    if source_function_id == 0:
        g = Expression('value', value=0.0, t=0, degree=0)
    elif source_function_id == 1:
        g = Expression(
            't < 2*pi/omega ? value * sin(omega*t) * pow(t * (2*pi/omega - t) * pow(omega/pi, 2), 2) : 0.0',
            L=Constant(L), H=Constant(H), t=0, omega=2*np.pi/0.2, value=1e4, degree=0)
    else:
        raise NotImplementedError

    bcs = []
    for i in dirichlet_boundary_ids:
        bcs.append(DirichletBC(V, d2pdt2_bc, boundary_markers, i))

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

    d2pdt2 = TrialFunction(V) # "d2pdt2|n+1" 
    d3pdt3_k = d3pdt3_(k, d2pdt2_1=d2pdt2)
    d2pdt2_k = d2pdt2_(k, d2pdt2_1=d2pdt2)
    dpdt_k = dpdt_(k, d2pdt2_1=d2pdt2)
    p_k = p_(k, d2pdt2_1=d2pdt2)

    # Linearize $d^2/dt^2 (p^2) = 2 (p d^2p/dt^2 + (dp/dt)^2)$
    p_d2pdt2_k = p_0*d2pdt2_k + (p_k - p_0)*d2pdt2_0
    dpdt_dpdt_k = (2.0*dpdt_k - dpdt_0)*dpdt_0
    d2p2dt2_k = 2.0 * (p_d2pdt2_k + dpdt_dpdt_k)

    C0 = Constant((c*T)**2)
    C1 = Constant(1)
    C2 = Constant(δ / (c**2 * T))
    C3 = Constant(β / (c**2 * ρ))
    C4 = Constant(T**2)

    v = TestFunction(V)

    # Weak form
    F_k = (C0 * dpdn_bc * v * ds
         - C0 * inner(grad(p_k), grad(v)) * dx
         - C1 * d2pdt2_k * v * dx
         + C2 * d3pdt3_k * v * dx
         + C3 * d2p2dt2_k * v * dx
         + C4 * g * v * ds_source)

    a = lhs(F_k) # Bilinear form
    l = rhs(F_k) # Linear form

    p = Function(V, name="p|n+1")
    dpdt = Function(V, name="dp/dt|n+1")
    d2pdt2 = Function(V, name="d2p/dt2|n+1")

    dpdt_1 = dpdt_(1, d2pdt2_1=d2pdt2)
    p_1 = p_(1, d2pdt2_1=d2pdt2)
    
    # Assign initial conditions
    p.assign(p_ic)
    dpdt.assign(dpdt_ic)
    d2pdt2.assign(d2pdt2_ic)

    def update_dirichlet_bcs(t):
        d2pdt2_bc.t = t

    def update_neumann_bcs(t):
        dpdn_bc.t = t

    def update_source_term(t):
        g.t = t
        
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

            tn = self.t0 + self.nt * self.dt
            
            update_dirichlet_bcs(tn + dt)
            update_neumann_bcs(tn + dt_k)
            update_source_term(tn + dt_k)

            A, b = assemble_system(a, l, bcs)
            solve(A, d2pdt2.vector(), b)

            dpdt.assign(dpdt_1)
            p.assign(p_1)
            
            self.nt += 1

    integrator = Integrator(t0, dt)

    #
    # Sample points for cross-section animation
    #

    n_smp = min(100, nt)
    t_smp = np.zeros((n_smp,))

    x_smp = np.linspace(0, L, 4*nx+1)
    y_smp = np.ones_like(x_smp) * (H*0.5)

    # Sampled values over time
    p_smp = np.zeros((n_smp, len(x_smp)))
    dpdt_smp = np.zeros((n_smp, len(x_smp)))
    d2pdt2_smp = np.zeros((n_smp, len(x_smp)))

    def save_sampled_solutions(i_smp):
        p_smp[i_smp, :] = [p(x, y) for x, y in zip(x_smp, y_smp)]
        dpdt_smp[i_smp, :] = [dpdt(x, y) for x, y in zip(x_smp, y_smp)]
        d2pdt2_smp[i_smp, :] = [d2pdt2(x, y) for x, y in zip(x_smp, y_smp)]

    i_smp = 0
    t_smp[i_smp] = integrator.tn
    save_sampled_solutions(i_smp)
    
    report_period = max(int(nt / 10), 1)

    for i in range(1, nt+1):
        integrator.step()
        
        if i % report_period == 0 or i == nt:
            print(f"[{int(i/nt*100):3}% ]")

        if np.isnan(d2pdt2.vector()[0]):
            print("DIVERGED")
            n_smp = i_smp + 1
            break

        if i_smp < int(i / nt * (n_smp-1)):
            i_smp += 1
            t_smp[i_smp] = integrator.tn
            save_sampled_solutions(i_smp)
    
    def animate_1d(n_smp=n_smp, p=True, dpdt=True, d2pdt2=True, figname="animation (1D)"):

        plt.close(figname)

        num_plots = int(p) + int(dpdt) + int(d2pdt2)

        if num_plots == 0:
            return

        fig, ax = plt.subplots(
            num_plots, 1, num=figname, figsize = (6, 8), sharex="col", tight_layout=True)
        
        for ax_ in ax:
            ax_.clear()

        lines = []

        i = 0
        if p:
            lines.append(ax[i].plot(x_smp, p_smp[0, :], line_style+'r'))
            y_min, y_max = p_smp[:n_smp,:].min(), p_smp[:n_smp,:].max()
            y_mar = (y_max - y_min) * 0.05
            ax[i].set_ylim([y_min-y_mar, y_max+y_mar])
            ax[i].set_ylabel('p')
            i += 1
        if dpdt: 
            lines.append(ax[i].plot(x_smp, dpdt_smp[0, :], line_style+'b'))
            y_min, y_max = dpdt_smp[:n_smp,:].min(), dpdt_smp[:n_smp,:].max()
            y_mar = (y_max - y_min) * 0.05
            ax[i].set_ylim([y_min-y_mar, y_max+y_mar])
            ax[i].set_ylabel('dp/dt')
            i += 1
        if d2pdt2:
            lines.append(ax[i].plot(x_smp, d2pdt2_smp[0, :], line_style+'k'))
            y_min, y_max = d2pdt2_smp[:n_smp,:].min(), d2pdt2_smp[:n_smp,:].max()
            y_mar = (y_max - y_min) * 0.05
            ax[i].set_ylim([y_min-y_mar, y_max+y_mar])
            ax[i].set_ylabel('d2p/dt2')
            i += 1
        
        ax[0].set_title(f"t = {t_smp[0]:.3e}")
        plt.show()

        for i in range(1, n_smp):
            j = 0
            
            if p:
                lines[j][0].set_ydata(p_smp[i,:])
                j += 1
            if dpdt:
                lines[j][0].set_ydata(dpdt_smp[i,:])
                j += 1
            if d2pdt2:
                lines[j][0].set_ydata(d2pdt2_smp[i,:])

            ax[0].set_title(f"t = {t_smp[i]:.3e}")
            plt.pause(0.050)

    figname = ("animation (1D) "
              f"[degree={element_degree}]"
              f"[space-time={space_time_refinement}]"
              f"[extra-time={extra_time_refinement}]"
              f"[k={k}]")

    animate_1d(d2pdt2=True, figname=figname)
