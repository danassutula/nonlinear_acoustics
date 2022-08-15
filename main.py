'''
The propagation of sound waves can be modeled by the Westervelt equation

(C0 ∇^2 - C5 d/dt - C1 d^2/dt^2 + C2 d^3/dt^3) p + C3 d^2/dt^2 p^2 + C4 g = 0,

where

C0 = c^2 T^2
C1 = 1
C2 = δ / c^2 / T 
C3 = β / c^2 / ρ
C4 = T^2
C5 = d T

and 

c - Speed of sound (constant) [m/s]
T - Time scaling constant (optional)
δ - Diffusivity of sound [m^2/s]
ρ - Density of medium [kg/m^3]
β - Nonlinearity coefficient [-]
d - Damping coefficient [1/s]

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

    space_time_refinement = 0
    extra_time_refinement = 0 # 0, 1, 2, ...
    line_style = ('.-', '-', '--', '-.', ':')[1]

    with_animation_1d = True
    with_animation_2d = False

    if with_animation_1d or with_animation_2d:
        require_solution_sampling = True
    else:
        require_solution_sampling = False

    # require_solution_sampling = True # override

    #
    # Usually need `extra_time_refinement > 0` when the non-linearity is big
    # 

    if 1:
        c = 1.0 # Speed of sound [m/s]
        ρ = 1.0 # Density of medium [kg/m^3]
        δ = 0.0 # Diffusivity of sound [m^2/s]
        β = 0.0 # Nonlinearity coefficient [-]
        d = 0.0 # Damping coefficient [1/s]

        if 1:
            β = 1e-4 # Scaled upto instability

    else:
        # Parameters of brain tissue
        c = 1500.0 # Speed of sound [m/s]
        ρ = 1100.0 # Density of medium [kg/m^3]
        δ = 4.5e-6 # Diffusivity of sound [m^2/s] (NOTE: An imperceivable effect)
        β = 6.0    # Nonlinearity coefficient [-]
        d = 0.0    # Damping coefficient [1/s]

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

    element_degree = 3

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
    boundary_markers.set_all(9000) # Some unique value

    for i in boundary_ids:
        boundary_subdomains[i].mark(boundary_markers, i)

    # Initial conditions
    p_ic = Function(V)
    dpdt_ic = Function(V)
    d2pdt2_ic = Function(V)

    # Dirichlet boundary conditions
    d2pdt2_bc = Expression('0.0', t=0, degree=0) 
    # Neumann boundary conditions
    dpdn_bc = Expression('0.0', t=0, degree=0) 

    # Dirichlet and Neumann BC's
    boundary_conditions_id = 2

    if boundary_conditions_id == 0:
        dirichlet_boundary_ids = (0, 1, 2, 3) # 0 - all edges fixed
    elif boundary_conditions_id == 1:
        dirichlet_boundary_ids = (0, 2)       # 1 - bottom and top edges fixed
    elif boundary_conditions_id == 2:
        dirichlet_boundary_ids = (1,)         # 2 - right edge fixed
    elif boundary_conditions_id == 3:
        dirichlet_boundary_ids = ()           # 3 - all edges free
    else:
        raise NotImplementedError
    
    bcs = []
    for i in dirichlet_boundary_ids:
        bcs.append(DirichletBC(V, d2pdt2_bc, boundary_markers, i))

    # Boundary id's not in `dirichlet_boundary_ids` must be in `neumann_boundary_ids`
    neumann_boundary_ids = tuple(set(boundary_ids).difference(dirichlet_boundary_ids))

    #
    # NOTE: Normaly, the source function is defined inside the domain, but
    # it is also possible to define it just on the boundary. In this case, 
    # the usual integration measure `dx` needs to be changed to `ds`.
    #

    source_function_id = 1
    source_boundary_ids = ()

    if source_function_id == 0:
        g = Expression('value', value=0.0, t=0, degree=0)
    elif source_function_id == 1:
        g = Expression(
            't < 2*pi/omega ? value * sin(omega*t) '
            '* pow(t * (2*pi/omega - t) * pow(omega/pi, 2), 2) : 0.0', # <- Smoothing weight
            L=Constant(L), H=Constant(H), t=0, omega=2*np.pi/(tn*0.2), value=1e4, degree=0)
        source_boundary_ids = (3,) # Boundary id where `g` is defined, i.e. left boundary
    elif source_function_id == 2:
        g = Expression(
            't < 2*pi/omega ? value * sin(pi*x[1]/H) * sin(omega*t) '
            '* pow(t * (2*pi/omega - t) * pow(omega/pi, 2), 2) : 0.0', # <- Smoothing weight
            L=Constant(L), H=Constant(H), t=0, omega=2*np.pi/(tn*0.2), value=1e4, degree=0)
        source_boundary_ids = (3,) # Boundary id where `g` is defined, i.e. left boundary
    else:
        raise NotImplementedError

    assert all(i in neumann_boundary_ids for i in source_boundary_ids)

    ds_neumann = ds(subdomain_id=neumann_boundary_ids, domain=mesh, subdomain_data=boundary_markers) 
    ds_source = ds(subdomain_id=source_boundary_ids, domain=mesh, subdomain_data=boundary_markers) 
    
    # Approximations at the previous time step `n`

    p_0 = Function(V)
    dpdt_0 = Function(V)
    d2pdt2_0 = Function(V)

    # Approximations at an intermediate time step `n+k`

    def d3pdt3_(k, d2pdt2_1):
        return (d2pdt2_1 - d2pdt2_0) / dt

    def d2pdt2_(k, d2pdt2_1):
        return d2pdt2_0*(1-k) + d2pdt2_1*k

    def dpdt_(k, d2pdt2_1):
        return dpdt_0 + (d2pdt2_0*((1-(1-k)**2)/2) + d2pdt2_1*(k**2/2)) * dt

    def p_(k, d2pdt2_1):
        return p_0 + dpdt_0*(k*dt) + (d2pdt2_0*((3*k+(1-k)**3-1)/3) + d2pdt2_1*(k**3/3))*(dt**2/2)

    # Approximation at `n+1`
    d2pdt2 = TrialFunction(V)

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
    C5 = Constant(d*T)

    v = TestFunction(V)

    # Weak form
    F_k = (C0 * dpdn_bc * v * ds_neumann
         - C0 * inner(grad(p_k), grad(v)) * dx
         - C5 * dpdt_k * v * dx
         - C1 * d2pdt2_k * v * dx
         + C2 * d3pdt3_k * v * dx
         + C3 * d2p2dt2_k * v * dx
         + C4 * g * v * ds_source)

    a = lhs(F_k) # Bilinear form
    l = rhs(F_k) # Linear form

    # Approximations at the next time step `n+1`

    p = Function(V)
    dpdt = Function(V)
    d2pdt2 = Function(V)

    p.assign(p_ic)
    dpdt.assign(dpdt_ic)
    d2pdt2.assign(d2pdt2_ic)
 
    dpdt_1 = dpdt_(1, d2pdt2_1=d2pdt2)
    p_1 = p_(1, d2pdt2_1=d2pdt2)
    
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
        def t(self):
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

    if require_solution_sampling:
        
        n_smp = min(100, nt)

        t_smp = np.zeros((n_smp,))
        x_smp = np.linspace(0, L, 4*nx+1)
        y_smp = np.ones_like(x_smp) * (H*0.5)

        p_smp_1d = np.zeros((n_smp, len(x_smp)))
        dpdt_smp_1d = np.zeros((n_smp, len(x_smp)))
        d2pdt2_smp_1d = np.zeros((n_smp, len(x_smp)))
        
        p_smp_2d = []
        
        i_smp = -1

        def sample_solutions(i):
            global i_smp
            if i_smp < int(i / nt * (n_smp-1)):
                i_smp += 1
                t_smp[i_smp] = integrator.t
                p_smp_1d[i_smp, :] = [p(x, y) for x, y in zip(x_smp, y_smp)]
                dpdt_smp_1d[i_smp, :] = [dpdt(x, y) for x, y in zip(x_smp, y_smp)]
                d2pdt2_smp_1d[i_smp, :] = [d2pdt2(x, y) for x, y in zip(x_smp, y_smp)]
                p_smp_2d.append(p.vector().get_local())

    else:
        def sample_solutions(*args, **kwargs):
            return

    report_interval = max(int(nt / 10), 1)

    if require_solution_sampling:
        sample_solutions(0)

    for i in range(1, nt+1):
        integrator.step()
        
        if i % report_interval == 0 or i == nt:
            print(f"[{int(i/nt*100):3}% ]")

        if np.isnan(d2pdt2.vector()[0]):
            print("SOLUTION DIVERGED")
            break

        if require_solution_sampling:
            sample_solutions(i)

    if require_solution_sampling:

        figname = ("animation-1D "
                f"[degree={element_degree}]"
                f"[space-time={space_time_refinement}]"
                f"[extra-time={extra_time_refinement}]"
                f"[k={k}]")

        def animate_1d(figname=figname, p=True, dpdt=True, d2pdt2=True, n_smp=i_smp+1):
            
            assert n_smp <= t_smp.size

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
                lines.append(ax[i].plot(x_smp, p_smp_1d[0, :], line_style+'r'))
                y_min, y_max = p_smp_1d[:n_smp,:].min(), p_smp_1d[:n_smp,:].max()
                y_mar = (y_max - y_min) * 0.05
                ax[i].set_ylim([y_min-y_mar, y_max+y_mar])
                ax[i].set_ylabel('p')
                i += 1
            if dpdt: 
                lines.append(ax[i].plot(x_smp, dpdt_smp_1d[0, :], line_style+'b'))
                y_min, y_max = dpdt_smp_1d[:n_smp,:].min(), dpdt_smp_1d[:n_smp,:].max()
                y_mar = (y_max - y_min) * 0.05
                ax[i].set_ylim([y_min-y_mar, y_max+y_mar])
                ax[i].set_ylabel('dp/dt')
                i += 1
            if d2pdt2:
                lines.append(ax[i].plot(x_smp, d2pdt2_smp_1d[0, :], line_style+'k'))
                y_min, y_max = d2pdt2_smp_1d[:n_smp,:].min(), d2pdt2_smp_1d[:n_smp,:].max()
                y_mar = (y_max - y_min) * 0.05
                ax[i].set_ylim([y_min-y_mar, y_max+y_mar])
                ax[i].set_ylabel('d2p/dt2')
                i += 1
            
            ax[0].set_title(f"t = {t_smp[0]:.3e}")
            plt.show()

            for i in range(1, n_smp):
                j = 0
                
                if p:
                    lines[j][0].set_ydata(p_smp_1d[i,:])
                    j += 1
                if dpdt:
                    lines[j][0].set_ydata(dpdt_smp_1d[i,:])
                    j += 1
                if d2pdt2:
                    lines[j][0].set_ydata(d2pdt2_smp_1d[i,:])

                ax[0].set_title(f"t = {t_smp[i]:.3e}")
                plt.pause(0.050)

        figname = ("animation-2D "
                f"[degree={element_degree}]"
                f"[space-time={space_time_refinement}]"
                f"[extra-time={extra_time_refinement}]"
                f"[k={k}]")

        def animate_2d(figname=figname, n_smp=i_smp+1):

            assert n_smp <= t_smp.size

            plt.close(figname)

            fig, ax = plt.subplots(1, 1, num=figname, tight_layout=True)
            
            fun = p.copy(deepcopy=True)

            for i in range(n_smp):
                ax.clear()
                ax.set_title(f"p({t_smp[i]:.3e})")
                ax.set_ylabel('y')
                ax.set_xlabel('x')
                
                fun.vector()[:] = p_smp_2d[i]
                plot(fun)
                plt.show()
                plt.pause(0.100)

    else:
        def animate_1d(*args, **kwargs):
            raise NotImplementedError("animate_1d")

        def animate_2d(*args, **kwargs):
            raise NotImplementedError("animate_2d")

    if with_animation_1d:
        animate_1d()

    if with_animation_2d:
        animate_2d()
