import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Dict, List
import os
# =============================================================================
# 1. DATA CONTRACTS & CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class CFDConfig:
    """Rigid Configuration for the 64x64 domain. Values cannot be changed during runtime."""
    nx: int = 64
    ny: int = 64
    dx: float = 0.05
    dy: float = 0.05
    
    # Fluid Properties (Air)
    nu: float = 1.5e-5      # Kinematic viscosity (m^2/s)
    alpha: float = 2.2e-5   # Thermal diffusivity (m^2/s)
    beta: float = 3.4e-3    # Thermal expansion coefficient (1/K)
    g: float = 9.81         # Gravity (m/s^2)
    T_ref: float = 293.15   # Reference temperature (20 C)
    
    # Boundary Temperatures
    T_block: float = 310.15 # Block temperature (37 C)
    
    # Topologies (Indices strictly defined)
    inlet_y_range: Tuple[int, int] = (50, 58)
    inlet_x: int = 0
    outlet_y_range: Tuple[int, int] = (5, 13)
    outlet_x: int = -1
    block_x_range: Tuple[int, int] = (27, 37)
    block_y_range: Tuple[int, int] = (0, 10)

@dataclass
class SimulationState:
    """Holds the spatial arrays for a single time step."""
    u: np.ndarray  # X-velocity (64x64)
    v: np.ndarray  # Y-velocity (64x64)
    p: np.ndarray  # Pressure (64x64)
    T: np.ndarray  # Temperature (64x64)

# =============================================================================
# 2. MOCK INTERFACES (To be implemented in future phases)
# =============================================================================
def m2_generate_topology_and_samples(config: CFDConfig, n_samples: int = 500) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Module 2: Generates the 2D topology masks and performs Latin Hypercube Sampling (LHS).
    This function is purely functional, containing zero side-effects or state mutations.
    
    Args:
        config (CFDConfig): Frozen dataclass containing spatial domain constraints.
        n_samples (int): Number of LHS parameters to sample. Default is 500.
        
    Returns:
        Tuple[Dict[str, np.ndarray], np.ndarray]: 
            - Dictionary of boolean masks for 'inlet', 'outlet', and 'block'.
            - Scaled LHS samples array of shape (n_samples, 2).
    """
    
    # ---------------------------------------------------------
    # 1. Topology Generation ($O(1)$ purely vectorized slicing)
    # ---------------------------------------------------------
    
    # Initialize zero-state (False) boolean masks for computational domain
    topology = {
        'inlet': np.zeros((config.ny, config.nx), dtype=bool),
        'outlet': np.zeros((config.ny, config.nx), dtype=bool),
        'block': np.zeros((config.ny, config.nx), dtype=bool)
    }

    # Map spatial constraints to arrays. 
    # Logic: y maps to axis 0 (rows), x maps to axis 1 (columns)
    
    # Apply inlet mapping (y-range slice on specified x column)
    topology['inlet'][config.inlet_y_range[0]:config.inlet_y_range[1], config.inlet_x] = True
    
    # Apply outlet mapping (y-range slice on specified x column)
    topology['outlet'][config.outlet_y_range[0]:config.outlet_y_range[1], config.outlet_x] = True
    
    # Apply obstacle block mapping (2D region slice)
    topology['block'][config.block_y_range[0]:config.block_y_range[1], config.block_x_range[0]:config.block_x_range[1]] = True

    # ---------------------------------------------------------
    # 2. Latin Hypercube Sampling (LHS) Generation
    # ---------------------------------------------------------
    
    # Parameter bounds constraint definition
    # Dim 0: Inlet Velocity bound [0.5, 3.0]
    # Dim 1: Inlet Temperature bound [288.15, 303.15]
    l_bounds = [0.5, 288.15]
    u_bounds = [3.0, 303.15]

    # Initialize standard QMC LHS sampler in 2D space
    sampler = qmc.LatinHypercube(d=2)
    
    # Generate raw statistical distributions in space [0, 1)
    raw_samples = sampler.random(n=n_samples)
    
    # Linearly scale samples to the physical constraints bounds
    scaled_samples = qmc.scale(raw_samples, l_bounds, u_bounds)
    
    return topology, scaled_samples



def m3_predictor_step(state: SimulationState, config: CFDConfig, dt: float, masks: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the explicit forward-Euler predictor step for momentum (u*, v*) and updates 
    the energy equation (T_new) using fully vectorized NumPy operations.
    """
    # 1. Maintain boundary conditions by duplicating the arrays initially
    u_star = state.u.copy()
    v_star = state.v.copy()
    T_new = state.T.copy()
    
    # Pre-extract constants to reduce overhead
    dx, dy = config.dx, config.dy
    dx2, dy2 = dx**2, dy**2
    
    # 2. Extract internal node slices (C = Center)
    # i represents Y-axis (rows), j represents X-axis (cols)
    u_C = state.u[1:-1, 1:-1]
    v_C = state.v[1:-1, 1:-1]
    T_C = state.T[1:-1, 1:-1]
    
    # Define spatial slices for directional numerical schemes
    # X-Axis shifted slices: Left (j-1) and Right (j+1)
    u_L = state.u[1:-1, :-2];  u_R = state.u[1:-1, 2:]
    v_L = state.v[1:-1, :-2];  v_R = state.v[1:-1, 2:]
    T_L = state.T[1:-1, :-2];  T_R = state.T[1:-1, 2:]
    
    # Y-Axis shifted slices: Down/Bottom (i-1) and Up/Top (i+1)
    u_D = state.u[:-2, 1:-1];  u_U = state.u[2:, 1:-1]
    v_D = state.v[:-2, 1:-1];  v_U = state.v[2:, 1:-1]
    T_D = state.T[:-2, 1:-1];  T_U = state.T[2:, 1:-1]
    
    # 3. Advection - 1st-Order Upwind Differencing
    # Utilizes np.where to conditionally select the backward/forward derivative based on local flow direction
    def upwind_x(phi_C, phi_L, phi_R, vel_C):
        return np.where(vel_C > 0, (phi_C - phi_L) / dx, (phi_R - phi_C) / dx)
        
    def upwind_y(phi_C, phi_D, phi_U, vel_C):
        return np.where(vel_C > 0, (phi_C - phi_D) / dy, (phi_U - phi_C) / dy)
    
    # Convective terms: (U . nabla) Phi = u * (dPhi/dx) + v * (dPhi/dy)
    adv_u = u_C * upwind_x(u_C, u_L, u_R, u_C) + v_C * upwind_y(u_C, u_D, u_U, v_C)
    adv_v = u_C * upwind_x(v_C, v_L, v_R, u_C) + v_C * upwind_y(v_C, v_D, v_U, v_C)
    adv_T = u_C * upwind_x(T_C, T_L, T_R, u_C) + v_C * upwind_y(T_C, T_D, T_U, v_C)

    # 4. Diffusion - 2nd-Order Central Differencing (Laplacian)
    def laplacian(phi_C, phi_L, phi_R, phi_D, phi_U):
        return (phi_L - 2.0 * phi_C + phi_R) / dx2 + (phi_D - 2.0 * phi_C + phi_U) / dy2

    lap_u = laplacian(u_C, u_L, u_R, u_D, u_U)
    lap_v = laplacian(v_C, v_L, v_R, v_D, v_U)
    lap_T = laplacian(T_C, T_L, T_R, T_D, T_U)
    
    # 5. Boussinesq Approximation (Buoyancy Term)
    # Added exclusively to the Y-momentum
    buoyancy = config.g * config.beta * (T_C - config.T_ref)
    
    # 6. Explicit Forward-Euler Integration
    # Apply to internal nodes [1:-1, 1:-1]
    u_star[1:-1, 1:-1] = u_C + dt * (-adv_u + config.nu * lap_u)
    v_star[1:-1, 1:-1] = v_C + dt * (-adv_v + config.nu * lap_v + buoyancy)
    T_new[1:-1, 1:-1]  = T_C + dt * (-adv_T + config.alpha * lap_T)
    
    # 7. Strictly enforce obstacle masking
    if 'block' in masks:
        block_mask = masks['block']
        u_star[block_mask] = 0.0
        v_star[block_mask] = 0.0
        T_new[block_mask] = config.T_block  # تثبیت دمای منبع در گام پیش‌بینی
    return u_star, v_star, T_new
def m4_poisson_and_correction(
    u_star: np.ndarray, 
    v_star: np.ndarray, 
    P_old: np.ndarray,   # <--- پارامتر جدید اضافه شد
    config: CFDConfig, 
    dt: float,
    masks: Dict[str, np.ndarray], 
    n_iters: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Module 4: Corrector Step with Rhie-Chow Collocated Grid Filtering.
    Eliminates checkerboard pressure oscillations for clean SciML data.
    """
    dx, dy = config.dx, config.dy
    
    # -------------------------------------------------------------------------
    # 1. Rhie-Chow Divergence Calculation
    # -------------------------------------------------------------------------
    
    # الف) محاسبه گرادیان‌های فشار در مرکز سلول‌ها (2dx)
    dPdx = np.zeros_like(P_old)
    dPdy = np.zeros_like(P_old)
    dPdx[:, 1:-1] = (P_old[:, 2:] - P_old[:, :-2]) / (2 * dx)
    dPdy[1:-1, :] = (P_old[2:, :] - P_old[:-2, :]) / (2 * dy)
    
    # ب) میانگین‌گیری سرعت‌های پیش‌بینی‌شده روی وجوه (Linear Interpolation)
    U_E = 0.5 * (u_star[1:-1, 1:-1] + u_star[1:-1, 2:])
    U_W = 0.5 * (u_star[1:-1, :-2] + u_star[1:-1, 1:-1])
    V_N = 0.5 * (v_star[1:-1, 1:-1] + v_star[2:, 1:-1])
    V_S = 0.5 * (v_star[:-2, 1:-1] + v_star[1:-1, 1:-1])
    
    # ج) اعمال تصحیح Rhie-Chow روی وجوه X
    avg_dPdx_E = 0.5 * (dPdx[1:-1, 1:-1] + dPdx[1:-1, 2:])
    avg_dPdx_W = 0.5 * (dPdx[1:-1, :-2] + dPdx[1:-1, 1:-1])
    compact_dPdx_E = (P_old[1:-1, 2:] - P_old[1:-1, 1:-1]) / dx
    compact_dPdx_W = (P_old[1:-1, 1:-1] - P_old[1:-1, :-2]) / dx
    
    U_E_corr = U_E - dt * (compact_dPdx_E - avg_dPdx_E)
    U_W_corr = U_W - dt * (compact_dPdx_W - avg_dPdx_W)
    
    # د) اعمال تصحیح Rhie-Chow روی وجوه Y
    avg_dPdy_N = 0.5 * (dPdy[1:-1, 1:-1] + dPdy[2:, 1:-1])
    avg_dPdy_S = 0.5 * (dPdy[:-2, 1:-1] + dPdy[1:-1, 1:-1])
    compact_dPdy_N = (P_old[2:, 1:-1] - P_old[1:-1, 1:-1]) / dy
    compact_dPdy_S = (P_old[1:-1, 1:-1] - P_old[:-2, 1:-1]) / dy
    
    V_N_corr = V_N - dt * (compact_dPdy_N - avg_dPdy_N)
    V_S_corr = V_S - dt * (compact_dPdy_S - avg_dPdy_S)
    
    # هـ) محاسبه دیورژانس نهایی بر اساس سرعت‌های تصحیح شده وجوه
    div = np.zeros_like(u_star)
    div[1:-1, 1:-1] = (U_E_corr - U_W_corr) / dx + (V_N_corr - V_S_corr) / dy
    
    # صفر کردن دیورژانس داخل بلوک مانع
    if 'block' in masks:
        div[masks['block']] = 0.0

 # 2 & 3. Pressure Poisson Equation (PPE) via Jacobi Iteration
    # -------------------------------------------------------------------------
    P = P_old.copy()  # <--- [اصلاحیه: استفاده از فشار گام قبل برای همگرایی سریع‌تر]
    pn = np.empty_like(P)
    
    dx2, dy2 = dx**2, dy**2
    denom = 2 * (dx2 + dy2)
    
    for _ in range(n_iters):
        pn = P.copy()
        P[1:-1, 1:-1] = ( (pn[1:-1, 2:] + pn[1:-1, :-2]) * dy2 + 
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx2 - 
                          (div[1:-1, 1:-1] / dt) * dx2 * dy2 ) / denom
        
        # Neumann BC
        P[:, -1] = P[:, -2]  
        P[:, 0]  = P[:, 1]   
        P[-1, :] = P[-2, :]  
        P[0, :]  = P[1, :]   
        
        # Dirichlet BC
        if 'outlet' in masks:
            P[masks['outlet']] = 0.0

    # -------------------------------------------------------------------------
    # 4. Velocity Correction (Projection)
    # -------------------------------------------------------------------------
    u_new = u_star.copy()
    v_new = v_star.copy()
    
    # گرادیان مرکزی (2dx) روی شبکه Collocated اعمال می‌شود (به دلیل فیلتر بالا، نوسان ایجاد نمی‌کند)
    u_new[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * (P[1:-1, 2:] - P[1:-1, :-2]) / (2 * dx)
    v_new[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * (P[2:, 1:-1] - P[:-2, 1:-1]) / (2 * dy)
    
    # -------------------------------------------------------------------------
    # 5. Boundary & Obstacle Enforcement
    # -------------------------------------------------------------------------
    wall_mask = np.ones_like(u_new, dtype=bool)
    wall_mask[1:-1, 1:-1] = False
    
    if 'inlet' in masks: wall_mask[masks['inlet']] = False
    if 'outlet' in masks: wall_mask[masks['outlet']] = False
        
    u_new[wall_mask] = 0.0
    v_new[wall_mask] = 0.0
    
    if 'block' in masks:
        u_new[masks['block']] = 0.0
        v_new[masks['block']] = 0.0
    # === [اصلاحیه حیاتی: اضافه شدن شرط مرزی خروج سیال] ===
    # اعمال گرادیان-صفر (Zero-Gradient) روی سرعت‌های خروجی
   # === [مکان جایگزینی: انتهای تابع m4_poisson_and_correction] ===
    # کد قبلی خود را که مربوط به outlet بود پاک کنید و این را قرار دهید:

    # === [اصلاحیه حیاتی: اضافه شدن شرط مرزی خروج سیال] ===
    # اعمال گرادیان-صفر (Zero-Gradient) فقط روی سلول‌های مربوط به خروجی (نه کل دیوار راست)
    if 'outlet' in masks:
        outlet_mask = masks['outlet']
        # استخراج اندیس‌های محور Y که فقط متعلق به دهانه خروجی هستند
        y_out = np.where(outlet_mask[:, -1])[0]
        u_new[y_out, -1] = u_new[y_out, -2]
        v_new[y_out, -1] = v_new[y_out, -2]

    return u_new, v_new, P
def m5_calculate_dt(state: SimulationState, config: CFDConfig, cfl_max: float = 0.45) -> float:
    """
    Calculates the maximum allowable time-step (dt) based on the CFL condition.
    
    Args:
        state: Current fluid state matrices (u, v, p, T).
        config: Simulation grid configuration (dx, dy).
        cfl_max: Target maximum Courant number.
        
    Returns:
        float: The adaptively computed maximum stable time-step.
    """
    eps = 1e-6
    
    # Extract the absolute maximum velocity components
    u_max = np.max(np.abs(state.u))
    v_max = np.max(np.abs(state.v))
    
    # Calculate stable time intervals for spatial dimensions
    dt_x = config.dx / (u_max + eps)
    dt_y = config.dy / (v_max + eps)
    dt_conv = cfl_max * min(dt_x, dt_y)
    
    # === NEW CODE: Calculate Diffusion stability limit (Fourier number < 0.25) ===
    # For explicit central-differencing diffusion
    dt_diff = 0.25 * min(config.dx**2, config.dy**2) / max(config.nu, config.alpha)
    
    # Limit dt by the most restrictive condition to guarantee stability
    dt = min(dt_conv, dt_diff)
    
    return float(dt)

def save_fno_dataset(all_inputs: list, all_fields: list) -> None:
    """
    Stacks arrays and serializes the dataset for Fourier Neural Operator (FNO) training.
    
    Args:
        all_inputs: List of shape (2, 64, 64) containing inlet masks.
        all_fields: List of shape (3, 64, 64) containing state fields (Vx, Vy, T).
    """
    # np.stack expands the dimension at axis=0, generating batch dimension N
    inputs_tensor = np.stack(all_inputs, axis=0)
    fields_tensor = np.stack(all_fields, axis=0)
    
    # Serialize stacked multidimensional tensors directly to disk
    np.save('hvac_inputs.npy', inputs_tensor)
    np.save('room_fields.npy', fields_tensor)
# =============================================================================
# 3. MAIN ORCHESTRATOR
# =============================================================================

def run_single_simulation(v_in: float, t_in: float, config: CFDConfig, masks: Dict[str, np.ndarray]) -> SimulationState:
    """Runs a single CFD simulation to pseudo-steady state."""
    # Initialize zero fields
    state = SimulationState(
        u=np.zeros((config.ny, config.nx)),
        v=np.zeros((config.ny, config.nx)),
        p=np.zeros((config.ny, config.nx)),
        T=np.ones((config.ny, config.nx)) * config.T_ref
    )
    
    # Boundary injection for this specific run
    state.u[masks['inlet']] = v_in
    state.T[masks['inlet']] = t_in
    state.T[masks['block']] = config.T_block
    
    # Main temporal loop (Simplified for Skeleton testing)
  # Main temporal loop 
    max_iter = 25000  # Increased for realistic convergence
    tolerance = 1e-3  # Convergence residual
    
    for step in range(max_iter):
        dt = m5_calculate_dt(state, config)
        u_star, v_star, T_new = m3_predictor_step(state, config, dt, masks)
        u_new, v_new, p_new = m4_poisson_and_correction(u_star, v_star,state.p, config, dt, masks)
        
        # === NEW CODE: Check for Steady-State Convergence ===
        # توجه: این بررسی باید "قبل" از آپدیت State انجام شود
        if step % 500 == 0 and step > 0:
            # Calculate maximum change in velocity field scaled by dt
            residual = np.max(np.abs(u_new - state.u)) / dt
            if residual < tolerance:
                # Converged
                # چاپ یک پیام کوتاه برای اطمینان از صحت عملکرد در حین اجرای ۵۰۰ نمونه مفید است
                print(f"          -> Converged at step {step} (Residual: {residual:.2e})")
                break

        # Update State
        state.u, state.v, state.p, state.T = u_new, v_new, p_new, T_new
        
        # === NEW CODE: Re-enforce constant boundary conditions at every step ===
        # === NEW CODE: Re-enforce constant boundary conditions at every step ===
        state.u[masks['inlet']] = v_in
        state.v[masks['inlet']] = 0.0
        state.T[masks['inlet']] = t_in
        state.T[masks['block']] = config.T_block
        
        # === [اصلاحیه فیزیکی: اجازه خروج حرارت از دهانه اگزاست] ===
        if 'outlet' in masks:
            y_out = np.where(masks['outlet'][:, -1])[0]
            state.T[y_out, -1] = state.T[y_out, -2]
            
    return state
# =============================================================================
# UNIT TEST / INTEGRATION BLOCK
# =============================================================================
if __name__ == "__main__":
    print(">>> Starting Automated CFD Data Generation for FNO...")
    
    # ۱. فراخوانی تنظیمات اولیه هندسه
    cfg = CFDConfig()
    
    # ۲. تولید ماسک‌ها و ۵۰۰ نمونه سرعت و دما (ماژول ۲)
    print(">>> Generating Latin Hypercube Samples...")
    masks, samples = m2_generate_topology_and_samples(cfg, n_samples=500)
    
    # ساخت دو لیست خالی برای جمع‌آوری داده‌های ۵۰۰ شبیه‌سازی
    all_inputs = []
    all_fields = []
    
    # ۳. حلقه اصلی برای اجرای ۵۰۰ شبیه‌سازی
    for i, (v_in, t_in) in enumerate(samples):
        print(f"Running simulation {i+1}/500: V_in = {v_in:.2f} m/s, T_in = {t_in:.2f} K")
        
        # اجرای شبیه‌سازی و دریافت حالت نهایی (ماژول‌های ۳ و ۴ و ۵ در دل این تابع کار می‌کنند)
        final_state = run_single_simulation(v_in=v_in, t_in=t_in, config=cfg, masks=masks)
        
        # ۴. قالب‌بندی ماتریس ورودی (Inputs) برای این شبیه‌سازی
        # ساخت یک ماتریس ۳ بعدی خالی با ابعاد (2, 64, 64)
        input_tensor = np.zeros((2, cfg.ny, cfg.nx))
        # کانال 0: قرار دادن سرعت فقط در سلول‌های ورودی (Inlet)
        input_tensor[0][masks['inlet']] = v_in
        # کانال 1: قرار دادن دما فقط در سلول‌های ورودی (Inlet)
        input_tensor[1][masks['inlet']] = t_in
        
        # اضافه کردن این ماتریس به لیست کل ورودی‌ها
        all_inputs.append(input_tensor)
        
        # ۵. قالب‌بندی ماتریس خروجی (Fields) برای این شبیه‌سازی
        # چسباندن ماتریس‌های U و V و T روی هم تا یک ماتریس با ابعاد (3, 64, 64) ساخته شود
        field_tensor = np.stack([final_state.u, final_state.v, final_state.T], axis=0)
        
        # اضافه کردن این ماتریس به لیست کل خروجی‌ها
        all_fields.append(field_tensor)
    
    # ۶. ذخیره‌سازی داده‌ها (ماژول ۵)
    print(">>> All 500 simulations completed. Saving dataset to disk...")
    # فراخوانی تابعی که در ماژول ۵ ساختید تا لیست‌ها را به فایل .npy تبدیل کند
    save_fno_dataset(all_inputs, all_fields)
    
    print(">>> SUCCESS: 'hvac_inputs.npy' and 'room_fields.npy' have been generated!")