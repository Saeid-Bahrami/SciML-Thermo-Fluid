<div align="center">

<!-- Banner Image with Link -->
[![Real-Time SciML Surrogate Banner](banner2.jpg)](https://saeidbahrami.com/AI-CFDLab.php)

<!-- Badges -->
[![Live Interactive Demo](https://img.shields.io/badge/Live_Demo-saeidbahrami.com-0055ff?style=for-the-badge&logo=googlechrome&logoColor=white)](https://saeidbahrami.com/AI-CFDLab.php)
[![API Microservice](https://img.shields.io/badge/FastAPI-PyTorch%20CPU%20Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://saeidbahrami.com/AI-CFDLab.php)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

*Developed by **Saeid Bahrami Eynolghasi** | Computational Fluid Dynamics & SciML Researcher*

---

###  [Click Here to Run the Live Zero-Install Interactive CFD Simulation](https://saeidbahrami.com/AI-CFDLab.php)
*Explore real-time fluid velocity and temperature fields directly in your browser without local installation.*

---

</div>

##  Notice on Repository Scope & Live Testing

> **Test the Live Dashboard:** 
> You do not need to download this repository or install heavy software to test the model. Training the PyTorch model and generating the CFD dataset requires significant computing power. 
> 
> Instead, I have deployed the trained AI model on a cloud server using a **FastAPI** microservice. You can test the sub-second, real-time predictions directly in your browser:
> **[saeidbahrami.com/AI-CFDLab.php](https://saeidbahrami.com/AI-CFDLab.php)**
> 
> *This repository serves as documentation of software architecture, physics logic, and AI deployment.*

---

## 1. Project Summary

Traditional CFD solvers are highly accurate but too slow for real-time decisions, such as Model Predictive Control (MPC) in smart buildings. 

This project solves that problem. I connected a traditional fluid mechanics solver with a fast AI model called a **Fourier Neural Operator (FNO-2D)**. Built with **PyTorch** and served via **FastAPI**, this AI model instantly predicts the steady-state fluid velocity (Vx, Vy) and temperature (T) based on the room's boundary conditions.

```text
+--------------------------+     +----------------------------+     +-------------------------------+
|   Parametric CFD Solver  | --> |   PyTorch FNO-2D Pipeline  | --> | FastAPI CPU Cloud Microservice|
| (Pure Vectorized Python) |     |  (L2 Spatial Loss & Modes) |     |    (Real-Time Inference UI)   |
+--------------------------+     +----------------------------+     +-------------------------------+
```

---

## 2. Physics & Mathematics
 
To train the AI, I first wrote a custom 2D CFD solver to generate the data. The simulation models air and heat moving inside a 3.2m × 3.2m room on a 64 × 64 grid (Δx = Δy = 0.05m). The room has an air inlet, an exhaust, and a constant 37°C (310.15 K) heat source (representing a person or a machine).

Governing Equations
The fluid motion is governed by the incompressible Navier-Stokes equations coupled with heat transport using the Boussinesq approximation for natural convection:

Momentum Equations:

**Momentum Equations:**

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{g}\beta(T - T_{ref})$$

**Energy Equation:**

$$\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \alpha \nabla^2 T$$

**Continuity (Incompressibility Constraint):**

$$\nabla \cdot \mathbf{u} = 0$$

### Numerical Implementation (CFD Engine)
- **Time Integration & Advection:** Explicit forward-Euler integration using 1st-order Upwind differencing for advective terms and 2nd-order Central Differencing for diffusive terms.
- **Pressure-Velocity Coupling:** Modified Chorin's Projection Method with Rhie-Chow interpolation to prevent grid checkerboarding.
- **Stability Control:** Adaptive time-stepping enforcing Courant-Friedrichs-Lewy ($\text{CFL} \le 0.45$) and Fourier diffusion limits.

---

## 3. Scientific Machine Learning (SciML) Pipeline

### Data Generation & Latin Hypercube Sampling (LHS)
To train the AI, I needed a diverse dataset of different room conditions. I used **Latin Hypercube Sampling (LHS)** to create 500 unique boundary condition scenarios:
* **Inlet Velocity:** 0.5 m/s to 3.0 m/s
* **Inlet Temperature:** 288.15 K to 303.15 K

> **Why LHS?** 
> In Scientific Machine Learning, the quality of the dataset is just as important as the neural network. Unlike standard random sampling (which can leave "blind spots") or grid sampling (which is computationally expensive), LHS is a statistical method that guarantees the entire physical space is sampled evenly. This ensures the AI learns the fluid behavior across all possible scenarios without wasting computing power on redundant data.

### FNO Architecture & PyTorch TrainingData
* **Normalization:** Raw inputs and fields are scaled to a $[0, 1]$ range using Min-Max scaling to ensure stable gradient descent, then split into an 80/20 train/test ratio.
* **Spectral Modes & Channels:** The FNO model is configured with $16 \times 16$ Fourier modes to capture turbulent eddies and 64 hidden channels to resolve non-linear momentum-thermal cross-coupling.
 
* **Training Loop:** The network is trained for 100 epochs with a batch size of 16 using the Adam optimizer ($lr=1e^{-3}$).
* **Hidden Channels:** Expanded to 64 channels to capture the non-linear relationship between temperature and velocity.
* **Loss Function:** Integrated relative $L^2$ norm loss (LpLoss(d=2, p=2)) is used instead of standard MSE to preserve continuous physical field structures.
* **HPC Memory Safety:** HPC Memory Safety: The training loop utilizes non_blocking=True for fast GPU transfers and set_to_none=True for gradients to eliminate VRAM memory leaks.
* **Lightweight Export:** Saved the trained weights into a standard `.npz` file so they can run extremely fast on a simple CPU without memory leaks.

---

## 4. Software Architecture & Cloud Microservice

The microservice backend (`app.py`) is engineered for production-grade reliability:

1. **Zero Cold-Start:** Model weights are loaded into RAM during application lifespan boot, ensuring $0\text{ms}$ startup lag on incoming requests.
2. **Inference Acceleration:** Uses `@torch.inference_mode()` and vectorized tensor operations to eliminate autograd overhead.
3. **Cross-Origin Security:** Fully configured CORS middleware supporting asynchronous communication with the interactive frontend.

```text
       [ Client / Browser UI ]
                  |
    (POST /predict - Velocity & Temp)
                  |
                  v
       [ FastAPI Microservice ]  <--- Pre-loaded PyTorch Model in RAM
                  |
    (Inference Output: 3x64x64 Array)
                  |
                  v
       [ Real-time Field Render ]
```
   ---

## 5. Repository Structure

```text
├── ⚙️ SciML-Thermo-Fluid/            # THE ENGINE: Single Source of Truth
│   ├── generate_parametric_cfd.py 
│   └── README.md  
├── 📂 Archetypes/                    # THE DELIVERABLES: Micro-Projects & Live Deployments
│   │
│   ├──  HVAC_Surrogate_Project/      # Archetype: Neural surrogate for building energy MPC
│   │   ├── app.py                    # Server-Side Inference: FastAPI REST endpoint for real-time FNO evaluation
│   │   ├── Dockerfile                # Cloud Deployment: CPU-optimized Docker container (Render / Cloud-native)
│   │   ├── requirements.txt          # Production dependencies (FastAPI, Uvicorn, PyTorch CPU, NumPy)
│   │   ├── weights_real_v2.npz       # Pre-trained lightweight Fourier Neural Operator (FNO) weights
│   │   └── README.md                 # Architecture documentation & Live API Endpoint link                # Container Deployment Configuration
```

---

<div align="center">
  <h3>📬 Academic Collaborations & Full-Time Research Roles</h3>
  </div>
  <div>
  <p>My objective is to bridge advanced Computational Fluid Dynamics with strong engineering intuition, delivering creative and reproducible solutions for complex physical challenges.
    
 I am actively open to full-time academic research positions and collaborative roles within high-impact grant projects.</p>
</div>
    <div align="center">
  <p>
   • <a href="https://saeidbahrami.com"><b> Explore My CFD Portfolio & Lab </b></a> • <a href="mailto:mail@saeidbahrami.com"><b> Email Me </b></a>  
  </p>
</div>
