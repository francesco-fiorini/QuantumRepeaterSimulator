# TESQR Repeater Chain Simulator

This repository contains a **Python simulator** built with the [QuTiP library](http://qutip.org/) for modeling **Teleporation Enabled Superconducting Quantum Repeater (TESQR)** protocols in entanglement distribution links.

The simulator models realistic **hybrid quantum repeater links** that include:

- **Lossy optical fiber transmission**  
- **Noisy electro-optic (optical-to-microwave) transduction**  
- **Probabilistic coupling to superconducting matter qubits**  
- **Entanglement purification (BBPSSW protocol)** with multiple rounds  
- **Imperfect entanglement swapping** with noisy operations  

All physical operations are modeled using **Kraus operators**, **beam-splitter unitaries**, and **Monte Carlo sampling** to reproduce realistic experimental behavior.

---

## 📑 Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Architecture](#architecture)  
4. [Simulation Flow](#simulation-flow)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Output](#output)  
8. [License](#license)  

---

## Overview

This simulator provides a **complete end-to-end model** of a **TESQR segment**, capturing photon transmission, optical-to-microwave conversion, qubit coupling, purification processes and entanglement swapping under realistic noise.

It allows quantitative evaluation of:
- **Fidelity** of the final entangled qubits after purification and swapping  
- **Success probability** (heralding rate) of purification and swapping stages  
- **Effect of channel loss, thermal noise, and gate errors** on overall performance  

The code implements **Monte Carlo trajectory simulations** to sample stochastic loss and noise effects in hybrid photonic–microwave entanglement distribution.

---

## Features

✔️ **Photon loss channel** modeled via Kraus operators  
✔️ **Noisy optical–microwave transduction** using Gaussian-loss channels  
✔️ **Probabilistic coupling** between photonic and matter qubits  
✔️ **Multiple purification rounds** (BBPSSW-type) with realistic CNOT and measurement noise  
✔️ **Imperfect entanglement swapping** with depolarizing and gate noise  
✔️ **Monte Carlo averaging** over many trials for fidelity and success probability  
✔️ **3D visualization** of resulting density matrices with LaTeX-rendered plots  

---

## Architecture

The simulator is modular, composed of clearly separated stages:

| Module | Description |
|---------|-------------|
| **Loss and Transduction** | Defines Kraus operators for optical fiber loss and thermal transduction channels. |
| **State Generation** | Creates probabilistic photon–qubit entangled pairs. |
| **Coupling** | Couples the photonic mode with a superconducting matter qubit using a stochastic interaction model. |
| **Purification** | Implements the BBPSSW protocol with noisy CNOTs, measurement errors, and depolarizing noise. |
| **Swapping** | Performs noisy entanglement swapping and measurement on two purified pairs. |
| **Analysis & Visualization** | Computes fidelities and generates 3D plots of average density matrices. |

---

## Simulation Flow

1. **Source generation:** probabilistic emission of entangled photon–qubit states.  
2. **Fiber loss:** modeled by a set of Kraus operators with transmissivity `η_c`.  
3. **Transduction:** optical-to-microwave conversion with thermal environment noise (`n̄`, `η_t`).  
4. **Coupling:** interaction between photon and matter qubit with efficiency `η_qb`.  
5. **Purification rounds:** sequential BBPSSW purification with noisy CNOTs and measurements.  
6. **Entanglement swapping:** imperfect Bell-state measurement on two purified links.  
7. **Fidelity evaluation:** comparison between final qubit pair and ideal Bell states.  
8. **Plotting:** 3D visualization of real parts of the resulting density matrices.  

---

## Installation

### Prerequisites
- Python ≥ 3.8  
- [QuTiP](http://qutip.org/)  
- NumPy  
- Matplotlib  

### Installation via pip
```bash
pip install qutip numpy matplotlib
