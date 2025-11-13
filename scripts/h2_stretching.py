"""H2 Potential Energy Surface via Shadow Tomography.

This script computes the potential energy surface (PES) of molecular hydrogen (H2)
as a function of internuclear distance using shadow state tomography. The results
are compared against exact Full Configuration Interaction (FCI) calculations to
demonstrate the accuracy and statistical properties of the shadow tomography method.

The workflow:
1. For each interatomic distance:
   - Build H2 molecule with specified geometry
   - Compute Hartree-Fock reference and molecular Hamiltonian
   - Run exact FCI calculation for ground truth
   - Perform multiple shadow tomography estimations
2. Compute mean and standard deviation of shadow estimates
3. Plot PES comparing shadow tomography vs exact FCI
4. Save results to disk for further analysis

References:
    - Huggins et al., "Virtual Distillation for Quantum Error Mitigation"
      Nature Physics (2021)
    - Aaronson & Gottesman, "Improved Simulation of Stabilizer Circuits"
      Phys. Rev. A 70, 052328 (2004)
"""

from pyscf import gto, scf
from shadow_ci.estimator import GroundStateEstimator
from shadow_ci.solvers import FCISolver
from shadow_ci.utils import make_hydrogen_chain
import numpy as np
import matplotlib.pyplot as plt
from plotting_config import setup_plotting_style, save_figure

# Simulation parameters
N_SAMPLES = 1000          # Number of shadow measurement samples per estimation
N_ESTIMATORS = 20         # Number of median-of-means estimators (k in paper)
N_SIMULATIONS = 100       # Number of independent runs for statistical analysis
N_HYDROGEN = 10           # Number of hydrogen atoms in chain

# Interatomic distances to sample (in Angstroms)
INTERATOMIC_DISTANCES = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25])

def main():
    """Run H2 stretching analysis and generate PES plot."""

    # Initialize arrays to store results for each interatomic distance
    exact_fci = np.empty_like(INTERATOMIC_DISTANCES)        # Exact FCI energies
    estimated_mean = np.empty_like(INTERATOMIC_DISTANCES)   # Mean shadow estimates
    estimated_std = np.empty_like(INTERATOMIC_DISTANCES)    # Std dev of shadow estimates

    print("=" * 80)
    print(f"H2 Potential Energy Surface via Shadow Tomography")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Samples per run:       {N_SAMPLES}")
    print(f"  - Median-of-means bins:  {N_ESTIMATORS}")
    print(f"  - Independent runs:      {N_SIMULATIONS}")
    print(f"  - Number of H atoms:     {N_HYDROGEN}")
    print(f"  - Distances (Å):         {INTERATOMIC_DISTANCES}")
    print("=" * 80)

    # Loop over each interatomic distance
    for j, d in enumerate(INTERATOMIC_DISTANCES):
        print(f"\n[{j+1}/{len(INTERATOMIC_DISTANCES)}] Processing distance = {d:.2f} Å")

        # Array to store energy estimates from independent shadow runs
        estimations = np.empty(N_SIMULATIONS)

        # Build H2 molecule at current geometry
        mol_string = make_hydrogen_chain(N_HYDROGEN, d)
        mol = gto.Mole()
        mol.build(atom=mol_string, basis="sto-3g")

        # Compute Hartree-Fock reference state
        mf = scf.RHF(mol)
        # print(f"  HF Energy: {mf.h.hf_energy:.8f} Ha")

        # Run exact FCI calculation for ground truth comparison
        fci_solver = FCISolver(mf)
        estimator = GroundStateEstimator(mf, solver=fci_solver, verbose=4)
        print(f"  FCI Energy (exact): {estimator.E_exact:.8f} Ha")

        # Perform N_SIMULATIONS independent shadow tomography runs
        print(f"  Running {N_SIMULATIONS} shadow estimations...")
        for i in range(N_SIMULATIONS):
            E, _, _, _ = estimator.estimate_ground_state(N_SAMPLES, N_ESTIMATORS, use_qualcs=True, n_jobs=12)
            estimations[i] = E

            # Progress indicator every 10 runs
            if (i + 1) % 1 == 0:
                print(f"    Completed {i+1}/{N_SIMULATIONS} runs")

        # Store exact and statistical results
        exact_fci[j] = estimator.E_exact
        estimated_mean[j] = np.mean(estimations)
        estimated_std[j] = np.std(estimations)

        # Print summary statistics for this distance
        error = estimated_mean[j] - exact_fci[j]
        print(f"  Shadow Mean:  {estimated_mean[j]:.8f} Ha")
        print(f"  Shadow Std:   {estimated_std[j]:.8f} Ha")
        print(f"  Mean Error:   {error:+.2e} Ha")

    # === Plotting Section ===
    print("\n" + "=" * 80)
    print("Generating Potential Energy Surface Plot")
    print("=" * 80)

    # Apply consistent plotting style
    setup_plotting_style()

    # Create figure with appropriate size for publication
    _, ax = plt.subplots(figsize=(6, 4))

    # Plot exact FCI curve
    ax.plot(INTERATOMIC_DISTANCES, exact_fci,
            'o-', label='Exact FCI', linewidth=2, markersize=6, color='C0')

    # Plot shadow estimates with error bars (1 standard deviation)
    ax.errorbar(INTERATOMIC_DISTANCES, estimated_mean, yerr=estimated_std,
                fmt='s--', label='Shadow Estimate', linewidth=1.5,
                markersize=5, capsize=4, capthick=1.5, color='C1', alpha=0.8)

    # Configure axis labels with LaTeX formatting
    ax.set_xlabel(r'Interatomic Distance (\AA)')
    ax.set_ylabel(r'Ground State Energy (Ha)')
    ax.set_title(f'H$_{{{N_HYDROGEN}}}$ Potential Energy Surface')

    # Add legend
    ax.legend(loc='best', framealpha=0.9)

    # Add grid for readability
    ax.grid(True, alpha=0.3)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save figure in multiple formats
    output_prefix = f"h2_stretching_N{N_HYDROGEN}"
    save_figure(f"{output_prefix}.pdf")
    save_figure(f"{output_prefix}.png", dpi=300)

    # Display the plot
    plt.show()

    # === Save numerical results ===
    print("\nSaving numerical results...")
    results = np.column_stack((INTERATOMIC_DISTANCES, exact_fci,
                               estimated_mean, estimated_std))
    header = "Distance(Å)  FCI_Energy(Ha)  Shadow_Mean(Ha)  Shadow_Std(Ha)"
    np.savetxt(f"{output_prefix}_data.txt", results,
               header=header, fmt='%.10f', delimiter='  ')
    print(f"Results saved to {output_prefix}_data.txt")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()