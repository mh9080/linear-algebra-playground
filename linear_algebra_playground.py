Here’s the full code for linear_algebra_playground.py. Copy everything below:
Python"""
LINEAR ALGEBRA PLAYGROUND
Goal: Build the SKILL of turning math you already understand (from courses)
      into clean, working Python code + experiments + verification.

Focus areas we'll cover step by step (guided, one piece at a time):
- Symmetric matrices
- Eigendecomposition (eigenvalues + orthogonal eigenvectors)
- Reconstructing the original matrix (spectral theorem in code)
- Positive definite / semidefinite checks via eigenvalues
- Matrix rank
- Quadratic forms (x^T A x) and what positive definite really means
- Small experiments + "interview-style" thinking prompts

We'll go SLOW and guided. Run sections one at a time.
After each section, experiment, then tell me what you saw or want to change.

Run: python linear_algebra_playground.py
(You can also run sections by commenting/uncommenting or copying parts into a notebook)

Dependencies: numpy, matplotlib (optional for later viz)
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)  # cleaner printing

print("=" * 60)
print("LINEAR ALGEBRA PLAYGROUND — Let's build the skill together")
print("You know the theory. We turn it into code + experiments.")
print("=" * 60)

# ============================================================
# SECTION 1: Creating & Inspecting Matrices (numpy basics)
# ============================================================
print("\n" + "="*60)
print("SECTION 1: Creating symmetric matrices + quick inspection")
print("="*60)

# Create a simple symmetric matrix by hand (you already know what symmetric means)
A = np.array([
    [4.0, 2.0, 1.0],
    [2.0, 5.0, 3.0],
    [1.0, 3.0, 6.0]
])

print("Our starting matrix A:")
print(A)
print(f"\nShape: {A.shape}")
print(f"Is it square? {A.shape[0] == A.shape[1]}")

# Make a random symmetric matrix (common trick)
def make_random_symmetric(n=4, seed=42):
    np.random.seed(seed)
    M = np.random.randn(n, n)
    return (M + M.T) / 2   # force symmetry

B = make_random_symmetric(4)
print("\nRandom symmetric matrix B (4x4):")
print(B)

print("\n>>> EXPERIMENT 1:")
print("   - Change the seed or size in make_random_symmetric")
print("   - Print B and check a few off-diagonal pairs: B[i,j] should == B[j,i]")
print("   - What happens if you don't do (M + M.T)/2 ? Try it.")

# ============================================================
# SECTION 2: Eigendecomposition (the heart of symmetric matrices)
# ============================================================
print("\n" + "="*60)
print("SECTION 2: Eigendecomposition — eigenvalues + orthogonal eigenvectors")
print("="*60)

# This is the core thing you already understand from courses
eigvals_A, eigvecs_A = np.linalg.eigh(A)   # eigh = for symmetric/Hermitian matrices (more stable)

print("Eigenvalues of A (should be real for symmetric matrices):")
print(eigvals_A)

print("\nEigenvectors (columns of this matrix):")
print(eigvecs_A)

print("\n>>> KEY THING YOU ALREADY KNOW:")
print("   For symmetric matrices, eigenvectors are ORTHOGONAL.")
print("   We can check: Q^T @ Q should be close to Identity matrix")

Q = eigvecs_A
orthogonality_check = Q.T @ Q
print("\nQ^T @ Q (should be almost Identity):")
print(orthogonality_check)

print("\n>>> EXPERIMENT 2:")
print("   - Look at the eigenvalues. Are they all positive? (We'll use this soon)")
print("   - Try with matrix B (the random one). Are its eigenvectors orthogonal?")
print("   - What if the matrix is NOT symmetric? Try np.linalg.eig on a non-symmetric matrix.")

# ============================================================
# SECTION 3: Reconstruct the original matrix (spectral theorem)
# ============================================================
print("\n" + "="*60)
print("SECTION 3: Reconstruct A from eigenvalues + eigenvectors")
print("         A = Q @ Lambda @ Q^T     (this is the spectral theorem in code)")
print("="*60)

Lambda = np.diag(eigvals_A)          # diagonal matrix of eigenvalues
A_reconstructed = Q @ Lambda @ Q.T   # Q is orthogonal, so this works beautifully

print("Original A:")
print(A)
print("\nReconstructed A (should be almost identical):")
print(A_reconstructed)
print(f"\nMax difference: {np.max(np.abs(A - A_reconstructed)):.2e}")

print("\n>>> EXPERIMENT 3:")
print("   - Try reconstructing with matrix B too.")
print("   - Add a tiny bit of noise to A and see how reconstruction behaves.")
print("   - This reconstruction is why symmetric matrices are so nice to work with.")

# ============================================================
# SECTION 4: Positive Definite / Semidefinite via eigenvalues
# ============================================================
print("\n" + "="*60)
print("SECTION 4: Checking Positive Definite vs Semidefinite")
print("         (All eigenvalues > 0  → Positive Definite)")
print("         (All eigenvalues >=0  → Positive Semidefinite)")
print("="*60)

def check_positive_definite(eigenvalues, tol=1e-10):
    if np.all(eigenvalues > tol):
        return "POSITIVE DEFINITE"
    elif np.all(eigenvalues >= -tol):
        return "POSITIVE SEMIDEFINITE"
    else:
        return "INDEFINITE (has negative eigenvalues)"

status_A = check_positive_definite(eigvals_A)
print(f"Matrix A status: {status_A}")
print(f"Eigenvalues: {eigvals_A}")

# Let's make an example that is only semidefinite (singular)
C = np.array([
    [1, 1, 0],
    [1, 1, 0],
    [0, 0, 0]
])
eigvals_C, _ = np.linalg.eigh(C)
print(f"\nSingular matrix C status: {check_positive_definite(eigvals_C)}")
print(f"Eigenvalues of C: {eigvals_C}")

print("\n>>> EXPERIMENT 4:")
print("   - Create your own matrix with a negative eigenvalue and check it.")
print("   - What does it mean practically if a covariance matrix has a negative eigenvalue?")
print("   - Try the random matrix B — is it positive definite?")

# ============================================================
# SECTION 5: Matrix Rank
# ============================================================
print("\n" + "="*60)
print("SECTION 5: Matrix Rank (how many independent directions?)")
print("="*60)

rank_A = np.linalg.matrix_rank(A)
rank_C = np.linalg.matrix_rank(C)

print(f"Rank of A (full rank expected): {rank_A}")
print(f"Rank of C (should be less than 3 because of the zero row/column): {rank_C}")

print("\n>>> EXPERIMENT 5:")
print("   - For matrix C, notice that rank + nullity = number of columns (rank-nullity theorem)")
print("   - How does rank relate to the number of non-zero eigenvalues?")
print("   - Try making a matrix with duplicate rows and check its rank.")

# ============================================================
# SECTION 6: Quadratic Forms (x^T A x) — what positive definite really feels like
# ============================================================
print("\n" + "="*60)
print("SECTION 6: Quadratic Forms — x^T A x  (the heart of positive definite)")
print("="*60)

def quadratic_form(x, matrix):
    return x.T @ matrix @ x

# Test with a few vectors
x1 = np.array([1.0, 0.0, 0.0])
x2 = np.array([1.0, 1.0, 1.0])
x3 = np.array([0.5, -1.0, 2.0])

print(f"For matrix A (which is positive definite):")
print(f"  x1^T A x1 = {quadratic_form(x1, A):.4f}  (>0)")
print(f"  x2^T A x2 = {quadratic_form(x2, A):.4f}  (>0)")
print(f"  x3^T A x3 = {quadratic_form(x3, A):.4f}  (>0)")

print(f"\nFor matrix C (positive semidefinite but singular):")
print(f"  x1^T C x1 = {quadratic_form(x1, C):.4f}")
print(f"  Some directions give exactly 0 (along the null space)")

print("\n>>> EXPERIMENT 6 (important!):")
print("   - Try a vector where quadratic_form is negative for an indefinite matrix.")
print("   - Positive definite means the quadratic form is ALWAYS positive for x ≠ 0.")
print("   - This is exactly why we care in optimization, covariance, kernels, etc.")

# ============================================================
# FINAL: Small challenges to build the "interview skill"
# ============================================================
print("\n" + "="*60)
print("SMALL CHALLENGES — Practice turning math into code + thinking")
print("="*60)
print("""
1. Write a small function `is_orthogonal(Q)` that checks if Q^T @ Q ≈ I
2. Create a matrix that is symmetric but NOT positive definite. Show its eigenvalues.
3. Take matrix A, change one off-diagonal value to break symmetry slightly.
   What happens to the eigenvalues? (they can become complex)
4. For a positive definite matrix, the quadratic form x^T A x > 0 for all x ≠ 0.
   Can you find the minimum value of the quadratic form? (hint: related to smallest eigenvalue)
5. Bonus thinking: Why do we prefer eigh() over eig() for symmetric matrices?

When you're ready, tell me:
- What did you notice / experiment with?
- Which part felt most useful?
- Want to add visualizations next? (eigenvectors, quadratic form surface, etc.)
- Ready for the next concept (or go back and deepen something)?
""")

print("\n" + "="*60)
print("END OF CURRENT PLAYGROUND")
print("Run the script, play with the experiments, then reply with what you saw!")
print("We'll add the next piece together (more on PSD, or visualizations, or bridge to stats).")
print("="*60)
