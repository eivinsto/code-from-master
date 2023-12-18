import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.font_manager as font_manager

"""
This script contains a function that identifies the non-interacting
fermionic quasiparticle operators for a Kitaev chain using a method
that first goes through a set of Majorana operators and then 
transforms and combines these to form fermionic operators instead 
of transforming the fermion operators directly.

The function takes the following args:

N: Integer number of sites.
mu: Array containing confinement energies of sites.
t: Array containing hopping terms.
delta: Array containing superconding terms. 

And the following kwarg:

cutoff: Float used to determine allowed error thresholds. 
"""

def find_fermionic_operators(N,mu,t,delta, cutoff=1e-8):
    # Generate A
    A = np.zeros((2*N,2*N))       # Coefficient matrix (to be filled)
    for i in range(N):
        A[2*i, 2*i + 1] = -mu[i]/4
        A[2*i+1, 2*i] = mu[i]/4
        if i!=(N-1):
            A[2*i, 2*i + 3] = (t[i] + np.abs(delta[i]))/4
            A[2*i + 1, 2*i + 2] = -(t[i] - np.abs(delta[i]))/4
            A[2*i + 2, 2*i + 1] = (t[i] - np.abs(delta[i]) )/4
            A[2*i + 3, 2*i] = -(t[i] + np.abs(delta[i]))/4


    # Diagonalize A
    eigvals,eigvecs = np.linalg.eig(A)

    """
    Sort eigenvalues
    """

    site_numbers = np.arange(0,2*N)             # Numerator
    weights = np.dot(eigvecs,site_numbers)      # Weights based on distance between average occupancy


    # Identify paired eigenvectors
    sorted_pairs = np.zeros((N,2),dtype='int')      # Vector to contain indices of paired eigvals
    sorted_eigvals = np.zeros(1*N)                  # Vector to contain the sorted eigvals for fermionic modes
    i = 0                                           # Iterative index
    k = 0                                           # Second iterative index
    while k<N:
        # Check whether eigenvalue has been paired already
        check = True
        if i==0:
            # Vector is initialized with zeros
            check = False
        while check:
            if i in sorted_pairs:
                # Move to next if this is the case
                i += 1
            else:
                check=False

                # Check if index moved past bounds during previous check
            if i>=(2*N):
                break

        # Indices where matching eigenvalues are located
        idx = np.argwhere(np.abs(eigvals[i]+eigvals) < cutoff)[0]

        # Check if multiple if eigenvalue are identified
        if np.size(idx)>1:
            # Choose the minimum weight (closest average occupancy location)
            paired_idx = np.where(np.min(np.abs(weights[idx]-weights[i]))==np.abs(weights[idx]-weights[i]))
            # Pick one if multiple fulfill the requirement, otherwise removing nesting from np.where

            if paired_idx in sorted_pairs:
                print(f"Error at index %d"%(i))

            m = 0
            if np.size(paired_idx)>1:
                while True:
                    if paired_idx[m] not in sorted_pairs:
                        paired_idx = int(paired_idx[m])

                    m+=1

                    if m>=np.size(paired_idx):
                        print(r"Couldn't identify paired eigenvalue for index %d"%(i))
                        print("Exiting script.")
                        sys.exit()

            # Store indixes of pair
            sorted_pairs[k,0] = i
            sorted_pairs[k,1] = idx[paired_idx]

            # Store eigenvalues for final fermionic modes in a sorted array
            sorted_eigvals[k] = np.abs(eigvals[i])
        else:
            # Choose the one if only one

            # Store indixes of pair
            sorted_pairs[k,0] = i
            sorted_pairs[k,1] = int(idx)

            # Store eigenvalues for final fermionic modes in a sorted array
            sorted_eigvals[k] = np.abs(eigvals[i])

        i += 1
        k += 1

    """
    Transform A (and eigenvectors) to canonical form
    """

    W = np.zeros((2*N,2*N),dtype='complex') # Transformation matrix (to be set up)

    # Pre-calculate values to reduce flops
    real_factor = 1/np.sqrt(2)
    complex_factor = complex(0,1)*real_factor

    # Setting up transformation matrix
    for i in range(N):
        W[sorted_pairs[i,0],sorted_pairs[i,0]] = real_factor
        W[sorted_pairs[i,0],sorted_pairs[i,1]] = -complex_factor
        W[sorted_pairs[i,1],sorted_pairs[i,0]] = complex_factor
        W[sorted_pairs[i,1],sorted_pairs[i,1]] = -real_factor

    # Finding canonical A
    canonical_A = np.dot(W,np.dot(np.diag(eigvals),W))   # Canonical form of A


    for i in range(N):
        theta = np.pi/4 
        eigvecs[:,sorted_pairs[i,0]] *= np.exp(-complex(0,1)*np.real(theta))
        eigvecs[:,sorted_pairs[i,1]] *= np.exp(-complex(0,1)*np.real(theta))

        if np.imag(theta)>cutoff:
            print("WARNING: Phase fixing of eigenvectors has computed a non-zero imaginary part for the phase. Results should not be trusted.")


    final_operators = np.dot(W,np.linalg.inv(eigvecs))   # Transformed Majorana operators in Majorana basis

  

    """
    Transformation to fermionic operator basis
    """

    # Set up transformation matrix
    P = np.zeros((2*N,2*N),dtype='complex')
    for i in range(N):
        P[2*i,2*i] = -complex(0,1)
        P[2*i,2*i+1] = complex(0,1)
        P[2*i+1,2*i] = 1
        P[2*i+1,2*i+1] = 1


    # Majorana operators in fermionic operator basis
    maj_op_ferm_basis = np.zeros((2*N,2*N),dtype='complex')
    for i in range(2*N):
        for k in range(N):
            maj_op_ferm_basis[i,2*k] = final_operators[i,2*k+1] - complex(0,1)*final_operators[i,2*k]
            maj_op_ferm_basis[i,2*k+1] = final_operators[i,2*k+1] + complex(0,1)*final_operators[i,2*k]



    """
    Find fermionic operators
    """

    # New version

    fermionic_operators = np.zeros((N,2*N),dtype='complex')
    for i in range(N):
        # Add Majorana operators (c_i = gamma_2i+1 + i*gamma_2i)
        fermionic_operators[i,:] = maj_op_ferm_basis[sorted_pairs[i,0],:] + complex(0,1)*maj_op_ferm_basis[sorted_pairs[i,1],:]
        # Normalize
        fermionic_operators[i,:] *= 1/np.linalg.norm(fermionic_operators[i,:])


    # Check orthogonality
    check = False
    for i in range(N-1):
        for j in range(i,N):
            val = np.abs(np.sum(np.dot(np.conjugate(fermionic_operators[i,:]),fermionic_operators[j,:])))
            if i==j:
                if np.abs(val-1)>cutoff:
                    if not check:
                        print("Fermionic operator vectors are not orthogonal.")
                        check = True
                    print("Offending indices: ", i,j)
            else:
                if val > cutoff:
                    if not check:
                        print("Fermionic operator vectors are not orthogonal.")
                        check = True
                    print("Offending indices: ", i,j)

    if check:
        print('------------------------------')


    """
    Check fermionic anticommutation relations
    """


    # {d_i,d_j} = 0
    for i in range(N):
        for j in range(i,N):
            # Value of anticommutator
            val = np.abs(np.sum(fermionic_operators[i,::2]*fermionic_operators[j,1::2] + fermionic_operators[j,::2]*fermionic_operators[i,1::2]))
            if val>cutoff:
                print(i,j)
                check = True


    # {d_i^\dagger, d_j} = delta_ij
    check = False              # Error flag
    for i in range(N):
        for j in range(i,N):
            # Value of anticommutator
            val = np.abs(np.dot(np.conjugate(fermionic_operators[i,:]),fermionic_operators[j,:]))
            if i==j:
                if (val-1)>cutoff:
                    print("WARNING: Fermionic operator anticommutation relations, {d_i^\dagger, d_j} = delta_ij, are not satisfied.")

                    check = True
            else:
                if val>cutoff:
                    check = True

    if check:
        print("WARNING: Fermionic operator anticommutation relations are not satisfied")



    """
    Majorana anticommutation check
    """

    # {gamma_i^\dagger, gamma_j} = delta_ij
    check = False
    for i in range(2*N):
        for j in range(i,2*N):
            # Value of anticommutator
            val = np.abs(np.dot(final_operators[i,:],final_operators[j,:]))
            #print(val)
            if i==j:
                if np.abs((val-1))>cutoff:
                    check = True
            else:
                if val>cutoff:
                    check = True

    if check:
        print("WARNING: Majorana anticommutation relations in Majorana basis are not fulfilled.")


    # Same but for transformed operators
    check = False
    for i in range(2*N):
        for j in range(i,2*N):
            # Value of anticommutator
            val = np.abs(np.sum(maj_op_ferm_basis[i,::2]*maj_op_ferm_basis[j,1::2] + maj_op_ferm_basis[i,1::2]*maj_op_ferm_basis[j,::2]))
            if i==j:
                if np.abs((val-2))>cutoff:
                    check = True
            else:
                if val>cutoff:
                    check = True


    if check:
        print("WARNING: Majorana anticommutation relations in fermionic basis are not fulfilled.")



    """
    General tests
    """

    # Check that canonical form is achieved
    check = False           # Error flag
    for i in range(2*N):
        for j in range(2*N):
            # Check whether pair (i,j) are indices for one of the elements that are supposed to be non-zero
            condition = True
            for k in range(N):
                if np.logical_and(i in sorted_pairs[k,:], j in sorted_pairs[k,:]):
                    condition = False

            # If not check that they are withing numerical error from zero
            if condition:
                if np.abs(canonical_A[i,j])>cutoff:
                    check = True
    # Print only once to avoid spam, detailed testing can be added if necessary
    if check:
        print("WARNING: Canonical form not achieved.")


    # Locate lowest energy single-particle operator
    min_idx = np.ravel(np.argwhere(np.abs(sorted_eigvals)==np.min(np.abs(sorted_eigvals))))

    # Small test to see whether support for higher dimensionality needs to be added
    if np.size(min_idx)>1:
        print("Dimensionality error of minimum energy index. If the ground state is further degenerate then the script needs to be modified to fit this.")
        print("Indices with minimum energy (there should only be one): ",min_idx)
        print("Exiting script.")
        sys.exit()


    # Check reality of Majorana basis Majorana Transformation
    if np.any(np.abs(np.imag(final_operators))>cutoff):
        print("WARNING: Majorana transformation is not real")


    return fermionic_operators, min_idx, sorted_eigvals
