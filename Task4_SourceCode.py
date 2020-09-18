import numpy as np
from random import random
from scipy.optimize import minimize

from qiskit import *
from qiskit.circuit.library.standard_gates import U2Gate
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import NumPyEigensolver
from qiskit.visualization import plot_histogram


'''
    Hamiltonian Matrix
'''

H = np.array([[ 1,  0,  0,  0],
       [ 0,  0, -1,  0],
       [ 0, -1,  0,  0],
       [ 0,  0,  0,  1]])


print ('\nHamiltonian:\n', H)
print ('\nEigen Values:', np.linalg.eig(H)[0])
print ('\nLowest Eigen Value:', min(np.linalg.eig(H)[0]))


'''
    Pauli Matrix Representation: Finding the Coefficient
    
        H = a*II + b*XX + c*YY + d*ZZ
        
        [a,b,c,d] <- PauliCoeff(H)
'''

def PauliCoeff(M):

    #   2*2 Pauli Matrices
    I = np.array ( [ [1,0], [0,1] ] )
    X = np.array ( [ [0,1], [1,0] ] )
    Y = np.array ( [ [0,complex(0,1)], [complex (0,-1),0] ] )
    Z = np.array ( [ [1,0],[0,-1] ] )

    #   4*4 Matrices (Tesor Product of 2*2 Pauli Matrices)
    II = np.kron(I,I)
    XX = np.kron(X,X)
    YY = np.kron(Y,Y)
    ZZ = np.kron(Z,Z)

    Pauli = [II,  XX, YY, ZZ]
    Coeff = []
    for P in Pauli:
        a = (1/4)* np.trace(np.matmul(M,np.conjugate(np.transpose(P)))) #Refer to Solution document a = (1/4) trace (M P*)
        Coeff.append(a)

    return Coeff


'''
    Quantum Circuits:
    
    |00> -- [Anstoz]--- [Quatum Mdule]---(Measurement)
    
'''

def quantum_circuit(parameters, Pauli_Matrix):

    qc = QuantumCircuit(2,2)

    #Anstoz
    #   |00>--[HI]--[CNOT]---[RX RX]---
    qc.h(0)
    qc.cx(0,1)
    qc.rx(parameters[0],0)
    qc.rx(parameters[0],1)

    #Gates
    '''
        XX: H * H
        YY: Y_gate * Y_gate
        ZZ: None
    '''
    
    if Pauli_Matrix == 'ZZ':
        qc.measure([0,1], [0,1])
        
    elif Pauli_Matrix == 'XX':
        qc.h(0)
        qc.h(1)
        qc.measure([0,1], [0,1])
        
    elif Pauli_Matrix == 'YY':
        qc.u2(0, np.pi/2, 0)
        qc.u2(0, np.pi/2, 1)
        qc.measure([0,1], [0,1])
    else:
        raise ValueError('Not valid input for measurement: input should be "X" or "Y" or "Z"')
    return qc


'''
    Computing Expectation of Individual Pauli Componenets
'''

def expectation_value(parameters, Pauli_Matrix):

    # measure
    if Pauli_Matrix == 'II':
        return 1
    elif Pauli_Matrix == 'ZZ':
        circuit = quantum_circuit(parameters, 'ZZ')
    elif Pauli_Matrix == 'XX':
        circuit = quantum_circuit(parameters, 'XX')
    elif Pauli_Matrix == 'YY':
        circuit = quantum_circuit(parameters, 'YY')
    else:
        raise ValueError('Not valid input for measurement: input should be "II" or "XX" or "ZZ" or "YY"')
    
    shots = 8192
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()

    
    # expectation value estimation from counts
    expectation_value = 0
    for measure_result in counts:
        sign = -1
        if measure_result == '11' or measure_result == '00':
            sign = +1
        expectation_value += sign * counts[measure_result] / shots
        
    return expectation_value

'''
    Computing Total Expectation Value
'''

def total_expectation (parameters):

    '''
        H = a*II + b*XX + c*YY + d*ZZ
        <H> = a*<II> + b*<XX> + c*<YY> + d*<ZZ>
    '''
    
    # Individual Expectations
    EX_I = H_Pauli['II'] * expectation_value(parameters, 'II')
    EX_X = H_Pauli['XX'] * expectation_value(parameters, 'XX')
    EX_Y = H_Pauli['YY'] * expectation_value(parameters, 'YY')
    EX_Z = H_Pauli['ZZ'] * expectation_value(parameters, 'ZZ')
    
    # Summing the results
    EX_Total = EX_I + EX_X + EX_Y + EX_Z

    return EX_Total




C = PauliCoeff(H)
print ('H = ', C[0],' II + ', C[1],' XX + ', C[2],' YY + ', C[3],' ZZ')

#   Pauli Representation of H

H_Pauli = {'II':C[0], 'XX':C[1], 'YY':C[2], 'ZZ':C[3]}

#H_Pauli = {'II':0.5, 'XX':-0.5, 'YY':-0.5, 'ZZ':0.5}

parameters_array = np.array([2*np.pi, 2*np.pi])

result = minimize(total_expectation, parameters_array, method="nelder-mead",options={'xatol': 1e-8, 'disp': True})

print('Lowest Eigen Value (Estimated ground state energy from VQE algorithm): {}'.format(result.fun))

print ('\nLowest Eigen Value (Using Classical Method):', min(np.linalg.eig(H)[0]))


