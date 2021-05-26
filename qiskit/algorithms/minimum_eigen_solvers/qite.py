# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The QITE algorithm

"""

from typing import Optional, List, Callable, Union, Dict
import logging
from time import time
import numpy as np

from qiskit import ClassicalRegister,QuantumRegister, QuantumCircuit, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.opflow import (
    OperatorBase,
    ExpectationBase,
    ExpectationFactory,
    StateFn,
    CircuitStateFn,
    ListOp,
    I,
    CircuitSampler,
)
from qiskit.opflow.gradients import GradientBase
from qiskit.utils.validation import validate_min
from qiskit.utils.backend_utils import is_aer_provider
from qiskit.utils.quantum_instance import QuantumInstance
from ..optimizers import Optimizer, SLSQP
from ..variational_algorithm import VariationalAlgorithm, VariationalResult
from .minimum_eigen_solver import MinimumEigensolver, MinimumEigensolverResult
from ..exceptions import AlgorithmError
from .vqe import VQEResult

# Own imports
from qiskit.circuit.library.standard_gates.rz import CRZGate, RZGate
from qiskit.circuit.library.standard_gates.z import CZGate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library.standard_gates.rx import RXGate
from qiskit.circuit.library.standard_gates.x import CXGate
from qiskit.circuit import Qubit
from copy import deepcopy

logger = logging.getLogger(__name__)

# disable check for ansatzes, optimizer setter because of pylint bug
# pylint: disable=no-member


class VQITE(VariationalAlgorithm, MinimumEigensolver):
    r"""The Variational QITE algorithm

    """

    def __init__(
        self,
        ansatz: Optional[QuantumCircuit] = None,
        initial_point: Optional[np.ndarray] = None,
        max_iter: Optional[int] = 1,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ) -> None:
        """

        Args:
            ansatz: A parameterized circuit used as Ansatz for the wave function.
            initial_point: An optional initial point (i.e. initial parameter values)
            mas_iter: Maximal number of iteration to optimize the parameters
            quantum_instance: Quantum Instance or Backend
        """
        if ansatz is None:
            ansatz = RealAmplitudes()

        # set the initial point to the preferred parameters of the ansatz
        if initial_point is None and hasattr(ansatz, "preferred_init_points"):
            initial_point = ansatz.preferred_init_points

        num_parameters = len(initial_point)

        self._max_iter = max_iter
        self._num_parameters = num_parameters
        self._ansatz = ansatz
        self._initial_point = initial_point
        self._quantum_instance = quantum_instance

        self._ret = VQEResult()

    def construct_circuit(
        self,
        parameter: Union[List[float], List[Parameter], np.ndarray],
        operator: OperatorBase,
    ) -> List[QuantumCircuit]:
        """Return the circuits used to compute the expectation value.

        Args:
            parameter: Parameters for the ansatz circuit.
            operator: Qubit operator of the Observable

        Returns:
            A list of the circuits used to compute the expectation value.
        """

        # Construct derivative circuits -----------------------

        # Copy ansatz and add additional register to evaluate 
        # and perform hadamard test
        ansatz = deepcopy(self._ansatz)
        qr_eval = QuantumRegister(1, 'eval')
        ansatz.add_register(qr_eval)

        # add H gate and set to the beginning
        ansatz.h(qr_eval[0])
        gate = ansatz.data.pop()
        ansatz.data.insert(0,gate)

        i_max = self._num_parameters

        # Create structure to store circuits
        circuits_diff = []
        for idx in range(i_max):
            circuits_diff.append([])


        # Construct derivative circuits and add to list
        for i in range(i_max):
            for gate_idx, gate in enumerate(ansatz.data):
                if isinstance(gate[0],RZGate) or isinstance(gate[0],RXGate):
                
                    # Get factor before param
                    angle = str(gate[0].params[0]) # String
                    factor = float(angle[:-5])
                    # Get paramter index
                    param_idx = int(angle[-2])
            #         print(param_idx)

                    if param_idx == i:  
                    
                        # Get register and qubit position
                        qubit = gate[1][0]

                        # Create new gate
                        if isinstance(gate[0],RZGate):
                            sigma_gate = (CZGate(), [Qubit(qr_eval,0), qubit],[])
                        else:
                            sigma_gate = (CXGate(), [Qubit(qr_eval,0), qubit],[])



                        # Add gate to circuit data
                        circuit_temp = ansatz.copy()
                        data = circuit_temp.data
                        data.insert(gate_idx + 1, sigma_gate)


                        circuits_diff[i].append(circuit_temp)


        # Construct circuits to evaluate A_ij ---------------------
        circuits_A = []
        for idx in range(i_max):
            circuits_A.append([])
            for idx2 in range(i_max):
                circuits_A[idx].append([])



        for j in range(i_max):
            for i in range(j):
                for circ_i in circuits_diff[i]:
                    for circ_j in circuits_diff[j]:
                        new_circ = circ_i + circ_j.inverse()

                        # Transpile
                        circ_trans = self.quantum_instance.transpile(new_circ)
                        circuits_A[i][j].append(circ_trans)
        
        self._circuits_A = circuits_A

        # Construct circuits to evaluate C_i ---------------------
        op = self._operator

        # offset theta (fixed)
        theta_offset = np.pi/2        

        # Create circuit for Ci
        circuits_C = []
        for idx in range(i_max):
            circuits_C.append([])
            
        coeffs = []
        for idx in range(i_max):
            coeffs.append([])

        # order of the circuit_C
        # circuit[i]: circuit for i-th parameter order first regarding h_k and then the appearence in ansatz

        # Construct circuits and store in list    
        for i in range(i_max):
            for circ_i in circuits_diff[i]:
                for pauli in op:
                    # Add Hamiltonian h_k
                    coeff = pauli.coeff
                    coeffs[i].append(coeff)
                    print(coeff)
                    
                    circ_temp = circ_i.copy()
                    
                    # Add theta phase
                    theta_gate = (RZGate(theta_offset), [Qubit(qr_eval,0)],[])
                    circ_temp.data.insert(1,theta_gate)
                    
                    pauli_matrix = pauli.primitive.to_matrix()
                    
                    # Do nothing for identity
                    if not np.allclose(pauli_matrix, np.identity(pauli_matrix.shape[0])):
                        
                        pauli_circ = pauli.to_circuit()
                        pauli_gate = pauli_circ.to_gate(label=pauli.primitive.to_label())
                        c_gate = pauli_gate.control(1)
                        range_qubits = [i for i in range(c_gate.num_qubits)]
                        last = range_qubits.pop()
                        range_qubits.insert(0,last)
                    
                        circ_temp.append(c_gate,range_qubits)
                        
                        
                    

                    # Add second hadamard
                    circ_temp.h(qr_eval)
                    circ_trans = self.quantum_instance.transpile(circ_temp)

                    circuits_C[i].append(circ_trans)


        self._circuits_C - circuits_C
        self._coeffs = coeffs


        return 0

    def compute_minimum_eigenvalue(
        self, operator: OperatorBase, aux_operators: Optional[List[Optional[OperatorBase]]] = None
    ) -> MinimumEigensolverResult:

        op_pauli = operator.to_pauli_op()
        self._operator = op_pauli

        self.construct_circuit()

        self._ret = VQEResult()
        self._ret.combine(vqresult)

        return self._ret

    def get_optimal_cost(self) -> float:
        """Get the minimal cost or energy found by the VQE."""
        if self._ret.optimal_point is None:
            raise AlgorithmError(
                "Cannot return optimal cost before running the " "algorithm to find optimal params."
            )
        return self._ret.optimal_value

    def get_optimal_circuit(self) -> QuantumCircuit:
        """Get the circuit with the optimal parameters."""
        if self._ret.optimal_point is None:
            raise AlgorithmError(
                "Cannot find optimal circuit before running the "
                "algorithm to find optimal params."
            )
        return self.ansatz.assign_parameters(self._ret.optimal_parameters)

    def get_optimal_vector(self) -> Union[List[float], Dict[str, int]]:
        """Get the simulation outcome of the optimal circuit."""
        from qiskit.utils.run_circuits import find_regs_by_name

        if self._ret.optimal_point is None:
            raise AlgorithmError(
                "Cannot find optimal vector before running the " "algorithm to find optimal params."
            )
        qc = self.get_optimal_circuit()
        min_vector = {}
        if self._quantum_instance.is_statevector:
            ret = self._quantum_instance.execute(qc)
            min_vector = ret.get_statevector(qc)
        else:
            c = ClassicalRegister(qc.width(), name="c")
            q = find_regs_by_name(qc, "q")
            qc.add_register(c)
            qc.barrier(q)
            qc.measure(q, c)
            ret = self._quantum_instance.execute(qc)
            counts = ret.get_counts(qc)
            # normalize, just as done in CircuitSampler.sample_circuits
            shots = self._quantum_instance._run_config.shots
            min_vector = {b: (v / shots) ** 0.5 for (b, v) in counts.items()}
        return min_vector

    @property
    def optimal_params(self) -> List[float]:
        """The optimal parameters for the ansatz."""
        if self._ret.optimal_point is None:
            raise AlgorithmError("Cannot find optimal params before running the algorithm.")
        return self._ret.optimal_point

