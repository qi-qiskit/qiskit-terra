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

from qiskit import ClassicalRegister, QuantumCircuit
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

        self._max_iter = max_iter

        super().__init__(
            ansatz=ansatz,
            quantum_instance=quantum_instance,
        )
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
        circuits = []

        return circuits

    def compute_minimum_eigenvalue(
        self, operator: OperatorBase, aux_operators: Optional[List[Optional[OperatorBase]]] = None
    ) -> MinimumEigensolverResult:



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

