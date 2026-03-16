#assumes data here (stock_data) is already normalized and cleaned
#sends the data in quantum state format

import numpy as np
import qiskit as qk

def amplitude_encode(data: np.ndarray) -> qk.QuantumCircuit:
    norm = np.linalg.norm(data)
    if norm == 0:
        raise ValueError("Input data cannot be the zero vector.")
    normalized_data = data / norm
    num_qubits = int(np.ceil(np.log2(len(normalized_data))))
    padded_length = 2 ** num_qubits
    if len(normalized_data) < padded_length:
        normalized_data = np.pad(normalized_data, (0, padded_length - len(normalized_data)), mode='constant')
    circuit = qk.QuantumCircuit(num_qubits)
    circuit.initialize(normalized_data, range(num_qubits))

    return circuit
