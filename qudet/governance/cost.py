
from qiskit import QuantumCircuit

class ResourceEstimator:
    """
    Estimates the cost and complexity of a Quantum Job before running it.
    Crucial for budgeting in large-scale data pipelines.
    """
    
    @staticmethod
    def estimate_circuit_cost(circuit: QuantumCircuit, shots: int = 1024, hardware_rate: float = 0.5):
        """
        params:
            hardware_rate: Cost per second (dummy default $0.50/sec)
        """
        depth = circuit.depth()
        
        ops = circuit.count_ops()
        n_cnots = ops.get('cx', 0)
        n_qubits = circuit.num_qubits
        
        estimated_exec_time = (depth * 1e-6 * shots) + (0.1)
        
        estimated_price = estimated_exec_time * hardware_rate
        
        report = {
            "qubits_used": n_qubits,
            "circuit_depth": depth,
            "cnot_count": n_cnots,
            "total_shots": shots,
            "est_runtime_sec": round(estimated_exec_time, 4),
            "est_cost_usd": round(estimated_price, 4)
        }
        return report

    @staticmethod
    def check_pipeline_feasibility(n_samples: int, n_features: int, max_depth: int = 100):
        """
        Advises if the dataset is too big for current quantum tech.
        """
        if n_features > 127:
            return "INFEASIBLE: Too many features. Use RandomProjector first."
        
        if n_samples > 10000:
             return "EXPENSIVE: >10k samples. Recommended to use CoresetReducer."
             
        return "FEASIBLE: Job fits within standard NISQ limits."
