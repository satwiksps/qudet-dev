
from qiskit_aer import AerSimulator

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
    HAS_IBM = True
except ImportError:
    HAS_IBM = False


class BackendManager:
    """
    Centralized controller for Quantum Hardware connections.
    Handles the switch between Local Simulation and Cloud QPU (IBM/IonQ).
    
    Design Pattern: Factory Method for backend creation with graceful fallback.
    
    Example:
        >>> backend = BackendManager.get_backend("simulator")
        >>> # or connect to real quantum computer:
        >>> backend = BackendManager.get_backend("ibm_brisbane", token="YOUR_API_TOKEN")
    """
    
    @staticmethod
    def get_backend(name: str = "simulator", token: str = None):
        """
        Factory method to get a backend instance.
        
        Parameters
        ----------
        name : str
            Backend name: 'simulator', 'ibm_brisbane', 'ibm_kyoto', etc.
        token : str, optional
            IBM Quantum API Token. If None, looks for saved credentials.
            
        Returns
        -------
        Backend or AerSimulator
            Quantum backend instance for circuit execution
            
        Raises
        ------
        ImportError
            If requesting IBM backend but qiskit-ibm-runtime not installed
        """
        print(f"--- Connecting to Backend: {name} ---")
        
        if name == "simulator":
            return AerSimulator(method='statevector')
            
        if not HAS_IBM:
            raise ImportError(
                "qiskit-ibm-runtime is not installed. "
                "Install with: pip install qiskit-ibm-runtime"
            )

        try:
            if token:
                service = QiskitRuntimeService(channel="ibm_quantum", token=token)
            else:
                service = QiskitRuntimeService(channel="ibm_quantum")
                
            real_backend = service.backend(name)
            n_qubits = real_backend.num_qubits
            print(f"   Connected to QPU: {real_backend.name} ({n_qubits} qubits)")
            return real_backend
            
        except Exception as e:
            print(f"   Connection Failed: {e}")
            print("   -> Falling back to Simulator")
            return AerSimulator()

    @staticmethod
    def optimize_level(backend_name: str) -> int:
        """
        Returns recommended transpilation optimization level for a backend.
        
        Optimization Levels:
        - 0: No optimization (fastest transpile, may use more gates)
        - 1: Light optimization (balance speed/quality)
        - 2: Medium optimization
        - 3: Heavy optimization (slowest transpile, fewest gates)
        
        Parameters
        ----------
        backend_name : str
            Name of the backend
            
        Returns
        -------
        int
            Recommended optimization level (0-3)
        """
        if "simulator" in backend_name.lower():
            return 1
        return 3
