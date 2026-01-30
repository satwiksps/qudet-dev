"""Tests for Quantum Security Module (Privacy)"""

import pytest
from qiskit import QuantumCircuit
from qudet.governance.privacy import QuantumDifferentialPrivacy


class TestQuantumDifferentialPrivacy:
    """Test suite for QuantumDifferentialPrivacy class."""
    
    def test_initialization(self):
        """Test QuantumDifferentialPrivacy initialization."""
        privacy = QuantumDifferentialPrivacy(epsilon=1.0)
        assert privacy.epsilon == 1.0
        assert 0 < privacy.noise_prob < 0.5
    
    def test_initialization_with_different_epsilon(self):
        """Test initialization with various epsilon values."""
        for eps in [0.1, 0.5, 1.0, 2.0, 5.0]:
            privacy = QuantumDifferentialPrivacy(epsilon=eps)
            assert privacy.epsilon == eps
            assert privacy.noise_prob > 0
    
    def test_invalid_epsilon_raises_error(self):
        """Test that non-positive epsilon raises error."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            QuantumDifferentialPrivacy(epsilon=0)
        
        with pytest.raises(ValueError, match="epsilon must be positive"):
            QuantumDifferentialPrivacy(epsilon=-1.0)
    
    def test_sanitize_creates_new_circuit(self):
        """Test that sanitize creates a new circuit."""
        privacy = QuantumDifferentialPrivacy(epsilon=1.0)
        
        # Create a simple circuit
        qc = QuantumCircuit(2, name="test_circuit")
        qc.h(0)
        qc.cx(0, 1)
        
        # Sanitize it
        secure_qc = privacy.sanitize(qc)
        
        # Check it's a new circuit
        assert isinstance(secure_qc, QuantumCircuit)
        assert secure_qc is not qc
    
    def test_sanitize_preserves_qubits(self):
        """Test that sanitize preserves qubit count."""
        privacy = QuantumDifferentialPrivacy(epsilon=1.0)
        
        for n_qubits in [1, 2, 3, 5]:
            qc = QuantumCircuit(n_qubits)
            qc.h(range(n_qubits))
            
            secure_qc = privacy.sanitize(qc)
            assert secure_qc.num_qubits == n_qubits
    
    def test_sanitize_has_privacy_wall(self):
        """Test that sanitized circuit has privacy barrier."""
        privacy = QuantumDifferentialPrivacy(epsilon=1.0)
        
        qc = QuantumCircuit(2)
        qc.h(0)
        
        secure_qc = privacy.sanitize(qc)
        
        # Check circuit name indicates privacy
        assert "Privacy" in secure_qc.name or "privacy" in secure_qc.name.lower()
    
    def test_estimate_privacy_loss(self):
        """Test privacy loss estimation."""
        privacy = QuantumDifferentialPrivacy(epsilon=1.0)
        
        loss_1 = privacy.estimate_privacy_loss(1)
        loss_4 = privacy.estimate_privacy_loss(4)
        loss_9 = privacy.estimate_privacy_loss(9)
        
        # Privacy loss should increase with sqrt(n_queries)
        assert loss_1 < loss_4 < loss_9
        # Check proportions: sqrt(4) = 2, sqrt(9) = 3
        assert abs(loss_4 - 2 * loss_1) < 0.01
        assert abs(loss_9 - 3 * loss_1) < 0.01
    
    def test_estimate_privacy_loss_invalid_input(self):
        """Test that invalid input raises error."""
        privacy = QuantumDifferentialPrivacy(epsilon=1.0)
        
        with pytest.raises(ValueError, match="n_queries must be positive"):
            privacy.estimate_privacy_loss(0)
        
        with pytest.raises(ValueError, match="n_queries must be positive"):
            privacy.estimate_privacy_loss(-1)
    
    def test_get_noise_parameters(self):
        """Test getting noise parameters."""
        privacy = QuantumDifferentialPrivacy(epsilon=1.0)
        params = privacy.get_noise_parameters()
        
        assert "epsilon" in params
        assert "noise_probability" in params
        assert "privacy_level" in params
        assert params["epsilon"] == 1.0
    
    def test_privacy_levels(self):
        """Test privacy level classification."""
        # High privacy (low epsilon)
        privacy_high = QuantumDifferentialPrivacy(epsilon=0.1)
        assert "HIGH" in privacy_high.get_noise_parameters()["privacy_level"]
        
        # Medium privacy
        privacy_med = QuantumDifferentialPrivacy(epsilon=1.0)
        assert "MEDIUM" in privacy_med.get_noise_parameters()["privacy_level"]
        
        # Low privacy (high epsilon)
        privacy_low = QuantumDifferentialPrivacy(epsilon=5.0)
        assert "LOW" in privacy_low.get_noise_parameters()["privacy_level"]
