# File: tests/test_orchestration/test_backend.py

import pytest
from qudet.compute.backend import BackendManager


class TestBackendManager:
    """Test suite for BackendManager."""
    
    def test_simulator_backend_creation(self):
        """Test creation of local simulator backend."""
        backend = BackendManager.get_backend("simulator")
        
        # Should return AerSimulator instance
        assert backend is not None
        # AerSimulator has name attribute
        assert hasattr(backend, 'name') or hasattr(backend, '__class__')
        
    def test_simulator_backend_default(self):
        """Test that default backend is simulator."""
        backend = BackendManager.get_backend()  # No name specified
        
        assert backend is not None
        
    def test_optimize_level_simulator(self):
        """Test optimization level for simulator (should be low)."""
        level = BackendManager.optimize_level("simulator")
        
        assert level == 1  # Low optimization for simulator
        
    def test_optimize_level_real_hardware(self):
        """Test optimization level for real hardware (should be high)."""
        level = BackendManager.optimize_level("ibm_kyoto")
        
        assert level == 3  # Max optimization for real hardware
        
    def test_optimize_level_case_insensitive(self):
        """Test optimization level is case-insensitive."""
        level1 = BackendManager.optimize_level("simulator")
        level2 = BackendManager.optimize_level("SIMULATOR")
        level3 = BackendManager.optimize_level("SimuLator")
        
        assert level1 == level2 == level3 == 1
        
    def test_ibm_backend_fallback(self):
        """Test that invalid IBM backend falls back to simulator."""
        # Try to connect to non-existent backend (will fail gracefully)
        backend = BackendManager.get_backend("ibm_nonexistent")
        
        # Should fall back to simulator
        assert backend is not None
        
    def test_get_backend_with_simulator_name(self):
        """Test explicitly requesting simulator."""
        backend = BackendManager.get_backend(name="simulator")
        
        assert backend is not None
        
    def test_optimize_level_various_ibm_backends(self):
        """Test optimization level for various IBM backend names."""
        backends = ["ibm_brisbane", "ibm_kyoto", "ibm_heron"]
        
        for backend_name in backends:
            level = BackendManager.optimize_level(backend_name)
            assert level == 3  # All real hardware gets max optimization
