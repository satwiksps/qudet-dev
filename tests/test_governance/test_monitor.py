"""
Test suite for Job Monitor (qudet.utils.monitor)
"""

import pytest
import time
from io import StringIO
import sys
from qudet.governance.monitor import JobMonitor


class TestJobMonitor:
    """Test cases for JobMonitor class."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = JobMonitor(100, description="Test Job")
        
        assert monitor.total == 100
        assert monitor.desc == "Test Job"
        assert monitor.current == 0
        assert monitor.start_time > 0
    
    def test_initialization_invalid_total(self):
        """Test error on invalid total."""
        with pytest.raises(ValueError, match="must be positive"):
            JobMonitor(0)
        
        with pytest.raises(ValueError, match="must be positive"):
            JobMonitor(-10)
    
    def test_update_single(self):
        """Test updating progress by 1."""
        monitor = JobMonitor(10)
        
        assert monitor.current == 0
        monitor.update(1)
        assert monitor.current == 1
    
    def test_update_multiple(self):
        """Test updating progress by multiple items."""
        monitor = JobMonitor(100)
        
        monitor.update(25)
        assert monitor.current == 25
        
        monitor.update(25)
        assert monitor.current == 50
    
    def test_update_clamping(self):
        """Test that progress doesn't exceed total."""
        monitor = JobMonitor(10)
        
        monitor.update(5)
        monitor.update(10)  # Exceeds total
        
        assert monitor.current == 10  # Clamped to total
    
    def test_update_invalid(self):
        """Test error on invalid update value."""
        monitor = JobMonitor(100)
        
        with pytest.raises(ValueError, match="must be positive"):
            monitor.update(0)
        
        with pytest.raises(ValueError, match="must be positive"):
            monitor.update(-5)
    
    def test_get_elapsed(self):
        """Test elapsed time calculation."""
        monitor = JobMonitor(10)
        
        time.sleep(0.1)
        elapsed = monitor.get_elapsed()
        
        assert elapsed >= 0.1
    
    def test_get_rate(self):
        """Test processing rate calculation."""
        monitor = JobMonitor(100)
        
        monitor.update(10)
        time.sleep(0.1)
        
        rate = monitor.get_rate()
        assert rate > 0
    
    def test_get_rate_no_progress(self):
        """Test rate with no progress."""
        monitor = JobMonitor(100)
        
        rate = monitor.get_rate()
        assert rate == 0
    
    def test_get_eta(self):
        """Test ETA calculation."""
        monitor = JobMonitor(100)
        
        monitor.update(50)
        time.sleep(0.1)
        
        eta = monitor.get_eta()
        assert eta >= 0
    
    def test_get_eta_complete(self):
        """Test ETA when complete."""
        monitor = JobMonitor(10)
        
        monitor.update(10)
        eta = monitor.get_eta()
        
        assert eta == 0
    
    def test_get_status(self):
        """Test status dictionary."""
        monitor = JobMonitor(100)
        
        monitor.update(25)
        status = monitor.get_status()
        
        assert "current" in status
        assert "total" in status
        assert "percent" in status
        assert "elapsed_seconds" in status
        assert "rate_items_per_second" in status
        assert "eta_seconds" in status
        assert "is_complete" in status
        
        assert status["current"] == 25
        assert status["total"] == 100
        assert status["percent"] == pytest.approx(0.25)
        assert status["is_complete"] is False
    
    def test_get_status_complete(self):
        """Test status when complete."""
        monitor = JobMonitor(10)
        
        monitor.update(10)
        status = monitor.get_status()
        
        assert status["current"] == 10
        assert status["is_complete"] is True
    
    def test_reset(self):
        """Test progress reset."""
        monitor = JobMonitor(100)
        
        monitor.update(50)
        assert monitor.current == 50
        
        monitor.reset()
        assert monitor.current == 0
    
    def test_context_manager(self):
        """Test context manager usage."""
        with JobMonitor(10, description="Context Test") as monitor:
            assert monitor is not None
            assert monitor.total == 10
            monitor.update(5)
            assert monitor.current == 5
    
    def test_str_representation(self):
        """Test string representation."""
        monitor = JobMonitor(100, description="Test")
        
        monitor.update(50)
        status_str = str(monitor)
        
        assert "Test" in status_str
        assert "50/100" in status_str
    
    def test_complete_workflow(self):
        """Test complete progress workflow."""
        monitor = JobMonitor(10, description="Complete Test")
        
        for i in range(10):
            monitor.update(1)
            time.sleep(0.01)
        
        status = monitor.get_status()
        assert status["is_complete"] is True
        assert status["current"] == 10
        assert status["percent"] == pytest.approx(1.0)
    
    def test_close(self):
        """Test closing monitor."""
        monitor = JobMonitor(100)
        
        monitor.update(50)
        monitor.close()  # Should handle gracefully
    
    def test_multiple_updates(self):
        """Test multiple sequential updates."""
        monitor = JobMonitor(100)
        
        for i in range(10):
            monitor.update(10)
        
        assert monitor.current == 100
        status = monitor.get_status()
        assert status["is_complete"] is True
    
    def test_monitor_precision(self):
        """Test calculation precision."""
        monitor = JobMonitor(1000)
        
        monitor.update(333)
        status = monitor.get_status()
        
        assert status["percent"] == pytest.approx(0.333, abs=0.001)
    
    def test_large_numbers(self):
        """Test with large item counts."""
        monitor = JobMonitor(1000000)
        
        monitor.update(500000)
        status = monitor.get_status()
        
        assert status["current"] == 500000
        assert status["percent"] == pytest.approx(0.5)
