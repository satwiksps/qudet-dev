"""
Data integrity verification for quantum encodings.

Provides round-trip checks that validate whether a quantum encoding
preserves the information content of the original classical data.
"""

import logging
from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qudet.core.base import BaseEncoder

logger = logging.getLogger(__name__)


class DataIntegrityCheck:
    """Verifies that the quantum encoding process preserved data information.

    Performs a 'round-trip' check:
    ``Data → Quantum State → Measured Probabilities → Compare with Original``

    This is a **simulation-only** check (unit-test helper), useful for:

    * Validating encoder implementations.
    * Debugging encoding issues.
    * Unit-testing encoder correctness.

    Examples:
        >>> checker = DataIntegrityCheck()
        >>> is_valid = checker.verify_encoding(data, encoder, tolerance=1e-5)
        >>> stats = checker.compute_encoding_fidelity(data, encoder)
    """

    @staticmethod
    def verify_encoding(
        original_data: np.ndarray,
        encoder: BaseEncoder,
        tolerance: float = 1e-5,
    ) -> bool:
        """Check whether the encoded state correctly represents the data.

        Verification strategy depends on the encoder type:

        * **StatevectorEncoder**: amplitudes should match the normalised
          data vector.
        * **Other encoders**: only checks that encoding succeeds without
          errors.

        Args:
            original_data: Original input data (1-D array).
            encoder: Encoder instance to verify.
            tolerance: Absolute tolerance for floating-point comparison.

        Returns:
            ``True`` if data integrity is verified.

        Raises:
            TypeError: If *encoder* is not a ``BaseEncoder`` instance.
            ValueError: If the integrity check fails (with diagnostics).
        """
        if not isinstance(encoder, BaseEncoder):
            raise TypeError(
                f"encoder must be a BaseEncoder instance, "
                f"got {type(encoder).__name__}"
            )

        try:
            qc = encoder.encode(original_data)
            state = Statevector(qc)
            probs = state.probabilities()
        except Exception as e:
            raise ValueError(f"Encoding failed: {e}") from e

        # Use isinstance check on the encoder type rather than string matching
        from qudet.encoders.statevector import StatevectorEncoder
        if isinstance(encoder, StatevectorEncoder):
            try:
                norm_data = original_data / np.linalg.norm(original_data)
                squared_data = np.abs(norm_data) ** 2

                n = len(squared_data)

                if np.allclose(probs[:n], squared_data, atol=tolerance):
                    return True
                else:
                    raise ValueError(
                        f"Integrity Fail for StatevectorEncoder:\n"
                        f"  Expected probs (first 5): {squared_data[:5]}\n"
                        f"  Actual probs (first 5): {probs[:5]}\n"
                        f"  Tolerance: {tolerance}"
                    )
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(
                    f"StatevectorEncoder integrity check failed: {e}"
                ) from e

        return True

    @staticmethod
    def compute_encoding_fidelity(
        original_data: np.ndarray,
        encoder: BaseEncoder,
    ) -> dict:
        """Compute detailed fidelity statistics of the encoding.

        Fidelity measures how well the quantum state represents the data:
        ``F = |⟨data|ψ⟩|²`` — the overlap between the data and the
        quantum state.

        Args:
            original_data: Original input data (1-D array).
            encoder: Encoder instance.

        Returns:
            Dictionary with keys:
                - ``encoder`` (str): Encoder class name.
                - ``fidelity`` (float): Overall fidelity (0–1).
                - ``min_probability`` (float): Minimum state probability.
                - ``max_probability`` (float): Maximum state probability.
                - ``shannon_entropy`` (float): Shannon entropy of state.
                - ``purity`` (float): State purity (1 = pure).
                - ``num_qubits`` (int or None): Number of qubits used.
        """
        qc = encoder.encode(original_data)
        state = Statevector(qc)
        probs = state.probabilities()

        encoder_name = encoder.__class__.__name__

        # Use isinstance for type check
        from qudet.encoders.statevector import StatevectorEncoder
        if isinstance(encoder, StatevectorEncoder):
            norm_data = original_data / np.linalg.norm(original_data)
            squared_data = np.abs(norm_data) ** 2

            n = min(len(squared_data), len(probs))
            fidelity = float(np.sum(np.sqrt(squared_data[:n] * probs[:n])))
        else:
            fidelity = float(np.sum(probs ** 2))

        probs_nonzero = probs[probs > 0]
        entropy = float(-np.sum(probs_nonzero * np.log2(probs_nonzero + 1e-10)))

        purity = float(np.sum(probs ** 2))

        return {
            "encoder": encoder_name,
            "fidelity": fidelity,
            "min_probability": float(np.min(probs)),
            "max_probability": float(np.max(probs)),
            "shannon_entropy": entropy,
            "purity": purity,
            "num_qubits": getattr(encoder, "n_qubits", None),
        }

    @staticmethod
    def verify_batch(
        data_batch: np.ndarray,
        encoder: BaseEncoder,
        tolerance: float = 1e-5,
    ) -> Tuple[int, int]:
        """Verify encoding integrity for a batch of samples.

        Args:
            data_batch: Batch of data (2-D array, shape: n_samples × n_features).
            encoder: Encoder instance.
            tolerance: Tolerance for floating-point comparison.

        Returns:
            Tuple of ``(num_passed, num_failed)``.
        """
        num_passed = 0
        num_failed = 0

        for i, sample in enumerate(data_batch):
            try:
                DataIntegrityCheck.verify_encoding(sample, encoder, tolerance)
                num_passed += 1
            except Exception as e:
                num_failed += 1
                logger.warning("Sample %d failed: %s", i, str(e)[:100])

        logger.info(
            "Batch verification: %d/%d passed", num_passed, len(data_batch)
        )
        return num_passed, num_failed
