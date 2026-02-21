"""
Quantum Circuit and Model Serialization.

Data pipelines crash. You need to save your work. This module saves
Quantum Circuits (which are objects) into QASM (Quantum Assembly) or JSON
so they can be stored in a database or S3.
"""

import json
import logging
import pickle
from typing import List

from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps, loads

logger = logging.getLogger(__name__)


class QuantumSerializer:
    """Handles saving and loading of QuDET artifacts.

    Supported formats:

    * **QASM** — Quantum Assembly Language, the standard text format for
      quantum circuits.
    * **JSON** — Metadata and circuit descriptions stored as a JSON list
      of QASM strings.
    * **Pickle** — Full Python objects (models, trained instances).

    This enables persistence of quantum computations for reproducibility
    and pipeline recovery.
    """

    @staticmethod
    def save_circuits(circuits: List[QuantumCircuit], filepath: str) -> None:
        """Save a list of circuits to a JSON file of QASM strings.

        Args:
            circuits: List of quantum circuits to save.
            filepath: Path to output JSON file.

        Raises:
            TypeError: If *circuits* is not a list of ``QuantumCircuit``.
        """
        if not isinstance(circuits, list):
            raise TypeError("'circuits' must be a list of QuantumCircuit.")

        data = []
        for i, qc in enumerate(circuits):
            qasm_str = dumps(qc)
            record = {
                "id": i,
                "n_qubits": qc.num_qubits,
                "n_clbits": qc.num_clbits,
                "name": qc.name,
                "qasm": qasm_str,
            }
            data.append(record)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d circuits to %s", len(circuits), filepath)

    @staticmethod
    def load_circuits(filepath: str) -> List[QuantumCircuit]:
        """Load circuits from a QuDET JSON file.

        Args:
            filepath: Path to JSON file with saved circuits.

        Returns:
            List of loaded ``QuantumCircuit`` objects.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        circuits: List[QuantumCircuit] = []
        for record in data:
            qasm_str = record["qasm"]
            try:
                qc = loads(qasm_str)
            except Exception as exc:
                logger.warning(
                    "Could not load circuit %s: %s", record["id"], exc
                )
                continue
            circuits.append(qc)

        logger.info("Loaded %d circuits from %s", len(circuits), filepath)
        return circuits

    @staticmethod
    def save_model(model: object, filepath: str) -> None:
        """Save a trained QuDET model via pickle.

        This allows you to save the entire fitted model state, including
        centroids, metadata, and fitted parameters.

        Args:
            model: Trained QuDET model instance.
            filepath: Path to output pickle file.
        """
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved model to %s", filepath)

    @staticmethod
    def load_model(filepath: str) -> object:
        """Load a pickled QuDET model.

        .. warning::

            **Security**: ``pickle.load`` can execute arbitrary code.
            Only load pickle files from **trusted** sources.  Never
            unpickle data received from an untrusted or unauthenticated
            source.

        Args:
            filepath: Path to pickle file.

        Returns:
            Loaded model instance.
        """
        with open(filepath, "rb") as f:
            model = pickle.load(f)  # noqa: S301
        logger.info("Loaded model from %s", filepath)
        return model

    @staticmethod
    def export_circuit_qasm(circuit: QuantumCircuit, filepath: str) -> None:
        """Export a single circuit to a QASM file.

        Args:
            circuit: Circuit to export.
            filepath: Output QASM file path.
        """
        qasm_str = dumps(circuit)
        with open(filepath, "w") as f:
            f.write(qasm_str)
        logger.info("Exported circuit to %s", filepath)

    @staticmethod
    def import_circuit_qasm(filepath: str) -> QuantumCircuit:
        """Import a circuit from a QASM file.

        Args:
            filepath: Path to QASM file.

        Returns:
            Imported ``QuantumCircuit``.
        """
        with open(filepath, "r") as f:
            qasm_str = f.read()

        circuit = loads(qasm_str)
        logger.info("Imported circuit from %s", filepath)
        return circuit
