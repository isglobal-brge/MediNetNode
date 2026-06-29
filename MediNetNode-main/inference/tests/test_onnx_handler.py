"""
Tests for ONNX validator and inference engine.
"""
import pytest
import numpy as np
import os
import tempfile
from pathlib import Path

# Import ONNX modules for creating test models
try:
    import onnx
    from onnx import helper, TensorProto
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from inference.utils.onnx_handler import (
    ONNXValidator,
    ONNXInferenceEngine,
    ONNXValidationError,
    ONNX_AVAILABLE as HANDLER_ONNX_AVAILABLE
)


pytestmark = pytest.mark.skipif(
    not ONNX_AVAILABLE,
    reason="onnx and onnxruntime are required for these tests"
)


@pytest.fixture
def simple_onnx_model():
    """Create a simple valid ONNX model for testing."""
    # Create a simple linear model: output = input * 2 + 1
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 3])

    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 3])

    # Create weights (2.0)
    weight_tensor = helper.make_tensor(
        name='weight',
        data_type=TensorProto.FLOAT,
        dims=[3, 3],
        vals=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float32).flatten().tolist()
    )

    # Create bias (1.0)
    bias_tensor = helper.make_tensor(
        name='bias',
        data_type=TensorProto.FLOAT,
        dims=[3],
        vals=[1.0, 1.0, 1.0]
    )

    # Create Gemm node (General Matrix Multiplication)
    gemm_node = helper.make_node(
        'Gemm',
        inputs=['input', 'weight', 'bias'],
        outputs=['output'],
        alpha=1.0,
        beta=1.0,
        transB=0
    )

    graph = helper.make_graph(
        nodes=[gemm_node],
        name='simple_model',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_tensor, bias_tensor]
    )

    model = helper.make_model(graph, producer_name='test')
    model.opset_import[0].version = 11  # Use opset 11 for compatibility with onnxruntime
    model.ir_version = 8  # IR version 8 is compatible with opset 11

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        temp_path = f.name
        onnx.save(model, temp_path)

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sklearn_onnx_model():
    """Create a sklearn-based ONNX model (tree classifier)."""
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 2])

    # Output (class probabilities)
    output_tensor = helper.make_tensor_value_info('probabilities', TensorProto.FLOAT, [None, 2])

    # Output (class label)
    label_tensor = helper.make_tensor_value_info('label', TensorProto.INT64, [None])

    tree_node = helper.make_node(
        'TreeEnsembleClassifier',
        inputs=['input'],
        outputs=['label', 'probabilities'],
        name='tree_classifier',
        # Minimal tree parameters
        nodes_falsenodeids=[0],
        nodes_featureids=[0],
        nodes_hitrates=[1.0],
        nodes_missing_value_tracks_true=[0],
        nodes_modes=['BRANCH_LEQ'],
        nodes_nodeids=[0],
        nodes_treeids=[0],
        nodes_truenodeids=[0],
        nodes_values=[0.5],
        class_ids=[0, 1],
        class_nodeids=[0, 0],
        class_treeids=[0, 0],
        class_weights=[0.5, 0.5],
        classlabels_int64s=[0, 1],
        post_transform='NONE'
    )

    graph = helper.make_graph(
        nodes=[tree_node],
        name='sklearn_tree',
        inputs=[input_tensor],
        outputs=[label_tensor, output_tensor]
    )

    model = helper.make_model(graph, producer_name='test')
    model.opset_import[0].version = 11  # Use opset 11 for compatibility with onnxruntime
    model.ir_version = 8  # IR version 8 is compatible with opset 11
    # Add ML opset for sklearn operators
    ml_opset = model.opset_import.add()
    ml_opset.domain = 'ai.onnx.ml'
    ml_opset.version = 2

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        temp_path = f.name
        onnx.save(model, temp_path)

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestONNXValidator:
    """Test ONNXValidator class."""

    def test_validator_initialization(self):
        """Test validator can be initialized."""
        validator = ONNXValidator()
        assert validator.max_size_mb == 500
        assert validator.max_size_bytes == 500 * 1024 * 1024

    def test_validator_custom_max_size(self):
        """Test validator with custom max size."""
        validator = ONNXValidator(max_size_mb=100)
        assert validator.max_size_mb == 100
        assert validator.max_size_bytes == 100 * 1024 * 1024

    def test_validate_simple_model(self, simple_onnx_model):
        """Test validation of a simple valid ONNX model."""
        validator = ONNXValidator()
        result = validator.validate(simple_onnx_model)

        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert 'metadata' in result
        assert 'input_schema' in result['metadata']
        assert 'output_schema' in result['metadata']

    @pytest.mark.skip(reason="TreeEnsembleClassifier requires specific opset handling")
    def test_validate_sklearn_model(self, sklearn_onnx_model):
        """Test validation of sklearn ONNX model."""
        validator = ONNXValidator()
        result = validator.validate(sklearn_onnx_model)

        assert result['valid'] is True
        assert len(result['errors']) == 0

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        validator = ONNXValidator()
        result = validator.validate('/nonexistent/path/model.onnx')

        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert 'does not exist' in result['errors'][0]

    def test_validate_oversized_file(self, simple_onnx_model):
        """Test validation rejects oversized files."""
        # Use very small max size (1 byte) to trigger size check
        validator = ONNXValidator(max_size_mb=0.000001)  # ~1 byte
        result = validator.validate(simple_onnx_model)

        assert result['valid'] is False
        assert any('exceeds maximum allowed size' in err for err in result['errors'])

    def test_compute_checksum(self, simple_onnx_model):
        """Test checksum computation."""
        checksum1 = ONNXValidator.compute_checksum(simple_onnx_model)
        checksum2 = ONNXValidator.compute_checksum(simple_onnx_model)

        assert len(checksum1) == 64  # SHA256 produces 64 hex chars
        assert checksum1 == checksum2  # Consistent

    def test_verify_integrity(self, simple_onnx_model):
        """Test integrity verification."""
        validator = ONNXValidator()
        correct_checksum = validator.compute_checksum(simple_onnx_model)

        assert validator.verify_integrity(simple_onnx_model, correct_checksum) is True
        assert validator.verify_integrity(simple_onnx_model, 'wrong_checksum') is False

    def test_extract_metadata(self, simple_onnx_model):
        """Test metadata extraction."""
        validator = ONNXValidator()
        result = validator.validate(simple_onnx_model)

        assert result['valid'] is True
        metadata = result['metadata']

        assert 'input_schema' in metadata
        input_schema = metadata['input_schema']
        assert 'input' in input_schema['feature_names']
        assert 'input' in input_schema['dtypes']
        assert input_schema['dtypes']['input'] == 'float32'

        assert 'output_schema' in metadata
        output_schema = metadata['output_schema']
        assert 'output' in output_schema['output_names']


class TestONNXInferenceEngine:
    """Test ONNXInferenceEngine class."""

    def test_engine_initialization(self, simple_onnx_model):
        """Test inference engine can be initialized."""
        engine = ONNXInferenceEngine(simple_onnx_model)
        assert engine.input_names == ['input']
        assert engine.output_names == ['output']

    def test_engine_nonexistent_model(self):
        """Test engine initialization with non-existent model."""
        with pytest.raises(ONNXValidationError, match="Model file not found"):
            ONNXInferenceEngine('/nonexistent/model.onnx')

    def test_predict_simple_model(self, simple_onnx_model):
        """Test prediction on simple model."""
        engine = ONNXInferenceEngine(simple_onnx_model)

        # Create input data (batch of 2 samples, 3 features)
        input_data = {
            'input': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        }

        output = engine.predict(input_data)

        assert 'output' in output
        assert output['output'].shape == (2, 3)

        # Verify calculation: output = input * 2 + 1
        expected = input_data['input'] * 2 + 1
        np.testing.assert_array_almost_equal(output['output'], expected, decimal=5)

    def test_predict_missing_input(self, simple_onnx_model):
        """Test prediction fails with missing input."""
        engine = ONNXInferenceEngine(simple_onnx_model)

        # Missing required input
        input_data = {}

        with pytest.raises(ONNXValidationError, match="Missing required input"):
            engine.predict(input_data)

    def test_predict_invalid_input_type(self, simple_onnx_model):
        """Test prediction fails with invalid input type."""
        engine = ONNXInferenceEngine(simple_onnx_model)

        # Not a dictionary
        input_data = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ONNXValidationError, match="must be a dictionary"):
            engine.predict(input_data)

    def test_predict_batch(self, simple_onnx_model):
        """Test batch prediction."""
        engine = ONNXInferenceEngine(simple_onnx_model)

        # Create larger batch (10 samples)
        input_data = {
            'input': np.random.rand(10, 3).astype(np.float32)
        }

        output = engine.predict_batch(input_data, batch_size=3)

        assert 'output' in output
        assert output['output'].shape == (10, 3)

        # Verify calculation
        expected = input_data['input'] * 2 + 1
        np.testing.assert_array_almost_equal(output['output'], expected, decimal=5)

    def test_get_input_info(self, simple_onnx_model):
        """Test getting input information."""
        engine = ONNXInferenceEngine(simple_onnx_model)
        input_info = engine.get_input_info()

        assert len(input_info) == 1
        assert input_info[0]['name'] == 'input'

    def test_get_output_info(self, simple_onnx_model):
        """Test getting output information."""
        engine = ONNXInferenceEngine(simple_onnx_model)
        output_info = engine.get_output_info()

        assert len(output_info) == 1
        assert output_info[0]['name'] == 'output'
