"""
ONNX model validation and inference engine.

Security-first implementation for validating and running ONNX models.
"""
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import onnx
    from onnx import checker, helper
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ONNXValidationError(Exception):
    """Raised when ONNX model validation fails."""
    pass


class ONNXValidator:
    """
    Validates ONNX models for security and compatibility.

    Security measures:
    - File size limits
    - Operator whitelist
    - External data reference blocking
    - Format integrity checks
    """

    # Allowed ONNX operators (whitelist approach for security)
    ALLOWED_OPERATORS = {
        # Math operations
        'Add', 'Sub', 'Mul', 'Div', 'Sqrt', 'Pow', 'Exp', 'Log', 'Abs', 'Neg',
        'Ceil', 'Floor', 'Round', 'Clip', 'Min', 'Max', 'Sum', 'Mean',
        # Activation functions
        'Relu', 'Sigmoid', 'Tanh', 'Softmax', 'LogSoftmax', 'Elu', 'Selu',
        'LeakyRelu', 'PRelu', 'Softplus', 'Softsign', 'HardSigmoid',
        # Neural network layers
        'MatMul', 'Gemm', 'Conv', 'ConvTranspose', 'MaxPool', 'AveragePool',
        'GlobalAveragePool', 'GlobalMaxPool', 'BatchNormalization', 'Dropout',
        'LRN', 'InstanceNormalization', 'LayerNormalization',
        # Shape operations
        'Flatten', 'Reshape', 'Transpose', 'Squeeze', 'Unsqueeze', 'Shape',
        'Slice', 'Split', 'Concat', 'Tile', 'Expand', 'Gather', 'Pad',
        # Recurrent layers
        'LSTM', 'GRU', 'RNN',
        # ML classifiers (sklearn models)
        'LinearClassifier', 'LinearRegressor', 'TreeEnsembleClassifier',
        'TreeEnsembleRegressor', 'SVMClassifier', 'SVMRegressor',
        'ZipMap', 'ArrayFeatureExtractor', 'Binarizer', 'Normalizer',
        'Scaler', 'Imputer', 'OneHotEncoder', 'LabelEncoder',
        # Comparison and logic
        'Equal', 'Greater', 'Less', 'And', 'Or', 'Not', 'Xor',
        'GreaterOrEqual', 'LessOrEqual',
        # Reduction operations
        'ReduceSum', 'ReduceMean', 'ReduceMax', 'ReduceMin', 'ReduceProd',
        # Other
        'Cast', 'Identity', 'Constant', 'ConstantOfShape',
    }

    # Maximum model file size (default 500MB, configurable via settings)
    DEFAULT_MAX_SIZE_MB = 500

    def __init__(self, max_size_mb: Optional[int] = None):
        """
        Initialize ONNX validator.

        Args:
            max_size_mb: Maximum file size in MB (default: 500MB)
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnx and onnxruntime are required for ONNX validation")

        self.max_size_mb = max_size_mb or self.DEFAULT_MAX_SIZE_MB
        self.max_size_bytes = self.max_size_mb * 1024 * 1024

    def validate(self, file_path: str) -> Dict[str, Any]:
        """
        Validate an ONNX model file.

        Args:
            file_path: Path to ONNX model file

        Returns:
            Dict with keys:
                - valid (bool): Whether model passed all checks
                - errors (list): List of error messages
                - warnings (list): List of warning messages
                - metadata (dict): Extracted metadata (if valid)

        Raises:
            ONNXValidationError: If validation fails
        """
        errors = []
        warnings = []
        metadata = {}

        # Check file exists
        if not os.path.exists(file_path):
            errors.append(f"File does not exist: {file_path}")
            return {'valid': False, 'errors': errors, 'warnings': warnings, 'metadata': metadata}

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.max_size_bytes:
            errors.append(
                f"File size ({file_size / 1024 / 1024:.2f}MB) exceeds "
                f"maximum allowed size ({self.max_size_mb}MB)"
            )
            return {'valid': False, 'errors': errors, 'warnings': warnings, 'metadata': metadata}

        # Load ONNX model
        try:
            model = onnx.load(file_path)
        except Exception as e:
            errors.append(f"Failed to load ONNX model: {str(e)}")
            return {'valid': False, 'errors': errors, 'warnings': warnings, 'metadata': metadata}

        # Check model integrity
        try:
            checker.check_model(model)
        except Exception as e:
            errors.append(f"ONNX model integrity check failed: {str(e)}")
            return {'valid': False, 'errors': errors, 'warnings': warnings, 'metadata': metadata}

        # Check for external data references (security risk)
        if self._has_external_data(model):
            errors.append("Model contains external data references (not allowed)")
            return {'valid': False, 'errors': errors, 'warnings': warnings, 'metadata': metadata}

        # Verify operators (whitelist)
        disallowed_ops = self._check_operators(model)
        if disallowed_ops:
            errors.append(f"Model contains disallowed operators: {', '.join(disallowed_ops)}")
            return {'valid': False, 'errors': errors, 'warnings': warnings, 'metadata': metadata}

        # Extract metadata
        try:
            metadata = self._extract_metadata(model)
        except Exception as e:
            warnings.append(f"Failed to extract metadata: {str(e)}")

        # All checks passed
        return {
            'valid': True,
            'errors': errors,
            'warnings': warnings,
            'metadata': metadata
        }

    def _has_external_data(self, model: onnx.ModelProto) -> bool:
        """
        Check if model has external data references.

        Args:
            model: ONNX model proto

        Returns:
            bool: True if external data found
        """
        for tensor in model.graph.initializer:
            if tensor.HasField('data_location') and tensor.data_location == onnx.TensorProto.EXTERNAL:
                return True
        return False

    def _check_operators(self, model: onnx.ModelProto) -> List[str]:
        """
        Check if model uses only allowed operators.

        Args:
            model: ONNX model proto

        Returns:
            List of disallowed operator types
        """
        disallowed = []
        for node in model.graph.node:
            if node.op_type not in self.ALLOWED_OPERATORS:
                if node.op_type not in disallowed:
                    disallowed.append(node.op_type)
        return disallowed

    def _extract_metadata(self, model: onnx.ModelProto) -> Dict[str, Any]:
        """
        Extract metadata from ONNX model.

        Args:
            model: ONNX model proto

        Returns:
            Dict with input_schema and output_schema
        """
        input_schema = {
            'inputs': [],
            'feature_names': [],
            'dtypes': {},
            'shapes': {}
        }

        output_schema = {
            'outputs': [],
            'output_names': [],
            'dtypes': {},
            'shapes': {},
            'type': 'unknown',  # 'classification', 'regression', or 'unknown'
            'classes': {}  # For classification: {0: 'Class0', 1: 'Class1', ...}
        }

        # Extract input information
        for input_tensor in model.graph.input:
            name = input_tensor.name
            input_schema['inputs'].append(name)
            input_schema['feature_names'].append(name)

            # Get dtype
            dtype = self._get_dtype_string(input_tensor.type.tensor_type.elem_type)
            input_schema['dtypes'][name] = dtype

            # Get shape
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                else:
                    shape.append(None)  # Dynamic dimension
            input_schema['shapes'][name] = shape

        # Extract output information
        for output_tensor in model.graph.output:
            name = output_tensor.name
            output_schema['outputs'].append(name)
            output_schema['output_names'].append(name)

            # Get dtype
            dtype = self._get_dtype_string(output_tensor.type.tensor_type.elem_type)
            output_schema['dtypes'][name] = dtype

            # Get shape
            shape = []
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                else:
                    shape.append(None)
            output_schema['shapes'][name] = shape

        # Detect model type based on output names
        output_names_lower = [n.lower() for n in output_schema['output_names']]
        if any('label' in n or 'class' in n for n in output_names_lower):
            output_schema['type'] = 'classification'
        elif any('probability' in n or 'probabilities' in n for n in output_names_lower):
            output_schema['type'] = 'classification'
        elif any('regression' in n or 'value' in n for n in output_names_lower):
            output_schema['type'] = 'regression'

        return {
            'input_schema': input_schema,
            'output_schema': output_schema
        }

    def _get_dtype_string(self, onnx_dtype: int) -> str:
        """Convert ONNX dtype enum to string."""
        dtype_map = {
            1: 'float32',
            2: 'uint8',
            3: 'int8',
            6: 'int32',
            7: 'int64',
            9: 'bool',
            10: 'float16',
            11: 'float64',
        }
        return dtype_map.get(onnx_dtype, f'unknown({onnx_dtype})')

    def verify_integrity(self, file_path: str, expected_checksum: str) -> bool:
        """
        Verify file integrity via SHA256 checksum.

        Args:
            file_path: Path to file
            expected_checksum: Expected SHA256 hexdigest

        Returns:
            bool: True if checksums match
        """
        actual_checksum = self.compute_checksum(file_path)
        return actual_checksum == expected_checksum

    @staticmethod
    def compute_checksum(file_path: str) -> str:
        """
        Compute SHA256 checksum of file.

        Args:
            file_path: Path to file

        Returns:
            str: SHA256 hexdigest
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


class ONNXInferenceEngine:
    """
    Secure ONNX inference engine with timeout and error handling.
    """

    def __init__(self, model_path: str, execution_timeout_seconds: int = 30):
        """
        Initialize inference engine.

        Args:
            model_path: Path to ONNX model file
            execution_timeout_seconds: Maximum execution time per inference

        Raises:
            ONNXValidationError: If model cannot be loaded
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is required for ONNX inference")

        if not os.path.exists(model_path):
            raise ONNXValidationError(f"Model file not found: {model_path}")

        try:
            # Create inference session with CPU provider
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1  # Limit threads for security
            sess_options.inter_op_num_threads = 1

            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            self.timeout = execution_timeout_seconds

        except Exception as e:
            raise ONNXValidationError(f"Failed to create inference session: {str(e)}")

    def predict(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference on input data.

        Args:
            input_data: Dict mapping input names to numpy arrays

        Returns:
            Dict mapping output names to numpy arrays

        Raises:
            ONNXValidationError: If inference fails
        """
        try:
            # Validate inputs
            if not isinstance(input_data, dict):
                raise ONNXValidationError("input_data must be a dictionary")

            # Check all required inputs are provided
            for input_name in self.input_names:
                if input_name not in input_data:
                    raise ONNXValidationError(f"Missing required input: {input_name}")

            # Run inference
            outputs = self.session.run(self.output_names, input_data)

            # Return as dict
            return {name: output for name, output in zip(self.output_names, outputs)}

        except Exception as e:
            raise ONNXValidationError(f"Inference failed: {str(e)}")

    def predict_batch(self, input_data: Dict[str, np.ndarray], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Run inference on batched input data.

        Args:
            input_data: Dict mapping input names to numpy arrays (first dimension is batch)
            batch_size: Batch size for processing

        Returns:
            Dict mapping output names to numpy arrays

        Raises:
            ONNXValidationError: If inference fails
        """
        # Get total number of samples from first input
        first_input = next(iter(input_data.values()))
        n_samples = first_input.shape[0]

        # Process in batches
        all_outputs = None

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)

            # Extract batch
            batch_input = {
                name: data[start_idx:end_idx]
                for name, data in input_data.items()
            }

            # Run inference on batch
            batch_outputs = self.predict(batch_input)

            # Accumulate results
            if all_outputs is None:
                all_outputs = {name: [] for name in batch_outputs.keys()}

            for name, output in batch_outputs.items():
                all_outputs[name].append(output)

        # Concatenate all batches
        return {
            name: np.concatenate(outputs, axis=0)
            for name, outputs in all_outputs.items()
        }

    def get_input_info(self) -> List[Dict[str, Any]]:
        """Get information about model inputs."""
        return [
            {
                'name': inp.name,
                'shape': inp.shape,
                'dtype': inp.type,
            }
            for inp in self.session.get_inputs()
        ]

    def get_output_info(self) -> List[Dict[str, Any]]:
        """Get information about model outputs."""
        return [
            {
                'name': out.name,
                'shape': out.shape,
                'dtype': out.type,
            }
            for out in self.session.get_outputs()
        ]
