"""
Script to generate test ONNX models for upload testing.
Creates a simple neural network and exports to ONNX format.
"""
import torch
import torch.nn as nn

class SimpleCardioModel(nn.Module):
    """Simple model for cardiovascular risk prediction."""
    def __init__(self, input_size=10):
        super(SimpleCardioModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 3)  # 3 classes: no_risk, at_risk, high_risk
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

model = SimpleCardioModel(input_size=10)
dummy_input = torch.randn(1, 10)

torch.onnx.export(
    model,
    dummy_input,
    "onnx/cardionet_small.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("[OK] Created: onnx/cardionet_small.onnx (10 features)")


class MediumModel(nn.Module):
    """Medium-sized model for testing."""
    def __init__(self, input_size=100):
        super(MediumModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 2)  # Binary classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x

medium_model = MediumModel(input_size=100)
medium_dummy = torch.randn(1, 100)

torch.onnx.export(
    medium_model,
    medium_dummy,
    "onnx/clinical_medium.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("[OK] Created: onnx/clinical_medium.onnx (100 features)")


class GeneticsModel(nn.Module):
    """Large model for genetics data."""
    def __init__(self, input_size=5000):
        super(GeneticsModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(256, 4)  # 4 classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x

genetics_model = GeneticsModel(input_size=5000)
genetics_dummy = torch.randn(1, 5000)

torch.onnx.export(
    genetics_model,
    genetics_dummy,
    "onnx/genetics_large.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("[OK] Created: onnx/genetics_large.onnx (5000 features)")
print("\n[OK] All ONNX models created successfully!")
