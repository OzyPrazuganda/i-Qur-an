import torch
import json
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.nn as nn

# ---------------------------
# Load label mapping
# ---------------------------
with open("label_mapping.json", "r") as f:
    label_to_index = json.load(f)

input_size = 42  # must match your training setup
num_classes = len(label_to_index)
model_save_path = "mlp_model.pth"

# ---------------------------
# Recreate model architecture
# ---------------------------
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ---------------------------
# Load trained weights
# ---------------------------
model = MLP(input_size=input_size, num_classes=num_classes)
model.load_state_dict(torch.load(model_save_path, map_location="cpu"))
model.eval()

# ---------------------------
# Example input (shape [1, 42])
# ---------------------------
example_input = torch.randn(1, input_size)

# ---------------------------
# Convert to TorchScript
# ---------------------------
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save("hijaiyah_mlp.pt")

# ---------------------------
# Optimize for mobile and save
# ---------------------------
optimized_scripted_module = optimize_for_mobile(traced_script_module)
optimized_scripted_module._save_for_lite_interpreter("hijaiyah_mlp_mobile.ptl")

print("✅ Export complete: hijaiyah_mlp.pt and hijaiyah_mlp_mobile.ptl saved")
