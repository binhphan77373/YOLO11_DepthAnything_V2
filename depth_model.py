import torch
from depth_anything_v2.dpt import DepthAnythingV2
from config import DEPTH_MODEL_PATH, ENCODER

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def load_depth_model(device):
    print("Loading Depth-Anything-V2 model...")
    model = DepthAnythingV2(**model_configs[ENCODER])
    model.load_state_dict(torch.load(DEPTH_MODEL_PATH, map_location=device))
    model = model.to(device).eval()
    return model

def estimate_depth(model, frame, device):
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
    with torch.no_grad():
        depth_map = model.infer_image(frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    return depth_map
