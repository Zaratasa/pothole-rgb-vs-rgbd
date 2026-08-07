"""Compute depth-filter drift on pretrained U-Net RGB-D (seed 41, scene-disjoint)."""
import os, sys, json, random
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import pothrgbd_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("POTHRGBD_ROOT", os.path.join(HERE,"..","dataset","PothRGBD"))
SPLIT_JSON = os.path.join(HERE,"..","results","split_scene_disjoint.json")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
IMAGENET_MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(DEVICE)
IMAGENET_STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(DEVICE)

def set_seed(s): torch.manual_seed(s); np.random.seed(s); random.seed(s)

def norm_in(x):
    x = x.clone(); x[:,:3] = (x[:,:3]-IMAGENET_MEAN)/IMAGENET_STD; return x

def build_pretrained_rgbd():
    model = smp.Unet(encoder_name="resnet18",encoder_weights="imagenet",in_channels=3,classes=1)
    old = model.encoder.conv1
    new = torch.nn.Conv2d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
    with torch.no_grad():
        new.weight[:,:3] = old.weight.clone()
        new.weight[:,3:4] = old.weight.mean(dim=1,keepdim=True)
    model.encoder.conv1 = new
    return model.to(DEVICE)

set_seed(41)
samples = D.build_index(DATA_ROOT)
split   = D.split_from_json(samples, SPLIT_JSON)
DS = D.make_torch_dataset()
tr = DataLoader(DS(split["train"],mode="rgbd",out_size=(256,256),augment=True),
                batch_size=8,shuffle=True,num_workers=0,drop_last=True)

model = build_pretrained_rgbd()
# Save initial weights
init_w = model.encoder.conv1.weight.data.clone().cpu()

dice = smp.losses.DiceLoss(mode="binary",from_logits=True)
bce  = torch.nn.BCEWithLogitsLoss()
opt  = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)

best_val, best_state = -1.0, None
val_loader = DataLoader(DS(split["val"],mode="rgbd",out_size=(256,256),augment=False),
                        batch_size=8,shuffle=False,num_workers=0)
for ep in range(18):
    model.train()
    for x,y in tr:
        x,y = x.to(DEVICE),y.to(DEVICE)
        xn = norm_in(x)
        loss = dice(model(xn),y)+bce(model(xn),y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); iou=0.0; nb=0
    with torch.no_grad():
        for x,y in val_loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            xn=norm_in(x); p=(torch.sigmoid(model(xn))>=0.5).float()
            tp=(p*y).sum().item(); fp=(p*(1-y)).sum().item(); fn=((1-p)*y).sum().item()
            iou+=tp/(tp+fp+fn+1e-7); nb+=1
    val_iou=iou/max(nb,1)
    if val_iou>best_val: best_val=val_iou; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    print(f"  ep{ep+1:02d} val={val_iou:.4f}", flush=True)

trained_w = best_state["encoder.conv1.weight"]
rgb_drift   = (trained_w[:,:3] - init_w[:,:3]).abs().mean().item()
depth_drift = (trained_w[:,3:4] - init_w[:,3:4]).abs().mean().item()

result = {"rgb_drift":rgb_drift,"depth_drift":depth_drift,"ratio":depth_drift/rgb_drift,"best_val":best_val}
out = os.path.join(HERE,"..","results","pretrained_drift.json")
with open(out,"w") as f: json.dump(result,f,indent=2)
print(f"\nRGB drift:   {rgb_drift:.5f}")
print(f"Depth drift: {depth_drift:.5f}")
print(f"Ratio d/RGB: {depth_drift/rgb_drift:.3f}")
print(f"Saved -> {out}")
