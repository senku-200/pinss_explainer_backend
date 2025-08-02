from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Model Definition --------
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        self.weight_layers = []
        self.bias_layers = []
        for h in hidden_layers:
            linear = nn.Linear(prev_dim, h)
            layers.append(linear)
            layers.append(nn.Tanh())
            self.weight_layers.append(linear.weight)
            self.bias_layers.append(linear.bias)
            prev_dim = h
        final = nn.Linear(prev_dim, output_dim)
        layers.append(final)
        self.weight_layers.append(final.weight)
        self.bias_layers.append(final.bias)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def pde_residual(model, x, decay_lambda):
    x.requires_grad_(True)
    u = model(x)
    du_dt = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    residual = du_dt + decay_lambda * u
    return residual

# -------- Request Schema --------
class TrainPINNRequest(BaseModel):
    inputs: list[list[float]]
    targets: list[list[float]]
    inputs_collocation: list[list[float]]
    hidden_layers: list[int]
    epochs: int
    learning_rate: float
    decay: float = 0.0
    lambda_pde: float = 1.0
    previous_params: dict | None = None

# -------- Training Endpoint --------
@app.post("/train")
def train_pinn(req: TrainPINNRequest):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_data = torch.tensor(req.inputs, dtype=torch.float32, device=device)
    y_data = torch.tensor(req.targets, dtype=torch.float32, device=device)
    X_colloc = torch.tensor(req.inputs_collocation, dtype=torch.float32, device=device)

    input_dim = X_data.shape[1]
    output_dim = y_data.shape[1]
    model = PINN(input_dim, req.hidden_layers, output_dim).to(device)

    if req.previous_params:
        with torch.no_grad():
            weights = req.previous_params.get("weights", [])
            biases = req.previous_params.get("biases", [])
            for param, w in zip(model.weight_layers, weights):
                param.copy_(torch.tensor(w, dtype=torch.float32, device=device))
            for param, b in zip(model.bias_layers, biases):
                param.copy_(torch.tensor(b, dtype=torch.float32, device=device))

    optimizer = optim.Adam(model.parameters(), lr=req.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0 / (1.0 + req.decay))
    mse = nn.MSELoss()
    losses = []

    for epoch in range(req.epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_data)
        data_loss = mse(y_pred, y_data)

        residual = pde_residual(model, X_colloc, req.lambda_pde)
        physics_loss = mse(residual, torch.zeros_like(residual))

        loss = data_loss + req.lambda_pde * physics_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        prediction = model(X_data).cpu().numpy().tolist()
        weights = [param.detach().cpu().numpy().tolist() for param in model.weight_layers]
        biases = [param.detach().cpu().numpy().tolist() for param in model.bias_layers]

    return {
        "losses": losses,
        "prediction": prediction,
        "params": {
            "weights": weights,
            "biases": biases
        }
    }

@app.get("/ping")
def ping():
    return {"status": "ok"}
