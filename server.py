from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Network Class -----
class DynamicNeuralNet:
    def __init__(self, input_size, hidden_layers, output_size, lr=0.01, decay=0.0):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.lr = lr
        self.decay = decay
        self.epoch = 0
        self.init_params()

    def init_params(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            w = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2 / self.layers[i])
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def sigmoid_deriv(self, x): return x * (1 - x)

    def forward(self, X):
        activations = [X]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(activations[-1], w) + b
            a = self.sigmoid(z)
            activations.append(a)
        return activations

    def backward(self, activations, y_true):
        grads_w = [0] * len(self.weights)
        grads_b = [0] * len(self.biases)

        error = activations[-1] - y_true
        delta = error * self.sigmoid_deriv(activations[-1])

        for i in reversed(range(len(self.weights))):
            grads_w[i] = np.dot(activations[i].T, delta)
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)
            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_deriv(activations[i])

        # learning rate decay
        lr = self.lr * (1.0 / (1.0 + self.decay * self.epoch))
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i] -= lr * grads_b[i]

    def train(self, X, y, epochs):
        history = []
        for _ in range(epochs):
            activations = self.forward(X)
            loss = np.mean((y - activations[-1]) ** 2)
            self.backward(activations, y)
            self.epoch += 1
            history.append(float(loss))
        return history

    def predict(self, X):
        return self.forward(X)[-1]

    def get_params(self):
        return {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

# -------- API Schema --------

class TrainRequest(BaseModel):
    inputs: list[list[float]]
    targets: list[list[float]]
    hidden_layers: list[int]
    epochs: int
    learning_rate: float
    decay: float
    previous_params: dict | None = None

# -------- API Logic --------

@app.post("/train")
def train_model(req: TrainRequest):
    X = np.array(req.inputs)
    y = np.array(req.targets)

    # Initialize network
    network = DynamicNeuralNet(
        input_size=X.shape[1],
        hidden_layers=req.hidden_layers,
        output_size=y.shape[1],
        lr=req.learning_rate,
        decay=req.decay,
    )

    # Inject previous weights/biases if provided
    if req.previous_params:
        network.weights = [np.array(w) for w in req.previous_params["weights"]]
        network.biases = [np.array(b) for b in req.previous_params["biases"]]

    # Train the network
    losses = network.train(X, y, req.epochs)
    prediction = network.predict(X).tolist()
    return {
        "losses": losses,
        "prediction": prediction,
        "params": network.get_params(),
    }

@app.get("/ping")
def ping():
    return {"status": "ok"}
