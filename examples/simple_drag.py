import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

if False:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.layer = nn.Linear(dim, dim)
        nn.init.normal_(self.layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.layer.bias, val=0)

    def forward(self, t, x):
        out = self.layer(x)
        return self.layer(x)


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc

    def forward(self, steps: int, x: torch.Tensor, delta_t: float):
        time_steps = torch.linspace(0, steps * delta_t, steps, dtype=x.dtype)
        return odeint(
            self.odefunc,
            x,
            time_steps,
        ).transpose(0, 1)


class AirResistance(nn.Module):
    def __init__(self, coef=0.99):
        super(AirResistance, self).__init__()
        self.coef = coef - 1

    def forward(self, steps: int, x: torch.Tensor, delta_t: float):
        out = torch.zeros(size=(x.shape[0], steps), dtype=x.dtype)
        out[:, 0] = x[:, 0]
        for step in range(1, steps):
            prev_vel = out[:, step - 1]
            out[:, step] = prev_vel + prev_vel * self.coef * delta_t
        return out.reshape(out.shape[0], out.shape[1], 1)


def test_run():
    sim = AirResistance()
    model = ODEBlock(ODEfunc(1))
    initial_vals = torch.tensor([[1], [10]], dtype=torch.float32)
    steps = 1000
    delta_t = 1.0
    data = sim(steps, initial_vals, delta_t).detach().numpy()
    pred = model(steps, initial_vals, delta_t).detach().numpy()
    print(pred)
    print(pred.shape)
    print(data.shape)
    for d,p in zip(data, pred):
        plt.figure(0)
        plt.plot(d)
        plt.figure(1)
        plt.plot(p)
        plt.show()
    
def main():
  initial_vals = 1000 * torch.randn(10, 1, dtype=torch.float32)
  sim = AirResistance()
  model = ODEBlock(ODEfunc(1))

  steps = 100
  delta_t = 10.0
  epochs = 100

  optimizer = torch.optim.Adam(model.parameters())
  loss_func = torch.nn.functional.mse_loss

  for epoch in range(epochs):
    optimizer.zero_grad()
    target = sim(steps, initial_vals, delta_t)
    pred = model(steps, initial_vals, delta_t)
    loss = loss_func(pred, target)
    loss.backward()
    optimizer.step()
    print(loss)
  
  test = torch.tensor([[100]], dtype=torch.float32)
  target = sim(steps, test, delta_t).detach().numpy()
  pred = model(steps, test, delta_t).detach().numpy()
  for d,p in zip(target, pred):
        plt.plot(d)
        plt.plot(p)
        plt.show()

    

    

main()
