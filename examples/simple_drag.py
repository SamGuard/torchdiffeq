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
        self.layers = nn.ModuleList()
        
        
        """
        self.layers.append(nn.Linear(dim, 32))
        self.layers.append(nn.Linear(32, 32))
        self.layers.append(nn.Linear(32, 32))
        self.layers.append(nn.Linear(32, dim))"""
        self.layers.append(nn.Linear(dim, dim))

        for l in self.layers:
            #nn.init.normal_(l.weight, mean=0, std=0.00001)
            nn.init.constant_(l.weight, 0.02)
            nn.init.constant_(l.bias, val=0)
        

    def forward(self, t, x):
        #x = torch.cat((x, (torch.zeros_like(x) + t)), dim=1)
        for l in self.layers:
            x = l(x)
        return x


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc

    def forward(self, steps: int, x: torch.Tensor, delta_t: float):
        time_steps = torch.linspace(0, (steps-1) * delta_t, steps, dtype=x.dtype)
        return odeint(
            self.odefunc,
            x,
            time_steps,
        ).transpose(0, 1)


class BallSimulation(nn.Module):
    def __init__(self, coef=1.0):
        super(BallSimulation, self).__init__()
        self.coef = coef

    """def forward(self, steps: int, x: torch.Tensor, delta_t: float):
        
        The equation to simulate, represents drag on a ball
        dv/dt = -kv
        v = a * e^(bt)
        a = v0
        b = -k/a
        

        a = x[:, 0]
        b = -self.coef / a

        out = torch.zeros(size=(x.shape[0], steps), dtype=x.dtype)
        out[:, 0] = x[:, 0]
        for step in range(1, steps):
            out[:, step] = a * torch.e ** (b * (step * delta_t))
        return out.reshape(out.shape[0], out.shape[1], 1)"""

    def forward(self, steps, x, delta_t):
        out = torch.zeros(size=(x.shape[0], steps), dtype=x.dtype)
        out[:, 0] = x[:, 0]
        for step in range(1, steps):
            t = delta_t * step
            out[:, step] = t ** 2 + x[:, 0]
        return out.reshape(out.shape[0], out.shape[1], 1)

    """def forward(self, steps, x, delta_t):
        out = torch.zeros(size=(x.shape[0], steps), dtype=x.dtype)
        out[:, 0] = x[:, 0]
        for step in range(1, steps):
            t = delta_t * step
            out[:, step] = t + x[:, 0]
        return out.reshape(out.shape[0], out.shape[1], 1)"""

def test():
    initial_vals = 1000 * torch.abs(torch.randn(100, 1, dtype=torch.double))
    sim = BallSimulation()
    model = ODEBlock(ODEfunc(1))

    steps = 100
    delta_t = 10.0

    target = sim(steps, initial_vals, delta_t)
    print(model(steps, initial_vals, delta_t) - target)

    for d in target:
        plt.plot(d)
        plt.show()


def main():
    sim = BallSimulation()
    model = ODEBlock(ODEfunc(1)).double()

    steps = 3
    delta_t = 1
    epochs = 1000

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = torch.nn.functional.mse_loss
    for epoch in range(epochs):
        initial_vals = 1000 * torch.abs(torch.randn(1000, 1, dtype=torch.double))
        optimizer.zero_grad()
        target = sim(steps, initial_vals, delta_t)
        pred = model(steps, initial_vals, delta_t)
        """print(target[0])
        print(pred[0])
        print("---")"""

        """plt.clf()
        print(pred[0])
        plt.plot(pred[0].detach().numpy())
        #plt.plot(target[0].detach().numpy())
        plt.pause(0.5)"""
        
        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()
        print("loss", loss.item())
        """for p in model.parameters():
            print("param", p.item())"""
    test = torch.tensor([[1], [10], [100]], dtype=torch.double)
    target = sim(100, test, delta_t).detach().numpy()
    pred = model(100, test, delta_t).detach().numpy()
    for d, p in zip(target, pred):
        print(d)
        print(p)
        plt.plot(d)
        plt.plot(p)
        plt.show()


main()
