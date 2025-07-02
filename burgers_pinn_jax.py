import jax
import jax.numpy as jnp
import optax
import haiku as hk
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

# Set random seed
key = jax.random.PRNGKey(42)

# Neural network definition
def burgernet(x, t):
    inputs = jnp.concatenate([x, t], axis=1)
    mlp = hk.nets.MLP([64, 64, 64, 1], activation=jnp.tanh)
    return mlp(inputs)

# Transform network into pure functions
network = hk.without_apply_rng(hk.transform(burgernet))
params = network.init(key, jnp.zeros((1, 1)), jnp.zeros((1, 1)))

# PDE parameters
nu = 0.01 / jnp.pi

# Define PDE residual
def residual(params, x, t):
    u = network.apply(params, x, t)
    
    # First derivatives
    u_t = jax.grad(lambda t: network.apply(params, x, t).sum())(t)
    u_x = jax.grad(lambda x: network.apply(params, x, t).sum())(x)
    
    # Second derivative
    u_xx = jax.grad(lambda x: jax.grad(lambda x: network.apply(params, x, t).sum())(x))(x)
    
    return u_t + u * u_x - nu * u_xx

# Loss function components
def loss_fn(params, data):
    # PDE residual loss
    f_pred = residual(params, data['colloc_x'], data['colloc_t'])
    mse_f = jnp.mean(f_pred**2)
    
    # Initial condition loss
    u_pred_ic = network.apply(params, data['ic_x'], data['ic_t'])
    mse_ic = jnp.mean((u_pred_ic + jnp.sin(jnp.pi * data['ic_x']))**2)
    
    # Boundary condition loss
    u_pred_bc1 = network.apply(params, data['bc_x1'], data['bc_t'])
    u_pred_bc2 = network.apply(params, data['bc_x2'], data['bc_t'])
    mse_bc = jnp.mean(u_pred_bc1**2 + u_pred_bc2**2)
    
    return 0.8*mse_f + 0.1*mse_ic + 0.1*mse_bc

# Optimizer setup
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# Data sampling functions
def sample_initial_condition(key, n_samples):
    x = jax.random.uniform(key, (n_samples, 1), minval=-1, maxval=1)
    t = jnp.zeros((n_samples, 1))
    return x, t

def sample_boundary_condition(key, n_samples):
    t = jax.random.uniform(key, (n_samples, 1), minval=0, maxval=1)
    x1 = -jnp.ones((n_samples, 1))
    x2 = jnp.ones((n_samples, 1))
    return x1, x2, t

def sample_collocation_points(key, n_samples):
    x = jax.random.uniform(key, (n_samples, 1), minval=-1, maxval=1)
    t = jax.random.uniform(key, (n_samples, 1), minval=0,  maxval=1)
    return x, t

# Training step
@partial(jax.jit, static_argnums=(2,))
def train_step(params, opt_state, data):
    loss, grads = jax.value_and_grad(loss_fn)(params, data)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training data
def create_training_data(key, n_ic=500, n_bc=500, n_colloc=10000):
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Initial condition
    ic_x, ic_t = sample_initial_condition(key1, n_ic)
    
    # Boundary conditions
    bc_x1, bc_x2, bc_t = sample_boundary_condition(key2, n_bc)
    
    # Collocation points
    colloc_x, colloc_t = sample_collocation_points(key3, n_colloc)
    
    return {
        'ic_x': ic_x, 'ic_t': ic_t,
        'bc_x1': bc_x1, 'bc_x2': bc_x2, 'bc_t': bc_t,
        'colloc_x': colloc_x, 'colloc_t': colloc_t
    }

# Create training data
data = create_training_data(key)

# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    params, opt_state, loss = train_step(params, opt_state, data)
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d}, Loss: {loss:.4e}")

# Visualization
x_test = jnp.linspace(-1, 1, 100)[:, None]
t_test = jnp.linspace(0, 1, 50)[:, None]

X, T = jnp.meshgrid(x_test, t_test)
X_flat = X.reshape(-1, 1)
T_flat = T.reshape(-1, 1)

u_pred = network.apply(params, X_flat, T_flat).reshape(X.shape)

plt.figure(figsize=(10, 6))
plt.pcolormesh(T, X, u_pred, shading='auto', cmap='jet')
plt.colorbar(label='u(t,x)')
plt.xlabel('Time (t)')
plt.ylabel('Space (x)')
plt.title('Burgers Equation Solution with PINNs')
plt.show()
