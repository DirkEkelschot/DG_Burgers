import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import matplotlib.pyplot as plt

import numpy as np





def train_pinn(rng, num_epochs=5000, batch_size=256):
    # ... [data generation code remains same as previous fix] ...
    # Generate collocation points
    x = jax.random.uniform(rng, (10000, 1), minval=-1, maxval=1)  # Spatial domain [-1,1]
    t = jax.random.uniform(rng, (10000, 1), minval=0, maxval=1.4)   # Temporal domain [0,1]
    # Remove this line: 
    # batch = (x_batch, t_batch, u_ic, x_ic, t_ic, x_bc, t_bc)  # DELETE THIS
    
    # Create training state
    state = create_train_state(rng)
    
    # Training loop
    for epoch in range(num_epochs):
        # Create batch for PDE collocation points
        key, rng = jax.random.split(rng)
        idx = jax.random.choice(key, x.shape[0], (batch_size,))
        x_batch = x[idx]  # Now defined here
        t_batch = t[idx]  # Now defined here

        key, rng = jax.random.split(rng)
        x_ic = jax.random.uniform(key, (1000, 1), minval=-1, maxval=1)
        t_ic = jnp.zeros_like(x_ic)
        u_ic = -jnp.sin(jnp.pi * x_ic)  # IC for Burgers equation
        
        # Boundary conditions (x=-1 and x=1)
        key, rng = jax.random.split(rng)
        x_bc = jnp.concatenate([
            -jnp.ones((500, 1)),  # Left boundary
            jnp.ones((500, 1))    # Right boundary
        ])
        t_bc = jax.random.uniform(key, (1000, 1), minval=0, maxval=1)
        
        # Create full BC arrays (for visualization)
        x_bc_full = jnp.concatenate([-jnp.ones((100, 1)), jnp.ones((100, 1))])
        t_bc_full = jnp.linspace(0, 1, 200).reshape(-1, 1)
        
        # Create batch INSIDE LOOP after defining x_batch/t_batch
        batch = (
            x_batch, 
            t_batch, 
            u_ic, 
            x_ic, 
            t_ic, 
            x_bc_full, 
            t_bc_full
        )
        
        state, loss = train_step(state, batch)
        
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss:.4e}")
            plot_solution(state.params, BurgersPINN())
    
    return state



def plot_solution(params, model, x_range=(-1, 1), t_range=(0, 1.4), resolution=100):
    """Visualize the PINN solution as a space-time contour plot"""
    x_vals = jnp.linspace(x_range[0], x_range[1], resolution)
    t_vals = jnp.linspace(t_range[0], t_range[1], resolution)
    
    # Create grid and flatten for batch prediction
    X, T = jnp.meshgrid(x_vals, t_vals)
    X_flat = X.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)
    
    # Get predictions
    u_pred = model.apply(params, X_flat, T_flat)
    U = u_pred.reshape(X.shape)
    
    # Convert to numpy for plotting
    X_np, T_np, U_np = map(np.array, (X, T, U))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    levels = np.linspace(U_np.min(), U_np.max(), 50)
    cs = plt.contourf(X_np, T_np, U_np, levels=levels, cmap='viridis')
    plt.colorbar(cs, label='u(x,t)')
    plt.xlabel('Spatial coordinate (x)', fontsize=12)
    plt.ylabel('Temporal coordinate (t)', fontsize=12)
    plt.title('PINN Solution to 1D Burgers Equation', fontsize=14)
    plt.tight_layout()
    plt.show()

# Define neural network architecture
class BurgersPINN(nn.Module):
    @nn.compact
    def __call__(self, x, t):
        inputs = jnp.concatenate([x, t], axis=-1)
        x = nn.Dense(32)(inputs)
        x = nn.tanh(x)
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        outputs = nn.Dense(1)(x)
        return outputs

# Physics-informed loss function
def compute_loss(params, apply_fn, batch):  # Changed 'model' to 'apply_fn'
    x, t, u_ic, x_ic, t_ic, x_bc, t_bc = batch
    
    # PDE residual
    def residual(params, x, t):
        u = apply_fn(params, x, t)  # Directly use apply_fn
        u_x = jax.grad(lambda x: apply_fn(params, x, t).sum())(x)
        u_xx = jax.grad(lambda x: u_x.sum())(x)
        u_t = jax.grad(lambda t: apply_fn(params, x, t).sum())(t)
        return u_t + u * u_x - (0.002) * u_xx
    
    # Rest of compute_loss remains the same
    r = residual(params, x, t)
    pde_loss = jnp.mean(r**2)
    
    # Initial condition (corrected apply_fn usage)
    u_pred_ic = apply_fn(params, x_ic, t_ic)
    ic_loss = jnp.mean((u_pred_ic - u_ic)**2)
    
    # Boundary condition (corrected apply_fn usage)
    u_pred_bc = apply_fn(params, x_bc, t_bc)
    bc_loss = jnp.mean(u_pred_bc**2)
    
    total_loss = pde_loss + ic_loss + bc_loss
    return total_loss



# Training setup
def create_train_state(rng, learning_rate=1e-3):
    model = BurgersPINN()
    params = model.init(rng, jnp.ones((1,)), jnp.ones((1,)))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

# JIT compile training step
@jax.jit
def train_step(state, batch):
    # loss_fn = lambda params: compute_loss(params, state.apply_fn, batch)
    loss_fn = lambda params: compute_loss(params, state.apply_fn, batch)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def main():
    # Initialize and run training
    rng = jax.random.PRNGKey(0)
    trained_state = train_pinn(rng, num_epochs=5000)

    # Final visualization
    plot_solution(trained_state.params, BurgersPINN())

if __name__ == "__main__":
    main()
    