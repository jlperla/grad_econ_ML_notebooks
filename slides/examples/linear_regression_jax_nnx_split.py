# Takes the baseline version and uses split/merge for LBFGS support
import jax
import jax.numpy as jnp
from jax import random
import optax
import jax_dataloader as jdl
from jax_dataloader.loaders import DataLoaderJAX
from flax import nnx

N = 500  # samples
M = 2
sigma = 0.001
rngs = nnx.Rngs(42)
theta = random.normal(rngs(), (M,))
X = random.normal(rngs(), (N, M))
Y = X @ theta + sigma * random.normal(rngs(), (N,))  # Adding noise

def residual(model, x, y):
    y_hat = model(x)
    return (y_hat - y) ** 2

def residuals_loss(model, X, Y):
    return jnp.mean(jax.vmap(residual, in_axes=(None, 0, 0))(model, X, Y))

model = nnx.Linear(M, 1, use_bias=False, rngs=rngs)

wrt = nnx.Param

# Advantage: a little faster since split, and can use LBFGS
lr = 0.001
optimizer = nnx.Optimizer(model,
                          optax.lbfgs(),
                        #optax.sgd(lr),
                          wrt=wrt)

# Following the split/merge pattern from stochastic_growth_nnx.py
# This avoids the old hack and works with Flax 0.12+
@nnx.jit
def step(graphdef, state, X, Y):
    model, optimizer = nnx.merge(graphdef, state)
    model_graphdef, _ = nnx.split(model)

    def loss_fn(model):
        return residuals_loss(model, X, Y)

    def loss_fn_split(state):
        model = nnx.merge(model_graphdef, state)
        return residuals_loss(model, X, Y)

    grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, wrt))
    loss_value, grads = grad_fn(model)
    optimizer.update(model, grads, value=loss_value, grad=grads, value_fn=loss_fn_split)
    _, state = nnx.split((model, optimizer))
    return loss_value, state

# Split into JAX compatible values for compilation/differentiation
graphdef, state = nnx.split((model, optimizer))

num_epochs = 20
batch_size = 1024
dataset = jdl.ArrayDataset(X, Y)
train_loader = DataLoaderJAX(dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader:
        loss, state = step(graphdef, state, X_batch, Y_batch)

    # Merge back to read model parameters
    model, optimizer = nnx.merge(graphdef, state)
    if epoch % 2 == 0:
        print(
            f"Epoch {epoch},||theta - theta_hat|| = {jnp.linalg.norm(theta - jnp.squeeze(model.kernel.value))}"
        )

model, optimizer = nnx.merge(graphdef, state)
print(f"||theta - theta_hat|| = {jnp.linalg.norm(theta - jnp.squeeze(model.kernel.value))}")
