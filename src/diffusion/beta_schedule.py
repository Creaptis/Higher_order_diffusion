import flax
import jax.numpy as jnp


@flax.struct.dataclass
class BetaSchedule:
    """Class for keeping track of an item in inventory."""

    def beta_t(self, t):
        pass

    def integral_beta_t(self, t):
        pass


@flax.struct.dataclass
class LinearSchedule(BetaSchedule):
    beta_0: float
    beta_T: float
    t_0: float
    T: float

    def beta_t(self, t):
        return self.beta_0 + t * (self.beta_T - self.beta_0)

    def integral_beta_t(self, t):
        return self.beta_0 * t + 0.5 * t**2 * (self.beta_T - self.beta_0)


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    beta_schedule = LinearSchedule(beta_0=0.1, beta_T=0.9, t_0=0, T=1)
    beta_t = jax.vmap(beta_schedule.beta_t)
    integral_beta_t = jax.vmap(beta_schedule.integral_beta_t)
    t = jnp.linspace(0, 1, 100)
    plt.plot(t, beta_t(t))
    plt.plot(t, integral_beta_t(t))
    plt.show()
