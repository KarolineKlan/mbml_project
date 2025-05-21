import torch
import pyro
import pyro.distributions as dist


# Assume data tensors:
# x: [N, D_x] text embeddings
# d: [N, D_d] demographic features
# time: [N] continuous outcome
# visits: [N] count outcome
# Hyperparameters
K = 3  # number of clusters

# Model definition
def MixtureModel(x, d, time, visits):
    N, D_x = x.shape
    _, D_d = d.shape

    # Dirichlet prior over cluster proportions
    alpha = torch.ones(K)
    pi = pyro.sample("pi", dist.Dirichlet(alpha))

    # Group-specific parameters
    with pyro.plate("group", K):
        # Regression weights for time
        beta_time0 = pyro.sample("beta_time0", dist.Normal(0., 10.))
        beta_time_x = pyro.sample("beta_time_x", dist.Normal(torch.zeros(D_x), 1.).to_event(1))
        beta_time_d = pyro.sample("beta_time_d", dist.Normal(torch.zeros(D_d), 1.).to_event(1))
        sigma_time = pyro.sample("sigma_time", dist.HalfCauchy(scale=5.))

        # Regression weights for visits (log-rate)
        beta_vis0 = pyro.sample("beta_vis0", dist.Normal(0., 10.))
        beta_vis_x = pyro.sample("beta_vis_x", dist.Normal(torch.zeros(D_x), 1.).to_event(1))
        beta_vis_d = pyro.sample("beta_vis_d", dist.Normal(torch.zeros(D_d), 1.).to_event(1))

    with pyro.plate("data", N):
        # Mixture assignment
        z = pyro.sample("z", dist.Categorical(pi))

        # Select parameters for each datum
        bt0 = beta_time0[z]
        btx = beta_time_x[z]
        btd = beta_time_d[z]
        s_time = sigma_time[z]

        bv0 = beta_vis0[z]
        bvx = beta_vis_x[z]
        bvd = beta_vis_d[z]

        # Linear predictors
        mu_time = bt0 + (btx * x).sum(-1) + (btd * d).sum(-1)
        log_lambda = bv0 + (bvx * x).sum(-1) + (bvd * d).sum(-1)

        # Observations
        pyro.sample("obs_time", dist.Normal(mu_time, s_time), obs=time)
        pyro.sample("obs_visits", dist.Poisson(log_lambda.exp()), obs=visits)

# Guide (Mean-field VI)
def MixtureModelGuide(x,d, time=None, visits=None):
    N, D_x = x.shape
    _, D_d = d.shape
    # Learnable Dirichlet concentration
    q_alpha = pyro.param("q_alpha", torch.ones(K), constraint=dist.constraints.positive)
    pyro.sample("pi", dist.Dirichlet(q_alpha))

    # Group params
    for name, shape, constraint in [
        ("beta_time0", [K], None),
        ("sigma_time", [K], dist.constraints.positive),
        ("beta_vis0", [K], None)
    ]:
        pyro.param(f"loc_{name}", torch.randn(*shape))
        pyro.param(f"scale_{name}", torch.ones(*shape), constraint=dist.constraints.positive)
        pyro.sample(name, dist.Normal(pyro.param(f"loc_{name}"), pyro.param(f"scale_{name}")))

    # For weight vectors
    pyro.param("loc_beta_time_x", torch.randn(K, D_x))
    pyro.param("scale_beta_time_x", torch.ones(K, D_x), constraint=dist.constraints.positive)
    pyro.sample("beta_time_x", dist.Normal(pyro.param("loc_beta_time_x"), pyro.param("scale_beta_time_x")).to_event(2))

    pyro.param("loc_beta_time_d", torch.randn(K, D_d))
    pyro.param("scale_beta_time_d", torch.ones(K, D_d), constraint=dist.constraints.positive)
    pyro.sample("beta_time_d", dist.Normal(pyro.param("loc_beta_time_d"), pyro.param("scale_beta_time_d")).to_event(2))

    pyro.param("loc_beta_vis_x", torch.randn(K, D_x))
    pyro.param("scale_beta_vis_x", torch.ones(K, D_x), constraint=dist.constraints.positive)
    pyro.sample("beta_vis_x", dist.Normal(pyro.param("loc_beta_vis_x"), pyro.param("scale_beta_vis_x")).to_event(2))

    pyro.param("loc_beta_vis_d", torch.randn(K, D_d))
    pyro.param("scale_beta_vis_d", torch.ones(K, D_d), constraint=dist.constraints.positive)
    pyro.sample("beta_vis_d", dist.Normal(pyro.param("loc_beta_vis_d"), pyro.param("scale_beta_vis_d")).to_event(2))
