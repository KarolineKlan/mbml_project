import torch
import pyro
import pyro.distributions as dist

# Assume data tensors:
# x: [N, D_x] text embeddings
# d: [N, D_d] demographic features
# time: [N] continuous outcome
# visits: [N] count outcome

torch.set_default_dtype(torch.float64)

# Model definition
def MixtureModel(x, d, time, visits, G=10, device="cpu"):
    x = x.to(dtype=torch.float64, device=device)
    d = d.to(dtype=torch.float64, device=device)
    time = time.to(dtype=torch.float64, device=device)
    visits = visits.to(dtype=torch.float64, device=device)


    N, D_x = x.shape
    _, D_d = d.shape

    # Dirichlet prior over cluster proportions
    alpha = torch.ones(G, device=device, dtype=torch.float64)
    pi = pyro.sample("pi", dist.Dirichlet(alpha))

    # Group-specific parameters
    with pyro.plate("group", G):
        # Regression weights for time
        beta_time0 = pyro.sample("beta_time0", dist.Normal(torch.tensor(0., device=device), torch.tensor(1., device=device)))
        beta_time_x = pyro.sample("beta_time_x", dist.Normal(torch.zeros(D_x, device=device), 1.).to_event(1))
        beta_time_d = pyro.sample("beta_time_d", dist.Normal(torch.zeros(D_d, device=device), 1.).to_event(1))
        sigma_time = pyro.sample("sigma_time", dist.HalfCauchy(scale=torch.tensor(5., device=device)))

        # Regression weights for visits (log-rate)
        beta_vis0 = pyro.sample("beta_vis0", dist.Normal(torch.tensor(0., device=device), torch.tensor(1., device=device)))
        beta_vis_x = pyro.sample("beta_vis_x", dist.Normal(torch.zeros(D_x, device=device), 1.).to_event(1))
        beta_vis_d = pyro.sample("beta_vis_d", dist.Normal(torch.zeros(D_d, device=device), 1.).to_event(1))

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

        mu_time = bt0 + (btx * x).sum(-1) + (btd * d).sum(-1)
        log_lambda = bv0 + (bvx * x).sum(-1) + (bvd * d).sum(-1)
        log_lambda = torch.clamp(log_lambda, -10.0, 10.0)

        # Observations
        pyro.sample("obs_time", dist.Normal(mu_time, s_time), obs=time)
        pyro.sample("obs_visits", dist.Poisson(log_lambda.exp()), obs=visits)

# Guide (Mean-field VI)
def MixtureModelGuide(x,d, time=None, visits=None, G=10, device="cpu"):
    x = x.to(device=device, dtype=torch.float64)
    d = d.to(device=device, dtype=torch.float64)

    N, D_x = x.shape
    _, D_d = d.shape
    
    # Learnable Dirichlet concentration
    q_alpha = pyro.param("q_alpha", torch.ones(G, dtype=torch.float64, device=device), constraint=dist.constraints.positive)
    pi = pyro.sample("pi", dist.Dirichlet(q_alpha))

    # Group params
    with pyro.plate("group", G):
        for name, shape, constraint in [
            ("beta_time0", [G], None),
            ("beta_vis0", [G], None)
        ]:
            pyro.param(f"loc_{name}", torch.zeros(*shape, device=device , dtype=torch.float64))
            pyro.param(f"scale_{name}", torch.ones(*shape, device=device, dtype=torch.float64), constraint=dist.constraints.positive)
            pyro.sample(name, dist.Normal(pyro.param(f"loc_{name}"), pyro.param(f"scale_{name}")))

        #pyro.param("loc_sigma_time", torch.zeros(G))
        pyro.param("scale_sigma_time", torch.ones(G, device=device, dtype=torch.float64), constraint=dist.constraints.positive)
        pyro.sample("sigma_time", dist.HalfCauchy(pyro.param("scale_sigma_time")))

        pyro.param("loc_beta_time_x", torch.zeros(G, D_x, dtype=torch.float64, device=device))
        pyro.param("scale_beta_time_x", torch.ones(G, D_x, dtype=torch.float64, device=device), constraint=dist.constraints.positive)
        pyro.sample("beta_time_x", dist.Normal(pyro.param("loc_beta_time_x"), pyro.param("scale_beta_time_x")).to_event(1))

        pyro.param("loc_beta_time_d", torch.zeros(G, D_d, dtype=torch.float64, device=device))
        pyro.param("scale_beta_time_d", torch.ones(G, D_d, dtype=torch.float64, device=device), constraint=dist.constraints.positive)
        pyro.sample("beta_time_d", dist.Normal(pyro.param("loc_beta_time_d"), pyro.param("scale_beta_time_d")).to_event(1))

        pyro.param("loc_beta_vis_x", torch.zeros(G, D_x,dtype=torch.float64, device=device))
        pyro.param("scale_beta_vis_x", torch.ones(G, D_x,dtype=torch.float64, device=device), constraint=dist.constraints.positive)
        pyro.sample("beta_vis_x", dist.Normal(pyro.param("loc_beta_vis_x"), pyro.param("scale_beta_vis_x")).to_event(1))

        pyro.param("loc_beta_vis_d", torch.zeros(G, D_d,dtype=torch.float64, device=device))
        pyro.param("scale_beta_vis_d", torch.ones(G, D_d,dtype=torch.float64, device=device), constraint=dist.constraints.positive)
        pyro.sample("beta_vis_d", dist.Normal(pyro.param("loc_beta_vis_d"), pyro.param("scale_beta_vis_d")).to_event(1))
    
    with pyro.plate("data", N):
        pyro.sample("z", dist.Categorical(pi))