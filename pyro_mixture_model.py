import torch
import pyro
import math
import pyro.distributions as dist
from pyro.infer import config_enumerate

# Assume data tensors:
# x: [N, D_x] text embeddings
# d: [N, D_d] demographic features
# v: [N] visitation duration, continuous outcome
# c: [N] visitation count, positive integer outcome

torch.set_default_dtype(torch.float64)

# Model definition
@config_enumerate(default="parallel")              # ← enable enumeration
def MixtureModel(x, d, v, c, G=10, device="cpu", batch_size=None):
    x = x.to(dtype=torch.float64, device=device)
    d = d.to(dtype=torch.float64, device=device)
    if v is not None:
        v = v.to(dtype=torch.float64, device=device)
    if c is not None:
        c = c.to(dtype=torch.float64, device=device)

    R, D_x = x.shape
    _, D_d = d.shape

    # Dirichlet prior over cluster proportions
    alpha = torch.ones(G, device=device, dtype=torch.float64)
    theta = pyro.sample("theta", dist.Dirichlet(alpha))

    # Group-specific parameters
    with pyro.plate("group", G):
        # Regression weights for visitation duration v
        beta0 = pyro.sample("beta0", dist.Normal(torch.tensor(0., device=device), torch.tensor(1., device=device)))
        beta_x = pyro.sample("beta_x", dist.Normal(torch.zeros(D_x, device=device), 1.).to_event(1))
        beta_d = pyro.sample("beta_d", dist.Normal(torch.zeros(D_d, device=device), 1.).to_event(1))
        sigma = pyro.sample("sigma", dist.HalfCauchy(scale=torch.tensor(5., device=device)))

        # Regression weights for count of c (log-rate)
        prior_mean_count = 1.3  # Adjust this based on your data
        gamma0 = pyro.sample("gamma0", dist.Normal(torch.tensor(math.log(prior_mean_count), device=device), torch.tensor(1., device=device)))
        gamma_x = pyro.sample("gamma_x", dist.Normal(torch.zeros(D_x, device=device), 0.1).to_event(1))
        gamma_d = pyro.sample("gamma_d", dist.Normal(torch.zeros(D_d, device=device), 0.1.).to_event(1))

    with pyro.plate("records", R, subsample_size=batch_size) as ind:
        x_b, d_b = x[ind], d[ind]
        
        if v is not None:
            v_b = v[ind]
        else:
            v_b = None

        if c is not None:
            c_b = c[ind]
        else:
            c_b = None

        # Mixture assignment
        g = pyro.sample("g", dist.Categorical(theta), infer={"enumerate":"parallel"})

        # Select parameters for each datum
        bt0 = beta0[g]
        btx = beta_x[g]
        btd = beta_d[g]
        sigma = sigma[g]

        bv0 = gamma0[g]
        bvx = gamma_x[g]
        bvd = gamma_d[g]

        mu_v    = bt0 + (btx * x_b).sum(-1) + (btd * d_b).sum(-1)
        log_lambda = bv0 + (bvx * x_b).sum(-1) + (bvd * d_b).sum(-1)
        log_lambda = torch.clamp(log_lambda, -5.0, 5.4)
        #mu_v = torch.clamp(mu_v, -3.0, 4.0)
        
        # Observations
        
        #pyro.sample("obs_v", dist.LogNormal(mu_v, sigma), obs=v)
        pyro.sample("obs_logv", dist.Normal(mu_v, sigma), obs=v_b)
        pyro.sample("obs_c", dist.Poisson(log_lambda.exp()), obs=c_b)
        

# Guide (Mean-field VI)
def MixtureModelGuide(x,d, v=None, c=None, G=10, device="cpu", batch_size=None):
    x = x.to(device=device, dtype=torch.float64)
    d = d.to(device=device, dtype=torch.float64)

    N, D_x = x.shape
    _, D_d = d.shape
    
    # Learnable Dirichlet concentration
    q_alpha = pyro.param("q_alpha", torch.ones(G, dtype=torch.float64, device=device), constraint=dist.constraints.positive)
    theta = pyro.sample("theta", dist.Dirichlet(q_alpha))

    # Group params
    with pyro.plate("group", G):
        for name, shape, constraint in [
            ("beta0", [G], None),
            ("gamma0", [G], None)
        ]:
            pyro.param(f"loc_{name}", torch.zeros(*shape, device=device , dtype=torch.float64))
            pyro.param(f"scale_{name}", torch.ones(*shape, device=device, dtype=torch.float64), constraint=dist.constraints.positive)
            pyro.sample(name, dist.Normal(pyro.param(f"loc_{name}"), pyro.param(f"scale_{name}")))

        #pyro.param("loc_sigma", torch.zeros(G))
        pyro.param("scale_sigma", torch.ones(G, device=device, dtype=torch.float64), constraint=dist.constraints.positive)
        pyro.sample("sigma", dist.HalfCauchy(pyro.param("scale_sigma")))

        pyro.param("loc_beta_x", torch.zeros(G, D_x, dtype=torch.float64, device=device))
        pyro.param("scale_beta_x", torch.ones(G, D_x, dtype=torch.float64, device=device), constraint=dist.constraints.positive)
        pyro.sample("beta_x", dist.Normal(pyro.param("loc_beta_x"), pyro.param("scale_beta_x")).to_event(1))

        pyro.param("loc_beta_d", torch.zeros(G, D_d, dtype=torch.float64, device=device))
        pyro.param("scale_beta_d", torch.ones(G, D_d, dtype=torch.float64, device=device), constraint=dist.constraints.positive)
        pyro.sample("beta_d", dist.Normal(pyro.param("loc_beta_d"), pyro.param("scale_beta_d")).to_event(1))

        pyro.param("loc_gamma_x", torch.zeros(G, D_x,dtype=torch.float64, device=device))
        pyro.param("scale_gamma_x", torch.ones(G, D_x,dtype=torch.float64, device=device), constraint=dist.constraints.positive)
        pyro.sample("gamma_x", dist.Normal(pyro.param("loc_gamma_x"), pyro.param("scale_gamma_x")).to_event(1))

        pyro.param("loc_gamma_d", torch.zeros(G, D_d,dtype=torch.float64, device=device))
        pyro.param("scale_gamma_d", torch.ones(G, D_d,dtype=torch.float64, device=device), constraint=dist.constraints.positive)
        pyro.sample("gamma_d", dist.Normal(pyro.param("loc_gamma_d"), pyro.param("scale_gamma_d")).to_event(1))
