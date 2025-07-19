def sample_lca_prior():
    """
    Generates a single random draw from the joint prior distribution for LCA parameters.

    Returns:
    --------
    dict
        A dictionary containing a single sample for each LCA parameter.
    """
    # Drift rates (v_0, v_1): Normal(loc=0.0, scale=1.5)
    v_0 = np.random.normal(loc=0.0, scale=1.5)
    v_1 = np.random.normal(loc=0.0, scale=1.5)

    # Threshold (a): Uniform(low=0.5, high=2.5)
    a = np.random.uniform(low=0.5, high=2.5)

    # Non-decision time (ndt): Uniform(low=0.1, high=0.4)
    ndt = np.random.uniform(low=0.1, high=0.4)

    # Leakage rate (la): Uniform(low=0.0, high=0.5)
    la = np.random.uniform(low=0.0, high=0.5)

    # Lateral inhibition rate (ka): Uniform(low=0.0, high=0.5)
    ka = np.random.uniform(low=0.0, high=0.5)

    # Diffusion noise scale (s): Uniform(low=0.5, high=1.5)
    s = np.random.uniform(low=0.5, high=1.5)

    # Correct answer [0, 1] from bernoulli
    answer = np.random.binomial(1, 0.5)

    return {
        "v_0": v_0,
        "v_1": v_1,
        "a": a,
        "ndt": ndt,
        "la": la,
        "ka": ka,
        "s": s,
        "answer": answer
    }
