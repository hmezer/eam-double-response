import numpy as np

def lca_simulation(
    v_0,     # Drift rate for accumulator 0 (single float)
    v_1,     # Drift rate for accumulator 1 (single float)
    a,       # Decision threshold (single float)
    ndt,     # Non-decision time (single float)
    la,      # Leakage rate (single float)
    ka,      # Lateral inhibition rate (single float)
    s=1.0,   # Diffusion noise scale (single float), typically fixed or inferred
    dt=0.001,# Time step size (in seconds, float)
    max_sim_time=5.0, # Maximum total simulation time (in seconds, float)
    double_response_window=0.25, # Time window for detecting double responses (in seconds, float)
    return_path=False # Boolean flag to control path tracking and output
):
    """
    Simulates a single trial of a Leaky Competing Accumulator (LCA) model,
    extended to account for double responses, for exactly TWO alternatives.

    Instead of one accumulator moving between two boundaries, it uses two
    accumulators, each representing a choice option, competing with each other
    through inhibition and decaying via leakage, racing to hit a common threshold.

    Parameters:
    -----------
    v_0 : float
        Drift rate for decision alternative 0.
    v_1 : float
        Drift rate for decision alternative 1.
    a : float
        Decision threshold. Must be positive.
        (Analogous to DDM's 'a').
    ndt : float
        Non-decision time (in seconds). Must be non-negative.
        (Analogous to DDM's 'ter').
    la : float
        Leakage rate. Evidence decays towards zero. Must be non-negative. (NEW LCA parameter)
    ka : float
        Lateral inhibition rate. Accumulators inhibit each other. Must be non-negative. (NEW LCA parameter)
    s : float, optional
        Scale of the diffusion noise for each accumulator. Default is 1.0.
        (Analogous to DDM's 's').
    dt : float, optional
        Time step (in seconds) for the discrete simulation. Default is 0.001 (1 ms).
    max_sim_time : float, optional
        Maximum total simulation time (in seconds) to prevent infinite loops. Default is 5.0s.
    double_response_window : float, optional
        Time window (in seconds) after the initial response during which a second
        response is considered a 'double response'. Default is 0.25s (250ms).
    return_path : bool, optional
        Whether to track and return the accumulation path. Default is False.

    Returns:
    --------
    dict:
        If return_path=False:
            initial_rt : float
                The simulated initial response time (RT) for the trial (in seconds).
                Returns np.nan if max_sim_time is reached without an initial decision.
            initial_choice : int
                The index of the initially chosen alternative (0 or 1).
                Returns -1 if max_sim_time is reached without an initial decision.
            has_double_response : int
                1 if a double response occurred, 0 otherwise.
            double_rt_relative : float
                The time from the initial response to the double response (in seconds).
                np.nan if no double response.
            double_choice : int
                The index of the double response alternative (0 or 1).
                -1 if no double response.
        
        If return_path=True, additionally includes:
            x_store : ndarray
                The full accumulation path history for both accumulators.
    """
    # Line 40: Fixed number of alternatives for this specific LCA implementation
    num_alternatives = 2 # Fixed to 2
    # Line 41: Initialize accumulator evidence levels to zero
    x = np.zeros(num_alternatives) # Accumulator evidence for each alternative
    # Line 42: Combine individual drift rates (v_0, v_1) into an array 'v'
    v = np.array([v_0, v_1]) # Create array from individual drift rates
    # Line 43: Calculate the standard deviation of the noise term per time step
    # This scales the noise correctly for a Wiener process (diffusion process)
    noise_std = s * np.sqrt(dt)    # Correct scaling for Wiener process noise

    # Lines 45-58: Input validation
    # These lines check if the input parameters are within valid ranges (e.g., positive thresholds, non-negative times/rates).
    # If any parameter is invalid, the function immediately returns NaN/default values.
    # This is good practice for robustness, especially when sampling from priors that might occasionally generate extreme values.
    if a <= 0 or ndt < 0 or s <= 0 or dt <= 0:
        base_return = {
            "init_rt": np.nan,
            "init_resp": -1,
            "dr_if": 0,
            "dr_rt": np.nan,
            "dr_resp": -1
        }
        if return_path:
            base_return["x_store"] = np.array([])
        return base_return
    
    if la < 0 or ka < 0: # Leakage and inhibition rates must be non-negative
        base_return = {
            "init_rt": np.nan,
            "init_resp": -1,
            "dr_if": 0,
            "dr_rt": np.nan,
            "dr_resp": -1
        }
        if return_path:
            base_return["x_store"] = np.array([])
        return base_return

    # Line 60: Initialize iteration counter (time steps)
    num_iter = 0
    # Line 61: Calculate maximum number of iterations based on max_sim_time and dt
    max_iter = int(max_sim_time / dt)

    # Line 63-64: Initialize variables to store results of the initial response
    initial_rt_acc = np.nan # Accumulation time for initial response
    initial_choice = -1

    # Line 66-67: Initialize path storage only if requested
    if return_path:
        x_store = np.zeros((max_iter, num_alternatives))

    # --- Phase 1: Accumulation for Initial Response ---
    # Line 70: Main simulation loop for the first response.
    # Continues as long as no accumulator has reached the threshold ('a') AND
    # the maximum number of iterations ('max_iter') has not been exceeded.
    while np.all(x < a) and num_iter < max_iter:
        # Line 71: Generate independent random noise for each accumulator for the current time step.
        noise = np.random.randn(num_alternatives) * noise_std
        # Line 72: Calculate the sum of evidence across all accumulators.
        # This is used for the lateral inhibition term.
        sum_x_others = np.sum(x)

        # Line 74: Loop through each accumulator to update its evidence
        for i in range(num_alternatives):
            # Line 76: Calculate the lateral inhibition term for the current accumulator 'i'.
            # It's the lateral inhibition rate 'ka' multiplied by the sum of evidence in *other* accumulators.
            inhibition_term = ka * (sum_x_others - x[i])

            # Line 80: Calculate the change in evidence (dx_i) for the current accumulator.
            # This is the core LCA equation:
            # dx = (drift - leakage*current_evidence - inhibition_from_others) * dt + noise
            dx_i = (v[i] - la * x[i] - inhibition_term) * dt + noise[i]

            # Line 81: Update the accumulator's evidence level.
            x[i] += dx_i

            # Line 84: Truncation at 0.
            # This ensures that evidence levels do not drop below zero. If an accumulator
            # falls below zero, its evidence is reset to zero. This is a common feature
            # of LCA models.
            x[i] = max(0.0, x[i])

            # Store path only if requested
            if return_path:
                x_store[num_iter, i] = x[i]

        # Line 90: Increment the time step counter.
        num_iter += 1

    # Lines 93-102: Check if an initial decision was made or if max_sim_time was reached.
    # If max_sim_time was reached without any accumulator hitting the threshold,
    # it means no decision was made, and NaN/default values are returned.
    if num_iter >= max_iter:
        base_return = {
            "init_rt": np.nan,
            "init_resp": -1,
            "dr_if": 0,
            "dr_rt": np.nan,
            "dr_resp": -1
        }
        if return_path:
            base_return["x_store"] = x_store
        return base_return # RT=NaN, no decision, no double response

    # Lines 105-106: If a decision was made, identify which accumulator crossed the threshold first.
    initial_choice = np.where(x >= a)[0][0]
    initial_rt_acc = num_iter * dt # Accumulation time for the first response

    # --- Phase 2: Continued Accumulation for Double Response ---
    # Lines 110-112: Initialize variables for double response tracking.
    has_double_response = 0
    double_rt_relative = np.nan
    double_choice = -1

    # Line 115: Calculate the absolute time when the double response detection window ends.
    # This is initial_rt_acc (time of first response) + double_response_window (e.g., 250 ms).
    double_response_end_time_abs = initial_rt_acc + double_response_window

    # Line 118: Continue simulation within the double response window.
    # The loop runs as long as the current time is within the double response window
    # AND the total simulation time has not exceeded max_sim_time.
    while num_iter * dt < double_response_end_time_abs and num_iter < max_iter:
        # Lines 119-120: Generate noise and sum of others' evidence for the current time step.
        noise = np.random.randn(num_alternatives) * noise_std
        sum_x_others = np.sum(x)

        # Line 122: Loop through each accumulator to update its evidence.
        for i in range(num_alternatives):
            # Lines 123-126: Update accumulator evidence (same LCA dynamics as Phase 1).
            inhibition_term = ka * (sum_x_others - x[i])
            dx_i = (v[i] - la * x[i] - inhibition_term) * dt + noise[i]
            x[i] += dx_i
            x[i] = max(0.0, x[i])

            # Store path only if requested
            if return_path:
                x_store[num_iter, i] = x[i]

            # Lines 129-133: Check for a double response.
            # A double response occurs if an accumulator *other than the initial choice*
            # crosses the threshold 'a'.
            if i != initial_choice and x[i] >= a:
                has_double_response = 1
                # Calculate the time of the double response relative to the initial response.
                double_rt_relative = (num_iter * dt) - initial_rt_acc
                double_choice = i # Record which alternative caused the double response
                break # Exit inner loop immediately if a double response is found

        # Lines 135-136: If a double response was found in the inner loop, exit the outer loop too.
        if has_double_response == 1:
            break

        # Line 138: Increment the overall time step counter.
        num_iter += 1

    # Line 141: Calculate the final initial response time by adding non-decision time.
    final_initial_rt = initial_rt_acc + ndt

    # Lines 143-150: Return the results as a dictionary.
    d = {
        "init_rt": final_initial_rt,
        "init_resp": initial_choice,
        "dr_if": has_double_response,
        "dr_rt": double_rt_relative,
        "dr_resp": double_choice
    }
    
    # Include path data only if requested
    if return_path:
        d["x_store"] = x_store

    return d
