# Influenza model: 80% vaccination coverage parameter sweep.
coverage_campaign = 0.80
include(joinpath(@__DIR__, "..", "models", "influenza_model.jl"))
run_influenza_sweep()
