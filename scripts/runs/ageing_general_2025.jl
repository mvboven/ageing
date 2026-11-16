# General ageing scenario 2025.
scenario_year = 2025
mortality_reference_year_offset = 5.0
R0_vals = [1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]
include(joinpath(@__DIR__, "..", "models", "ageing_general_model.jl"))
run_general_sweep()
