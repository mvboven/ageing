# General ageing scenario 2050.
scenario_year = 2050
mortality_reference_year_offset = 30.0
R0_vals = [1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7.5, 10]
include(joinpath(@__DIR__, "..", "models", "ageing_general_model.jl"))
run_general_sweep()
