# Generate CSV data for Figure 4; plotting is done separately in Figure4_20251118.R.
# Change these parameters and output_tag to generate a different scenario.
coverage_campaign = 0.80
include(joinpath(@__DIR__, "..", "models", "influenza_model.jl"))

R0_single       = 3.0
forcing_single  = 0.05
nu_duration     = 2.0       # mean vaccine immunity duration in years
first_age_s     = 60.0
t_vac_start     = 200.0
output_tag      = "figure4_baseline"

β_single = gamma * R0_single
ν_single = 1.0 / nu_duration

mkpath(output_dir)
# First vaccination age in file names; non-integer ages use 'p' as decimal separator.
fva_age = isinteger(first_age_s) ? string(Int(first_age_s)) : replace(string(first_age_s), "." => "p")
fva_tag = "fva$(fva_age)"

mask_s = age_mask_from_first(age_points, first_age_s, end_age)
vac_s  = VacSched(mask_s, rate_in_window, camp_width, camp_phase, t_vac_start)

# solve single scenario with finer output for export: 20 time points per year
cache_s = ODECache(n_age, age_points, da, N0, gamma, forcing_single)
dt_out  = 1.0 / 20.0             # 20 points / year
prob_s  = ODEProblem(f_pde!, u0, (tmin, tmax),
                     (β_single, ν_single, contact_matrix, cache_s, vac_s, ALPHA_SCHED))
sol_s   = solve(prob_s, Tsit5(); reltol=1e-4, abstol=1e-6,
                callback=PositiveDomain(), saveat=dt_out, dense=false)

# unpack solution matrices
t     = sol_s.t
all_u = hcat(sol_s.u...)
S_mat  = all_u[1:n_age,                 :]'
I_mat  = all_u[n_age+1:2*n_age,         :]'
R1_mat = all_u[2*n_age+1:3*n_age,       :]'
V1_mat = all_u[3*n_age+1:4*n_age,       :]'
R2_mat = all_u[4*n_age+1:5*n_age,       :]'
V2_mat = all_u[5*n_age+1:6*n_age,       :]'
M_mat  = gamma .* (I_mat .* (ifr.(age_points))')  # instantaneous death *rate* per age bin

# totals over time (aggregated over age, using da as age-bin width)
tot_S = vec(sum(S_mat,  dims=2)) .* da
tot_I = vec(sum(I_mat,  dims=2)) .* da
tot_R = vec(sum(R1_mat .+ R2_mat, dims=2)) .* da
tot_V = vec(sum(V1_mat .+ V2_mat, dims=2)) .* da
tot_N = tot_S .+ tot_I .+ tot_R .+ tot_V

# total mortality incidence over time (aggregated over age)
mort_inc_total = vec(sum(M_mat, dims=2)) .* da   # integral over age

# export last 100 years
t_start_export = tmax - 100.0           # 100 years before tmax
idx_sel = findall(t .>= t_start_export)

# with dt_out=1/20, this gives 2001 points
if length(idx_sel) > 100 * 20
    idx_sel = idx_sel[2:end]
end

t_sel    = t[idx_sel]
S_sel    = tot_S[idx_sel]
I_sel    = tot_I[idx_sel]
R_sel    = tot_R[idx_sel]
V_sel    = tot_V[idx_sel]
N_sel    = tot_N[idx_sel]
M_sel    = mort_inc_total[idx_sel]

df_time = DataFrame(
    t     = t_sel,
    N     = N_sel,
    S     = S_sel,
    I     = I_sel,
    R     = R_sel,
    V     = V_sel,
    M_inc = M_sel,
)

CSV.write(joinpath(output_dir,
          "time_series_$(output_tag)_$(fva_tag).csv"),
          df_time)

# take same time points (last 100 years, 20/year)
I_sel_full = I_mat[idx_sel, :]

# one age value per year: 120 ages (0..119 years)
bins_per_year = Int(round(1 / da))   # e.g. 10 if da = 0.1
age_inds_year = 1:bins_per_year:n_age
ages_year     = age_points[age_inds_year]   # ~ 0,1,2,...,119

I_ta = I_sel_full[:, age_inds_year]  # (time × 120)

df_I = DataFrame()
df_I.t = t_sel
for (j, age) in enumerate(ages_year)
    colname = Symbol("age_$(Int(round(age)))")
    df_I[!, colname] = I_ta[:, j]
end

CSV.write(joinpath(output_dir,
          "time_age_$(output_tag)_$(fva_tag).csv"),
          df_I)
