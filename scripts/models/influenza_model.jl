import Pkg

# Project root: one level above this script's scripts/ directory
Pkg.activate(joinpath(@__DIR__, "..", ".."))

packages = [
    "DifferentialEquations",
    "DiffEqCallbacks",
    "CSV",
    "DataFrames",
    "QuadGK",
    "Distributions",
    "ThreadsX",
    "ProgressMeter",
    "Random",
]

# Install missing external packages
missing_packages = filter(name -> isnothing(Base.find_package(name)), packages)

if !isempty(missing_packages)
    Pkg.add(missing_packages)
end

for name in packages
    @eval using $(Symbol(name))
end

# Built-in modules: no installation needed
using LinearAlgebra
using LinearAlgebra: BLAS
using Dates
using Base.Threads

println("Julia threads : ", Threads.nthreads())
println("BLAS threads  : ", BLAS.get_num_threads())

# directories 
base_dir    = normpath(joinpath(@__DIR__, "..", ".."))
data_dir    = joinpath(base_dir, "data")
output_dir  = joinpath(base_dir, "output")

# time-age grid 
tmin, tmax  = 0.0, 250.0  # 200.0 for sweeps, 250 for runs 
amin, amax  = 0.0, 120.0
da          = 0.1
age_points  = collect(amin:da:(amax-da))
n_age       = length(age_points)

# transmission related parameters
gamma      = 50.0               # recovery rate (duration: 1 week)
# seasonal forcing amplitudes for the parameter sweep
forcing_vals = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

# R0 sweep 
R0_vals   = [1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 
             5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10] # influenza
beta_vals = gamma .* R0_vals

# influenza-style vaccination
# Coverage is supplied by the calling scenario before including this file.
camp_phase        = 10/12.0     # campaigns start in October
camp_width        = 2.0/12.0    # 2 month window (October–November)
rate_in_window    = -log(1 - coverage_campaign) / camp_width

end_age          = 100.0        # vaccinate up to and including this age
first_age_range  = vcat(collect(0.0:1.0:100.0), 200.0)  # 200 represents no vaccination

# vaccine immunity durations for sweep
nu_durations = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0] # influenza scenario
# nu_durations = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0] # general one-off
nu_vals      = 1.0 ./ nu_durations

# survival (scaled) used for initial age-density and to weight contact matrix
function survival_loglinear(age; a0=-5.054, a2=0.04695)
    return exp(-((10.0^a0)/(a2*log(10.0))) * (10.0^(a2*age) - 1.0))
end

# contact matrix
# read symmetric contact matrix and blow up to da grid
mat_df = CSV.read(joinpath(data_dir, "contact_matrix_data_120.txt"), DataFrame)
mat_df = filter(row -> row.part_age < 120 && row.cont_age < 120, mat_df)

n0 = Int(maximum(mat_df.part_age)) + 1
C0 = zeros(n0, n0)
for r in eachrow(mat_df)
    i0 = Int(r.part_age) + 1
    j0 = Int(r.cont_age) + 1
    C0[i0, j0] = r.c
end
C0 = (C0 + C0')/2

# expand to da bins
bins_per_year = Int(round(1/da))
N_da = n0 * bins_per_year
age_points = collect(amin:da:(amin + da*(N_da-1)))
n_age = length(age_points)

C = zeros(n_age, n_age)
for i0 in 1:n0, j0 in 1:n0
    i_inds = ((i0-1)*bins_per_year + 1):(i0*bins_per_year)
    j_inds = ((j0-1)*bins_per_year + 1):(j0*bins_per_year)
    C[i_inds, j_inds] .= C0[i0, j0]
end
C = (C + C')/2

# weigh by initial population to form next-generation-like matrix, then scale
popdist = survival_loglinear.(age_points)
K = C .* popdist' .* da
ρ = maximum(abs.(eigvals(K)))
contact_matrix = K ./ ρ

# infection fatality rate (probability) 
# doubling every 10 years, IFR(90)=0.01 and IFR(100)=0.02
ifr(age; a=1.86e-5, b=0.07, c=1.86e-5) = a*exp(b*age) - c

# background mortality 
@inline function mortality(t::Float64, age::Float64, a0::Float64, a1::Float64,
                           a2::Float64, twarmup::Float64)
    logm = t < twarmup ? (a0 + a1*5.0 + a2*age) : (a0 + a1*5.0 + a1*(t - twarmup) + a2*age) #2025
    #logm = t < twarmup ? (a0 + a1*30.0 + a2*age) : (a0 + a1*30.0 + a1*(t - twarmup) + a2*age) # 2050
    return 10.0^logm
end

# seasonal natural waning 
# seasons are [n, n+1), with influenza season action
# we draw season-specific durations from Uniform(2, 7) years
# season_edges = collect(tmin:1.0:tmax)  # edges at each integer year
season_edges = collect(tmin - 3.0/12.0 : 1.0 : tmax + 9.0/12.0) # shift edges to start in October
const NSEAS = length(season_edges) - 1

struct AlphaSched
    edges::Vector{Float64}   # season edges of length NSEAS+1
    rates::Vector{Float64}   # α per season (length NSEAS)
end

function build_alpha_sched(edges::Vector{Float64}; seed::Int=97)
    rng = MersenneTwister(seed)
    durations = rand(rng, Uniform(2.0, 7.0), length(edges)-1) # years
    rates = 1.0 ./ durations
    return AlphaSched(edges, rates)
end

@inline function alpha_t(t::Float64, as::AlphaSched)
    # find season index s such that edges[s] <= t < edges[s+1]
    s = clamp(searchsortedlast(as.edges, t), 1, length(as.rates))
    return as.rates[s]
end

const ALPHA_SCHED = build_alpha_sched(season_edges; seed=97)

# vaccination schedule (time window + age mask)
# 1 if age in [first_age, end_age], else 0
function age_mask_from_first(age_points::Vector{Float64}, first_age::Float64, end_age::Float64)
    m = zeros(Float64, length(age_points))
    @inbounds for i in eachindex(age_points)
        a = age_points[i]
        m[i] = (a >= first_age && a <= end_age) ? 1.0 : 0.0
    end
    return m
end

#= start vaccination at t=0
struct VacSched
    mask::Vector{Float64}    # 1 for eligible ages, else 0
    rate_in_window::Float64  # per-year rate during the window
    width::Float64           # length of yearly window (years)
    phase::Float64           # when it starts within the year (e.g., 0.5)
end

@inline function campaign_rate(t::Float64, vs::VacSched)
    τ = (t - vs.phase) % 1.0
    return (0.0 <= τ < vs.width) ? vs.rate_in_window : 0.0
end
=#

struct VacSched
    mask::Vector{Float64}    # 1 for eligible ages, else 0
    rate_in_window::Float64  # per-year rate during the window
    width::Float64           # length of yearly window (years)
    phase::Float64           # when it starts within the year (e.g., 0.5)
    t_start::Float64         # time (years) when vaccination starts
end

@inline function campaign_rate(t::Float64, vs::VacSched)
    # no vaccination before vs.t_start
    if t < vs.t_start
        return 0.0
    end
    τ = (t - vs.phase) % 1.0
    return (0.0 <= τ < vs.width) ? vs.rate_in_window : 0.0
end

# cache for ODE RHS 
struct ODECache
    N::Vector{Float64}
    I_div_N::Vector{Float64}
    FOI::Vector{Float64}
    ifr_vals::Vector{Float64}
    age_points::Vector{Float64}
    n_age::Int
    da::Float64
    N0::Float64
    gamma::Float64
    force_frac::Float64
    mu_a0::Float64
    mu_a1::Float64
    mu_a2::Float64
    mu_twarmup::Float64
end

function ODECache(n_age::Int, age_points::Vector{Float64}, da::Float64, N0::Float64,
                  gamma::Float64, force_frac::Float64)
    ifr_vals = [ifr(age) for age in age_points]
    return ODECache(
        zeros(n_age),
        zeros(n_age),
        zeros(n_age),
        ifr_vals,
        age_points,
        n_age,
        da,
        N0,
        gamma,
        force_frac,
        -5.054,     # a0
        -0.006121,  # a1
        0.04695,    # a2
        500.0       # warm-up end (years)
    )
end

# PDE RHS
function f_pde!(du, u, p, t)
    β, ν, contact_matrix, cache, vac, αsched = p

    n_age_c = cache.n_age
    da_c = cache.da
    N0_c = cache.N0
    γ = cache.gamma
    forc = cache.force_frac
    ifr_vals = cache.ifr_vals
    age = cache.age_points

    # season-specific natural waning rate
    α = alpha_t(t, αsched)

    # views
    S  = @view u[1:n_age_c]
    I  = @view u[n_age_c+1:2*n_age_c]
    R1 = @view u[2*n_age_c+1:3*n_age_c]
    V1 = @view u[3*n_age_c+1:4*n_age_c]
    R2 = @view u[4*n_age_c+1:5*n_age_c]
    V2 = @view u[5*n_age_c+1:6*n_age_c]

    N = cache.N; I_div_N = cache.I_div_N; FOI = cache.FOI
    @. N = S + I + R1 + R2 + V1 + V2 + 1e-6
    @. I_div_N = I / N
    mul!(FOI, contact_matrix, I_div_N)

    fsc = 1 + forc * cos(2π*t)
    vr_t = campaign_rate(t, vac)

    @inbounds begin
        μ = mortality(t, age[1], cache.mu_a0, cache.mu_a1, cache.mu_a2, cache.mu_twarmup)
        λ = β * fsc * FOI[1]
        vr_i = vr_t * vac.mask[1]
        dS_da  = (S[1] - N0_c)/da_c
        dI_da  =  I[1]/da_c
        dR1_da =  R1[1]/da_c
        dV1_da =  V1[1]/da_c
        dR2_da =  R2[1]/da_c
        dV2_da =  V2[1]/da_c
        du[1]           = -λ*S[1] + 2*α*R2[1] + 2*ν*V2[1] - vr_i*S[1] - μ*S[1] - dS_da
        du[n_age_c+1]   =  λ*S[1] - γ*I[1] - μ*I[1] - dI_da
        du[2*n_age_c+1] = (1 - ifr_vals[1]) * γ*I[1] - 2*α*R1[1] - μ*R1[1] - dR1_da
        du[3*n_age_c+1] =  vr_i*S[1] - 2*ν*V1[1] - μ*V1[1] - dV1_da
        du[4*n_age_c+1] =  2*α*R1[1] - 2*α*R2[1] - μ*R2[1] - dR2_da
        du[5*n_age_c+1] =  2*ν*V1[1] - 2*ν*V2[1] - μ*V2[1] - dV2_da
    end

    @inbounds for i in 2:n_age_c
        μ = mortality(t, age[i], cache.mu_a0, cache.mu_a1, cache.mu_a2, cache.mu_twarmup)
        λ = β * fsc * FOI[i]
        vr_i = vr_t * vac.mask[i]
        dS_da  = (S[i]  - S[i-1]) / da_c
        dI_da  = (I[i]  - I[i-1]) / da_c
        dR1_da = (R1[i] - R1[i-1]) / da_c
        dV1_da = (V1[i] - V1[i-1]) / da_c
        dR2_da = (R2[i] - R2[i-1]) / da_c
        dV2_da = (V2[i] - V2[i-1]) / da_c
        du[i]             = -λ*S[i] + 2*α*R2[i] + 2*ν*V2[i] - vr_i*S[i] - μ*S[i] - dS_da
        du[n_age_c+i]     =  λ*S[i] - γ*I[i] - μ*I[i] - dI_da
        du[2*n_age_c+i]   = (1 - ifr_vals[i]) * γ*I[i] - 2*α*R1[i] - μ*R1[i] - dR1_da
        du[3*n_age_c+i]   =  vr_i*S[i] - 2*ν*V1[i] - μ*V1[i] - dV1_da
        du[4*n_age_c+i]   =  2*α*R1[i] - 2*α*R2[i] - μ*R2[i] - dR2_da
        du[5*n_age_c+i]   =  2*ν*V1[i] - 2*ν*V2[i] - μ*V2[i] - dV2_da
    end
    return nothing
end

# life expectancy for YLL calculations, using cohort method
function life_expectancy(t0, age0, a0, a1, a2, twarmup)
    cond_surv(x_age) = begin
        integral_hz, _ = quadgk(a -> mortality(t0 + (a - age0), a, a0, a1, a2, twarmup),
                                age0, x_age; rtol=1e-6)
        return exp(-integral_hz)
    end
    integral_surv, _ = quadgk(cond_surv, age0, amax; rtol=1e-6)
    return integral_surv
end

# precalculate life expectancy at each age point at tmax
life_exp = [life_expectancy(tmax, a, -5.054, -0.006121, 0.04695, 500.0) for a in age_points]

# initial conditions
N0            = 200000.0
seed_fraction = 0.001
u0 = zeros(6*n_age)
for i in 1:n_age
    age = age_points[i]
    pop_i = N0 * survival_loglinear(age)
    frac_i = seed_fraction * (age/10)^1 * ((120-age)/110)^11
    I0 = frac_i * pop_i
    u0[i]       = pop_i - I0
    u0[n_age+i] = I0
end


# single run function
function run_one!(β::Float64, ν::Float64, first_age::Float64, forcing::Float64)
    # vaccination schdule, t_v=200 represents no vccination
    mask = first_age == 200.0 ? fill(0.0, n_age) : age_mask_from_first(age_points, first_age, end_age)
    vac  = VacSched(mask, rate_in_window, camp_width, camp_phase, tmin)  # vaccination allowed from t = tmin
    cache = ODECache(n_age, age_points, da, N0, gamma, forcing)
    prob  = ODEProblem(f_pde!, u0, (tmin, tmax), (β, ν, contact_matrix, cache, vac, ALPHA_SCHED))
    sol   = solve(prob, Tsit5(); reltol=1e-4, abstol=1e-6,  # -3 and -5 yield similar results; -4 and -5 identical
                  callback=PositiveDomain(), saveat=da, dense=false)
    
    # Average annual mortality, YLL and mean age of infection over last fifty years
    t_start_avg = tmax - 50.0  # last 50 years to average over seasonal variation 
    idx_last5y = findall(t -> t ≥ t_start_avg, sol.t)

    mort_vals = Float64[]
    yll_vals  = Float64[]
    mai_vals  = Float64[]         

    I_div_N = zeros(n_age)        
    FOI     = zeros(n_age)

    for i in idx_last5y
        u = sol.u[i]

        # state by age
        S  = @view u[1:n_age]
        I  = @view u[n_age+1:2*n_age]
        R1 = @view u[2*n_age+1:3*n_age]
        V1 = @view u[3*n_age+1:4*n_age]
        R2 = @view u[4*n_age+1:5*n_age]
        V2 = @view u[5*n_age+1:6*n_age]

        # mortality & YLL as before
        M = gamma .* (I .* ifr.(age_points))
        push!(mort_vals, sum(M) * da)
        push!(yll_vals,  sum(M .* life_exp) * da)

        # total population by age
        N = S .+ I .+ R1 .+ R2 .+ V1 .+ V2 .+ 1e-6

        @. I_div_N = I / N
        mul!(FOI, contact_matrix, I_div_N)  # FOI[a] up to a constant factor

        t_now = sol.t[i]
        fsc   = 1 + forcing * cos(2π * t_now)
        @. FOI = β * fsc * FOI               # λ(t,a) = β fsc * (C * I/N)

        weights = FOI .* S                  # λ(t,a) S(t,a)
        denom   = sum(weights) * da
        if denom > 0
            num = sum(age_points .* weights) * da
            push!(mai_vals, num / denom)    # mai(t) for this time point
        end
    end

    mean_mai = mean(mai_vals)

    return (
        R0 = β/gamma,
        beta = β,
        nu = ν,
        nu_duration_years = 1/ν,
        first_age = first_age,
        force_frac = forcing,
        total_mortality = mean(mort_vals),
        total_yll       = mean(yll_vals),
        mean_age_infection = mean_mai   
    )
end

# Sweep includes zero seasonality, the original 5% baseline, and stronger forcing.
function run_influenza_sweep(; forcing_values=forcing_vals)
    mkpath(output_dir)
    stamp = Dates.format(Dates.today(), "yyyymmdd")
    println("Output stamp: ", stamp)

    param_sets = [(β, ν, a0, forcing)
                  for forcing in forcing_values
                  for β in beta_vals
                  for ν in nu_vals
                  for a0 in first_age_range]

    prog = Progress(length(param_sets); desc="running sweeps", barlen=40, dt=5.0)
    tuples = ThreadsX.map(enumerate(param_sets)) do (i, ps)
        β, ν, a0, forcing = ps
        result = run_one!(β, ν, a0, forcing)
        next!(prog)
        result
    end

    df = DataFrame(tuples)
    df_mort = select(df, :beta, :R0, :nu, :nu_duration_years, :first_age, :force_frac, :total_mortality)
    df_yll  = select(df, :beta, :R0, :nu, :nu_duration_years, :first_age, :force_frac, :total_yll)

    coverage_tag = Int(round(100 * coverage_campaign))
    CSV.write(joinpath(output_dir, "mortality_influenza_2025_v50_forcing_sweep_coverage$(coverage_tag)_$(stamp).csv"), df_mort)
    CSV.write(joinpath(output_dir, "yll_influenza_2025_v50_forcing_sweep_coverage$(coverage_tag)_$(stamp).csv"), df_yll)

    println("Finished sweeps: ", nrow(df), " runs.")
    return df
end
