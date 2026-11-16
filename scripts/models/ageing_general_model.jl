# Shared general ageing model; the run script supplies scenario_year,
# mortality_reference_year_offset and R0_vals.
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

BLAS.set_num_threads(1)                                     # no matrix multithreading

# print number of threads
println("Julia threads : ", Threads.nthreads())
println("BLAS threads. :  ", BLAS.get_num_threads())

# directories
base_dir    = normpath(joinpath(@__DIR__, "..", ".."))
data_dir    = joinpath(base_dir, "data")
output_dir  = joinpath(base_dir, "output")
 
# Create destination folders if needed
mkpath(output_dir)

# time-age parameters
tmin, tmax  = 0.0, 200.0                                   # simulation duration (years)
twarm       = 200.0                                        # warmup period (years) - this impacts YLL
amin, amax  = 0.0, 120.0                                   # age range (years)
da          = 0.1                                          # age discretisation (years)
age_points  = collect(amin:da:(amax-da))
n_age       = length(age_points)

# transmission parameters
beta       = 125.0                                         # transmission rate parameter (modified in sweeps)
gamma      = 50.0                                          # recovery rate (usually approx 1 week, 1/50 year)
alpha      = 0.02                                          # waning rate post‐infection (modified in sweeps)
nu         = 1.0                                           # waning rate post‐vaccination (modified in sweeps)
force_frac = 0.0                                           # seasonal forcing

# vaccination, births, seeding
coverage      = 0.8                                        # vaccination fraction (transformed to rate such that coverage is reached in delta_a years)
delta_a       = 2.0/12.0                                   # age‐width for vaccination campaign (reaching coverage in delta_a years)
int_a         = 200.0                                      # interval between campaigns
vac_start     = 45.0                                       # time of first campaign // MEH not consistent
vac_end       = 150.0                                      # age of last campaign // MEH not consistent
N0            = 200000                                     # yearly cohort size
seed_fraction = 0.001                                      # initial fraction infected

# values for parameter sweep
beta_vals    = gamma .* R0_vals  # defined by the run script
alpha_vals   = [1.0, 0.5, 0.3333, 0.25, 0.2, 0.1, 0.05, 0.02, 0.01]
vac_age_vals = vcat(collect(0.0:1.0:100.0), 200.0) # 200 means no vaccination

#beta_vals    = [200.0]
#alpha_vals   = [0.2]                                     
#vac_age_vals = [60.0]

#collect parameter sets
param_sets = [(β, vac, α) for β in beta_vals for vac in vac_age_vals for α in alpha_vals] # try multithreading

# vaccination rate per year so that coverage is reached in delta_a years
rate_per_year = -log(1 - coverage)/delta_a

# initial population distribution; this determines the contact matrix and R0
# rates obtained from fitting loglinear model to AG2022 data
function survival_loglinear(age; a0=-5.054, a2=0.04695)
    return exp(-((10^a0)/(a2*log(10))) * (10^(a2*age) - 1))
end

# contact matrix
mat_df = CSV.read(joinpath(data_dir, "contact_matrix_data_120.txt"), DataFrame)
mat_df = filter(row -> row.part_age < 120 && row.cont_age < 120, mat_df)
n0 = maximum(mat_df.part_age) + 1
C0 = zeros(n0, n0)
for r in eachrow(mat_df)
  i0 = Int(r.part_age) + 1
  j0 = Int(r.cont_age) + 1
  C0[i0, j0] = r.c
end
C0 = (C0 + C0')/2                    # ensure full symmetry

# blow up 1-yr bins into da-yr bins
dens  = Int(round(1/da))            
n_age = n0 * dens                   
age_points = collect(amin:da:amin + da*(n_age-1))

C = zeros(n_age, n_age)
for i0 in 1:n0, j0 in 1:n0
  i_inds = ((i0-1)*dens + 1):(i0*dens)
  j_inds = ((j0-1)*dens + 1):(j0*dens)
  C[i_inds, j_inds] .= C0[i0, j0]
end
C = (C + C')/2   # bc small numerical asymmetries - obsolete?

# weigh by grid density
popdist = survival_loglinear.(age_points)   # length = n_age
K = C .* popdist' .* da                     # next generation contact matrix with initial demography (non-symmetric)

# normalise contact matrix to 1 
ρ = maximum(abs.(eigvals(K)))
contact_matrix = K ./ ρ

# infection fatality rate function
# ifr(age; a=1e-4,b=0.07,c=1e-5) = a*exp(b*age) - c 
ifr(age; a=1.86e-5, b=0.07, c=1.86e-5) = a*exp(b*age) - c

# new inline mortality function
@inline function mortality(t::Float64, age::Float64, a0::Float64, a1::Float64, a2::Float64, twarmup::Float64)
    logm = t < twarmup ?  # 1a*5 is 2025, a1*30 is 2050
           (a0 + a1*mortality_reference_year_offset + a2*age) :
           (a0 + a1*mortality_reference_year_offset + a1*(t - twarmup) + a2*age)
    return 10.0^logm
end

#= mortality rate function
function mu(t,age; a0=-5.054,a1=-0.006121,a2=0.04695,twarmup=150.0) # NB this matters for cohort vs period YLL
    #logm = t<twarmup ? a0 + a2*age : a0 + a1*(t-twarmup) + a2*age
    logm = t<twarmup ? a0 + a1*mortality_reference_year_offset + a2*age : a0 + a1*mortality_reference_year_offset + a1*(t-twarmup) + a2*age # start from 2025
    return 10.0^logm
end
=#
 
# prepare struct for ODE cache
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

# fill ODE cache
function ODECache(n_age::Int, age_points::Vector{Float64}, da::Float64, N0::Float64,
                  gamma::Float64, force_frac::Float64)
    ifr_vals = [ifr(age) for age in age_points]
    ODECache(
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
        -5.054,     # mu_a0
        -0.006121,  # mu_a1
        0.04695,    # mu_a2
        twarm       
    )
end

# define ODE function
function f_optim!(du,u,p,t)
    β, vac_rates, α, ν, contact_matrix, cache = p

    n_age_c = cache.n_age
    da_c = cache.da
    N0_c = cache.N0
    gamma_c = cache.gamma
    force_frac_c = cache.force_frac
    ifr_vals = cache.ifr_vals
    age_points_c = cache.age_points

    mu_a0 = cache.mu_a0
    mu_a1 = cache.mu_a1
    mu_a2 = cache.mu_a2
    mu_twarmup = cache.mu_twarmup

    S  = @view u[1:n_age_c]
    I  = @view u[n_age_c+1:2*n_age_c]
    R1 = @view u[2*n_age_c+1:3*n_age_c]
    V1 = @view u[3*n_age_c+1:4*n_age_c]
    R2 = @view u[4*n_age_c+1:5*n_age_c]
    V2 = @view u[5*n_age_c+1:6*n_age_c]

    N = cache.N
    I_div_N = cache.I_div_N
    FOI = cache.FOI

    @. N = S + I + R1 + R2 + V1 + V2 + 1e-6
    @. I_div_N = I / N
    mul!(FOI, contact_matrix, I_div_N)

    fsc = 1 + force_frac_c * cos(2π*t)
    
    # avoid costly ifelse inside the main loop
    @inbounds begin
        age  = age_points_c[1]
        λ    = β * fsc * FOI[1]
        vr   = vac_rates[1]
        #logm = t < mu_twarmup ? mu_a0 + mu_a1*mortality_reference_year_offset + mu_a2*age : mu_a0 + mu_a1*mortality_reference_year_offset + mu_a1*(t-mu_twarmup) + mu_a2*age
        #μ    = 10.0^logm
        μ = mortality(t, age, mu_a0, mu_a1, mu_a2, mu_twarmup) # still slightly slower than the above, not much
        dS_da  = (S[1] - N0_c)/da_c 
        dI_da  =  I[1]/da_c
        dR1_da =  R1[1]/da_c
        dV1_da =  V1[1]/da_c
        dR2_da =  R2[1]/da_c
        dV2_da =  V2[1]/da_c
        du[1]           = -λ*S[1] + 2*α*R2[1] + 2*ν*V2[1] - vr*S[1] - μ*S[1] - dS_da
        du[n_age_c+1]   =  λ*S[1] - gamma_c*I[1] - μ*I[1] - dI_da
        du[2*n_age_c+1] = (1 - ifr_vals[1]) * gamma_c*I[1] - 2*α*R1[1] - μ*R1[1] - dR1_da
        du[3*n_age_c+1] =  vr*S[1] - 2*ν*V1[1] - μ*V1[1] - dV1_da
        du[4*n_age_c+1] =  2*α*R1[1] - 2*α*R2[1] - μ*R2[1] - dR2_da
        du[5*n_age_c+1] =  2*ν*V1[1] - 2*ν*V2[1] - μ*V2[1] - dV2_da
    end

    @inbounds for i in 2:n_age_c # better berhaps: @inboudns @simd to parallelise the loop
        age  = age_points_c[i]
        λ    = β * fsc * FOI[i]
        vr   = vac_rates[i]
        #logm = t < mu_twarmup ? mu_a0 + mu_a1*mortality_reference_year_offset + mu_a2*age : mu_a0 + mu_a1*mortality_reference_year_offset + mu_a1*(t-mu_twarmup) + mu_a2*age
        #μ    = 10.0^logm
        μ = mortality(t, age, mu_a0, mu_a1, mu_a2, mu_twarmup)
        dS_da  = (S[i] - S[i-1])/da_c
        dI_da  = (I[i] - I[i-1])/da_c
        dR1_da = (R1[i] - R1[i-1])/da_c
        dV1_da = (V1[i] - V1[i-1])/da_c
        dR2_da = (R2[i] - R2[i-1])/da_c
        dV2_da = (V2[i] - V2[i-1])/da_c
        du[i]           = -λ*S[i] + 2*α*R2[i] + 2*ν*V2[i] - vr*S[i] - μ*S[i] - dS_da
        du[n_age_c+i]   =  λ*S[i] - gamma_c*I[i] - μ*I[i] - dI_da
        du[2*n_age_c+i] = (1 - ifr_vals[i]) * gamma_c*I[i] - 2*α*R1[i] - μ*R1[i] - dR1_da
        du[3*n_age_c+i] =  vr*S[i] - 2*ν*V1[i] - μ*V1[i] - dV1_da
        du[4*n_age_c+i] =  2*α*R1[i] - 2*α*R2[i] - μ*R2[i] - dR2_da
        du[5*n_age_c+i] =  2*ν*V1[i] - 2*ν*V2[i] - μ*V2[i] - dV2_da
    end
    return nothing
end

# cohort life expectancy based on mu(t,a) from current time to end of life
#=
function life_expectancy(t0, age0)
    # conditional survival from age 0 to x at time t0 + (x-age  0)
    cond_surv(x_age) = begin
        # integrate hazard over age from age0 to x_age
        #integral_hz, _ = quadgk(a -> mu(t0 + (a - age0), a), age0, x_age; rtol=1e-6)
        integral_hz, _ = quadgk(a -> mortality(t0 + (a - age0), a, -5.054, -0.006121, 0.04695, 150.0),
                        age0, x_age; rtol=1e-6)
        return exp(-integral_hz)
    end
    # integrate to amax
    integral_surv, _ = quadgk(cond_surv, age0, amax; rtol=1e-6)
    return integral_surv
end
=#

# new function, based on inline mortality
function life_expectancy(t0, age0, a0, a1, a2, twarmup)
    cond_surv(x_age) = begin
        integral_hz, _ = quadgk(a -> mortality(t0 + (a - age0), a, a0, a1, a2, twarmup),
                                age0, x_age; rtol=1e-6)
        return exp(-integral_hz)
    end
    integral_surv, _ = quadgk(cond_surv, age0, amax; rtol=1e-6)
    return integral_surv
end

# precompute life expectancy at tmax (on the grid)
#life_exp = [life_expectancy(tmax, a) for a in age_points]
life_exp = [life_expectancy(tmax, a, -5.054, -0.006121, 0.04695, 150.0) for a in age_points]

# seasonal forcing function 
forcing(t) = 1 + force_frac*cos(2π*t)
# initial conditions
u0 = zeros(6*n_age)
for i in 1:n_age
    age = age_points[i]
    pop_i = N0 * survival_loglinear(age)
    frac_i = seed_fraction * (age/10)^1 * ((120-age)/110)^11
    I0 = frac_i * pop_i
    u0[i]       = pop_i - I0
    u0[n_age+i] = I0
end

function get_vac_rate_vec(vac_ages::Vector{Float64})
    vr = zeros(length(age_points))
    @inbounds for i in eachindex(age_points)
        a_lo = age_points[i]; a_hi = a_lo + da
        s = 0.0
        for v in vac_ages
            win_lo = v; win_hi = v + delta_a
            overlap = max(0, min(a_hi, win_hi) - max(a_lo, win_lo))
            s += rate_per_year * (overlap / da)
        end
        vr[i] = s
    end
    return vr
end

# function for parameter sweep with threads - now from Maikel 
function run_one!(β::Float64, vac_start::Float64, α::Float64)
    ν = 2 * α # vaccination rate
    vac_rates = get_vac_rate_vec(collect(vac_start:int_a:vac_end))
    cache = ODECache(n_age, age_points, da, Float64(N0), gamma, force_frac)  # per-thread cache
    prob = ODEProblem(f_optim!, u0, (tmin, tmax), (β, vac_rates, α, ν, contact_matrix, cache))
    # try to avoid callbacks: 
    #prob = ODEProblem(f_optim!, u0, (tmin, tmax), (β, vac_rates, α, ν, contact_matrix, cache); isoutofdomain = (u,p,t)->any(u .< 0))
    sol  = solve(prob, Tsit5(); reltol=1e-4, abstol=1e-6,
                 callback=PositiveDomain(), saveat=da, dense=false)
    t_start_avg = tmax - 5
    idx_last5y = findall(t -> t ≥ t_start_avg, sol.t)
    mort_vals = Float64[]
    yll_vals  = Float64[] 
    for i in idx_last5y
        u = sol.u[i]
        I = u[n_age+1:2*n_age]
        M = gamma .* (I .* ifr.(age_points))
        push!(mort_vals, sum(M) * da)
        push!(yll_vals,  sum(M .* life_exp) * da)
    end
    (beta=β, vac_age=vac_start, alpha=α,
     total_mortality=mean(mort_vals), total_yll=mean(yll_vals))
end


function run_general_sweep()
    # solve with threads
    prog = Progress(length(param_sets); desc="running parameter sweeps", barlen=40, dt=5.0)
    tuples = ThreadsX.map(enumerate(param_sets)) do (i, ps)
        result = run_one!(ps[1], ps[2], ps[3])
        next!(prog)
        return result
    end

    # build dataframes
    df_mort = DataFrame(;
        beta            = getfield.(tuples, :beta),
        vac_age         = getfield.(tuples, :vac_age),
        alpha           = getfield.(tuples, :alpha),
        total_mortality = getfield.(tuples, :total_mortality),
    )

    df_yll = DataFrame(;
        beta     = getfield.(tuples, :beta),
        vac_age  = getfield.(tuples, :vac_age),
        alpha    = getfield.(tuples, :alpha),
        total_yll = getfield.(tuples, :total_yll),
    )

    # trick to select youngest vaccination age when outcomes are very low (essentially 0)
    age_penalty = 1e-2 # penalty per year of age (earlier 1e-4)
    df_mort.total_mortality_adj =
        df_mort.total_mortality .+ ifelse.(df_mort.vac_age .== 200.0, 0.0, age_penalty .* df_mort.vac_age)
    df_yll.total_yll_adj =
        df_yll.total_yll .+ ifelse.(df_yll.vac_age .== 200.0, 0.0, age_penalty .* df_yll.vac_age)

    # export results
    dat = "_" * Dates.format(Dates.today(), "yyyymmdd")
    CSV.write(joinpath(output_dir, "mortality_$(scenario_year)_nuhalf$(dat).csv"), df_mort)
    CSV.write(joinpath(output_dir, "yll_$(scenario_year)_nuhalf$(dat).csv"), df_yll)

    # goodbye
    println("finished parameter sweeps")
end
