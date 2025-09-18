# packages
using DifferentialEquations
using DiffEqCallbacks 
using LinearAlgebra
using CSV
using DataFrames
using Plots
using QuadGK
using Distributions
using NLsolve

# directories
base_dir    = @__DIR__
data_dir    = joinpath(base_dir, "data")
figures_dir = joinpath(base_dir, "figures")
output_dir  = joinpath(base_dir, "output")

# start a background process to prevent sleep
caffeinate_pid = run(`caffeinate -dimsu`, wait=false)
atexit() do
    kill(caffeinate_pid, Base.SIGTERM)
end
 
# time-age parameters
tmin, tmax  = 0.0, 150.0                                   # simulation duration (years)
amin, amax  = 0.0, 120.0                                   # age range (years)
da          = 0.25                                         # age discretisation (years)
age_points  = collect(amin:da:(amax-da))
n_age       = length(age_points)

# transmission parameters
beta       = 125.0                                         # transmission rate parameter (modified in sweeps)
gamma      = 50.0                                          # recovery rate (usually approx 1 week, 1/50 year)
alpha      = 0.02                                          # waning rate post‐infection (modified in sweeps)
nu         = 1.0                                           # waning rate post‐vaccination (modified in sweeps)
force_frac = 0.0                                           # seasonal forcing

# more realist waning rates
season_starts = [0.0; 0.5:1.0:(tmax-0.5)]                  # season starts in mid-summer
season_edges = [0.0; 0.5:1.0:(tmax-0.5); tmax]             # edges of seasons
q_low,  q_high = 2.0, 10.0                                 # range of duration of natrual immunity (years)
p_low,  p_high = 0.05, 0.95                                # quantiles of ranges; this can be used to solve for extremes
function gamma_resid!(F, x)
  k, θ = x
  d = Gamma(k, θ)
  F[1] = quantile(d, p_low)  - q_low
  F[2] = quantile(d, p_high) - q_high
end
sol = nlsolve(gamma_resid!, [3.0, 3.0])
const gamma_shape, gamma_scale = sol.zero
@show gamma_shape, gamma_scale

#const gamma_shape = 4.595
#const gamma_scale = 1.163

# gamma distribution for duration of natural immunity
dur_dist = Gamma(gamma_shape, gamma_scale)
n_seasons   = length(season_starts)
durations   = rand(dur_dist, n_seasons)
alpha_vals  = 1.0 ./ durations
#const alpha_vals = 1.0/rand(dur_dist_rate), length(season_starts)  # from Woolthuis et al (2017); mimcks influenza-like waning  
println("mean of waning rates = ", mean(alpha_vals))
function alpha_t(t)                                        # now waning rate can be made time-dependent
    idx = clamp(searchsortedlast(season_edges, t), 1, length(alpha_vals))
    return alpha_vals[idx]
end

# vaccination, births, seeding
coverage  = 0.9                                            # vaccination fraction (transformed to rate such that coverage is reached in delta_a years)
delta_a   = 0.25                                           # age‐width for vaccination campaign (reaching coverage in delta_a years)
int_a     = 200.0                                          # interval between campaigns
vac_start = 200.0                                          # first campaign at age vac_start
vac_end   = 80.0                                           # age of last campaign 
N0            = 200000                                     # yearly cohort size
seed_fraction = 0.001                                      # initial fraction infected

# values for parameter sweep
beta_vals    = [62.5, 75.0, 87.5, 100, 150, 200, 250, 500, 1000]
alpha_vals   = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]  # 1.0, 0.5, 0.2, 
#vac_age_vals = [0.5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200] 
vac_age_vals = vcat(collect(0.0:1.0:100.0), 200.0)

#beta_vals    = [500.0]
#alpha_vals   = [0.1]                                       
#vac_age_vals = [60.0]

# dataframes for sweeps
df_mort = DataFrame(beta=Float64[], vac_age=Float64[], alpha=Float64[], total_mortality=Float64[])
df_yll  = DataFrame(beta=Float64[], vac_age=Float64[], alpha=Float64[], total_yll=Float64[])
#df_mort = DataFrame(beta=Float64[], vac_age=Float64[], total_mortality=Float64[])
#df_yll  = DataFrame(beta=Float64[], vac_age=Float64[], total_yll=Float64[])

# age vaccination collection
vac_ages = collect(vac_start:int_a:vac_end)

# rate per year so that coverage is reached in delta_a years
rate_per_year = -log(1 - coverage)/delta_a

# initial population distribution; this determines the contact matrix and R0
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

# weight by by grid density
popdist = survival_loglinear.(age_points)   # length = n_age
K = C .* popdist' .* da                     # next generation contact matrix

# normalise contact matrix to 1 
ρ = maximum(abs.(eigvals(K)))
contact_matrix = K ./ ρ

# infection fatality rate function
ifr(age; a=1e-4,b=0.07,c=1e-5) = a*exp(b*age) - c

# mortality rate function
function mu(t,age; a0=-5.054,a1=-0.006121,a2=0.04695,twarmup=100.0)
    logm = t<twarmup ? a0 + a2*age : a0 + a1*(t-twarmup) + a2*age
    return 10.0^logm
end
 
# cohort life expectancy based on mu(t,a) from current time to end of life
function life_expectancy(t0, age0)
    # conditional survival from age 0 to x at time t0 + (x-age  0)
    cond_surv(x_age) = begin
        # integrate hazard over age from age0 to x_age
        integral_hz, _ = quadgk(a -> mu(t0 + (a - age0), a), age0, x_age; rtol=1e-6)
        return exp(-integral_hz)
    end
    # integrate to amax
    integral_surv, _ = quadgk(cond_surv, age0, amax; rtol=1e-6)
    return integral_surv
end

# precompute life expectancy at tmax (on the grid)
life_exp = [life_expectancy(tmax, a) for a in age_points]

# seasonal forcing function 
forcing(t) = 1 + force_frac*cos(2π*t)

# vaccination rate helper function
function get_vac_rate(i, vac_ages)
    vr = 0.0
    a_lo, a_hi = age_points[i], age_points[i] + da
    for s in vac_ages
        win_lo, win_hi = s, s + delta_a
        overlap = max(0, min(a_hi, win_hi) - max(a_lo, win_lo))
        vr += rate_per_year * (overlap / da)
    end
    return vr 
end

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

# parameter sweeps
# ODE system
#=
function f!(du,u,p,t)
    β, vac_ages, alpha = p
    S = @view u[1:n_age]
    I = @view u[n_age+1:2*n_age]
    R = @view u[2*n_age+1:3*n_age]
    V = @view u[3*n_age+1:4*n_age]  
    N = S .+ I .+ R .+ V .+ 1e-6
    FOI = contact_matrix * (I ./ N)
    for i in 1:n_age
        age = age_points[i]
        dS_da = i==1 ? (S[i] - N0)/da : (S[i] - S[i-1])/da
        dI_da = i==1 ?  I[i]/da   : (I[i] - I[i-1])/da
        dR_da = i==1 ?  R[i]/da   : (R[i] - R[i-1])/da
        dV_da = i==1 ?  V[i]/da   : (V[i] - V[i-1])/da
        λ = β * forcing(t) * FOI[i]
        vr = get_vac_rate(i, vac_ages)
        du[i]         = -λ*S[i] + alpha*R[i] + nu*V[i] - vr*S[i] - mu(t,age)*S[i] - dS_da
        du[n_age+i]   =  λ*S[i] - gamma*I[i] - mu(t,age)*I[i] - dI_da
        du[2*n_age+i] = (1-ifr(age))*gamma*I[i] - alpha*R[i] - mu(t,age)*R[i] - dR_da
        du[3*n_age+i] = vr*S[i] - nu*V[i] - mu(t,age)*V[i] - dV_da
    end
end
=#

# ODE system with Erlang waning 
function f!(du,u,p,t)
    β, vac_ages, alpha = p
    S = @view u[1:n_age]
    I = @view u[n_age+1:2*n_age]
    R1 = @view u[2*n_age+1:3*n_age]
    V1 = @view u[3*n_age+1:4*n_age]  
    R2 = @view u[4*n_age+1:5*n_age]
    V2 = @view u[5*n_age+1:6*n_age]  
    N = S .+ I .+ R1 .+ R2 .+ V1 .+ V2 .+ 1e-6
    FOI = contact_matrix * (I ./ N)
    for i in 1:n_age
        age = age_points[i]
        dS_da = i==1 ? (S[i] - N0)/da : (S[i] - S[i-1])/da
        dI_da = i==1 ?  I[i]/da   : (I[i] - I[i-1])/da
        dR1_da = i==1 ?  R1[i]/da   : (R1[i] - R1[i-1])/da
        dV1_da = i==1 ?  V1[i]/da   : (V1[i] - V1[i-1])/da
        dR2_da = i==1 ?  R2[i]/da   : (R2[i] - R2[i-1])/da
        dV2_da = i==1 ?  V2[i]/da   : (V2[i] - V2[i-1])/da
        λ = β * forcing(t) * FOI[i]
        vr = get_vac_rate(i, vac_ages)
        du[i]         = -λ*S[i] + 2*alpha*R2[i] + 2*nu*V2[i] - vr*S[i] - mu(t,age)*S[i] - dS_da
        du[n_age+i]   =  λ*S[i] - gamma*I[i] - mu(t,age)*I[i] - dI_da
        du[2*n_age+i] = (1-ifr(age))*gamma*I[i] - 2*alpha*R1[i] - mu(t,age)*R1[i] - dR1_da
        du[3*n_age+i] = vr*S[i] - 2*nu*V1[i] - mu(t,age)*V1[i] - dV1_da
        du[4*n_age+i] = 2*alpha*R1[i] - 2*alpha*R2[i] - mu(t,age)*R2[i] - dR2_da
        du[5*n_age+i] = 2*nu*V1[i] - 2*nu*V2[i] - mu(t,age)*V2[i] - dV2_da
    end
end

# parameter sweep
for local_beta in beta_vals
    for vac_start in vac_age_vals
        for local_alpha in alpha_vals
            nu = 2 * local_alpha # post-vaccination waning rate - now dependent on alpha
            # nu = 0.5
            global sol
            global nu
            local_vac_ages = collect(vac_start:int_a:vac_end)
            println("β = $local_beta \tvac_a = $vac_start \talpha = $local_alpha \tnu = $nu")
            prob = ODEProblem(f!, u0, (tmin, tmax), (local_beta, local_vac_ages, local_alpha))
            sol  = solve(prob, Tsit5(); reltol=1e-4, abstol=1e-6,
                         callback=PositiveDomain(), saveat=da)
            
            # calculate mortality at tmax/last 5 years
            #=
                         u_last   = sol.u[end]
            I_last   = u_last[n_age+1:2*n_age]
            M_last   = gamma .* (I_last .* ifr.(age_points)) 
            total_mort = sum(M_last) * da
            println("total mortality = ", total_mort)
            # calculate YLL at tmax
            total_yll  = sum(M_last .* life_exp) * da
            println("total YLL = ", total_yll)
            =#
            # get indices for last 5 years
            t_start_avg = tmax - 5
            idx_last5y = findall(t -> t ≥ t_start_avg, sol.t)

            # loop over time points in last 5 years
            mortality_series = Float64[]
            yll_series = Float64[]

            for i in idx_last5y
                u = sol.u[i]
                I = u[n_age+1:2*n_age]
                M = gamma .* (I .* ifr.(age_points))
                push!(mortality_series, sum(M) * da)
            push!(yll_series, sum(M .* life_exp) * da)
            end

            avg_mortality = mean(mortality_series)
            avg_yll = mean(yll_series)

            println("average mortality     = ", avg_mortality)
            println("average YLL           = ", avg_yll)
 
            # save results
            push!(df_mort, (local_beta, vac_start, local_alpha, avg_mortality))
            push!(df_yll,  (local_beta, vac_start, local_alpha, avg_yll))
            #push!(df_mort, (local_beta, vac_start, local_alpha, total_mort))
            #push!(df_yll,  (local_beta, vac_start, local_alpha, total_yll))
        end
    end
end

# ODE system - without parameter sweeps, variable waning
#=
function f!(du,u,p,t)
    β, vac_ages, alpha_fn = p
    alpha_current = alpha_fn(t)
    S = @view u[1:n_age]
    I = @view u[n_age+1:2*n_age]
    R = @view u[2*n_age+1:3*n_age]
    V = @view u[3*n_age+1:4*n_age]  
    N = S .+ I .+ R .+ V .+ 1e-6
    FOI = contact_matrix * (I ./ N)
    #alpha_s = alpha_fn(t) 
    for i in 1:n_age
        age = age_points[i]
        dS_da = i==1 ? (S[i] - N0)/da : (S[i] - S[i-1])/da
        dI_da = i==1 ?  I[i]/da   : (I[i] - I[i-1])/da
        dR_da = i==1 ?  R[i]/da   : (R[i] - R[i-1])/da
        dV_da = i==1 ?  V[i]/da   : (V[i] - V[i-1])/da
        λ = β * forcing(t) * FOI[i]
        vr = get_vac_rate(i, vac_ages)
        du[i]         = -λ*S[i] + alpha_current*R[i] + nu*V[i] - vr*S[i] - mu(t,age)*S[i] - dS_da
        du[n_age+i]   =  λ*S[i] - gamma*I[i] - mu(t,age)*I[i] - dI_da
        du[2*n_age+i] = (1-ifr(age))*gamma*I[i] - alpha_current*R[i] - mu(t,age)*R[i] - dR_da
        du[3*n_age+i] = vr*S[i] - nu*V[i] - mu(t,age)*V[i] - dV_da
    end
end

for local_beta in beta_vals
    for vac_start in vac_age_vals
            # nu = 2 * local_alpha # post-vaccination waning rate - now dependent on alpha
            # set up ODE problem
            global sol
            global nu
            local_vac_ages = collect(vac_start:int_a:vac_end)
            println("β = $local_beta \tvac_a = $vac_start \talpha = Beta(5,2) \tnu = $nu")
            prob = ODEProblem(f!, u0, (tmin, tmax), (local_beta, local_vac_ages, alpha_t))
            sol  = solve(prob, Tsit5(); reltol=1e-4, abstol=1e-6,
                         callback=PositiveDomain(), saveat=tmax)
            # calculate mortality at tmax
            u_last   = sol.u[end]
            I_last   = u_last[n_age+1:2*n_age]
            M_last   = gamma .* (I_last .* ifr.(age_points)) 
            total_mort = sum(M_last) * da
            println("total mortality = ", total_mort)
            # calculate YLL at tmax
            total_yll  = sum(M_last .* life_exp) * da
            println("total YLL = ", total_yll)
            # save results
            push!(df_mort, (local_beta, vac_start, total_mort))
            push!(df_yll,  (local_beta, vac_start, total_yll))
    end
end
=#

# export results
CSV.write(joinpath(output_dir, "mortality_by_scenario_demo.csv"), df_mort)
CSV.write(joinpath(output_dir, "yll_by_scenario_demo.csv"), df_yll)

# say something
println("finished parameter sweeps")

# extract solution for plotting of last simulation
t = sol.t
all_u = hcat(sol.u...)
S_mat = all_u[1:n_age,   :]'
I_mat = all_u[n_age+1:2*n_age, :]'
R1_mat = all_u[2*n_age+1:3*n_age, :]'
V1_mat = all_u[3*n_age+1:4*n_age, :]'
R2_mat = all_u[4*n_age+1:5*n_age, :]'
V2_mat = all_u[5*n_age+1:6*n_age, :]'
M_mat = gamma .* (I_mat .* (ifr.(age_points)'))

# time series plot
plot(t, sum(S_mat,dims=2)[:], label="S")
plot!(t, sum(I_mat,dims=2)[:], label="I")
plot!(t, sum(R1_mat,dims=2)[:]+sum(R2_mat,dims=2)[:], label="R")
plot!(t, sum(V1_mat,dims=2)[:]+sum(V2_mat,dims=2)[:], label="V")
xlabel!("time (years)")
ylabel!("number")
savefig(joinpath(figures_dir, "compartments_over_time.pdf"))

# total mortality plot
plot(t, sum(M_mat,dims=2)[:], label="mort")
xlabel!("time (years)")
ylabel!("number")
savefig(joinpath(figures_dir, "mortality_over_time.pdf"))

# time-age heatmaps
maxI = maximum(I_mat[end, :])
maxM = maximum(M_mat[end, :])
heatmap(t, age_points, S_mat', xlabel="time (years)", ylabel="age (years)")
savefig(joinpath(figures_dir, "heatmap_S.pdf"))
heatmap(t, age_points, I_mat', xlabel="time (years)", ylabel="age (years)", clim=(0, maxI))
savefig(joinpath(figures_dir, "heatmap_I.pdf"))
heatmap(t, age_points, (R1_mat+R2_mat)', xlabel="time (years)", ylabel="age (years)")
savefig(joinpath(figures_dir, "heatmap_R.pdf"))
heatmap(t, age_points, (V1_mat+V2_mat)', xlabel="time (years)", ylabel="age (years)")
savefig(joinpath(figures_dir, "heatmap_V.pdf"))
heatmap(t, age_points, M_mat', xlabel="time (years)", ylabel="age (years)", clim=(0, maxM))
savefig(joinpath(figures_dir, "heatmap_M.pdf"))
heatmap(t, age_points, (S_mat+I_mat+R1_mat+R2_mat+V1_mat+V2_mat)', xlabel="time (years)", ylabel="age (years)")
savefig(joinpath(figures_dir, "heatmap_N.pdf"))
