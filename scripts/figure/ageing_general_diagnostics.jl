# Optional diagnostic plots from the original general ageing script.
scenario_year = 2025
mortality_reference_year_offset = 5.0
R0_vals = [1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]
include(joinpath(@__DIR__, "..", "models", "ageing_general_model.jl"))
using Plots
figures_dir = joinpath(base_dir, "figures")
mkpath(figures_dir)

#########################################################################
### this part is just for plotting a single simulation - remove later ###
#########################################################################

# re-solve one for plotting
βp, vacp, αp = param_sets[1]
νp = 2 * αp
vac_rates_p = get_vac_rate_vec(collect(vacp:int_a:vac_end))
cache_p = ODECache(n_age, age_points, da, Float64(N0), gamma, force_frac)
prob_plot = ODEProblem(f_optim!, u0, (tmin, tmax), (βp, vac_rates_p, αp, νp, contact_matrix, cache_p))
sol = solve(prob_plot, Tsit5(); reltol=1e-4, abstol=1e-6,
            callback=PositiveDomain(), saveat=da, dense=false)

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
plot!(t, sum(S_mat,dims=2)[:]+sum(I_mat,dims=2)[:]+sum(R1_mat,dims=2)[:]+sum(R2_mat,dims=2)[:]+sum(V1_mat,dims=2)[:]+sum(V2_mat,dims=2)[:], label="N")
xlabel!("time (years)")
ylabel!("number")
savefig(joinpath(figures_dir, "compartments_over_time.pdf"))

# total mortality plot
mort_total = vec(sum(M_mat, dims=2))
sel = t .>= 20
plot(t[sel], mort_total[sel], label="mortality")
xlabel!("time (years)")
ylabel!("~number")
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

println("finished plotting")
