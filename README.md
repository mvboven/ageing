**Age-structured SIRV transmission model with population ageing**

Note: this is work in progress and comes with no warranty!

This repository contains code for simulating an age-structured SIRV model with semi-realistic demographic projections for the Netherlands, Erlang-distributed immunity durations, age-targeted vaccination, seasonal forcing, and mortality/years-of-life-lost (YLL) calculations. The code is written in Julia and makes use of the _DifferentialEquations.jl_ ecosystem. Related code under development based on the _MethodOfLines.jl_ package is available on request. Please send enquiries to Michiel van Boven at r.m.vanboven-2@umcutrecht.nl or Mirjam Kretzschmar at m.e.e.kretzschmar@umcutrecht.nl.


**What does the program do**
* Sets up an age grid (0–120 years, default step 0.25y) and simulates dynamics over long horizons.
* Constructs a contact matrix using data in _contact_matrix_data_120.txt_. The contact matrix is scaled so that its largest eigenvalue equals 1. Hence R0=beta/gamma. code has been checked for threshold behaviour at R0:=1 (using fixed demography). code also yield correct final size when all contact matrix entries are identical.
* It implements Erlang-distributed immunity, both after vaccination and infection.
* It allows for season-to-season variation in waning rates by sampling from a (Gamma) distribution.
* It includes seasonal forcing of transmission, age-window vaccination campaigns, and age-dependent background mortality.
* It computes mortality and years of life lost from infections, using an infection fatality rate function and life expectancy integrals.
* Currently, it runs parameter sweeps over the transmission rate, the waning rates, and the vaccination start age. Notice that code is not yet streamlined; care should be taken when adapting code to your own needs.
* It saves the mortality and years-of-life lost (YLL) and produces plots and time–age heatmaps. Notice that YLL is based on cohort life expectancy, NOT period life expectancy.


**Method**
* The code discretises the age-structured partial differential equations (PDEs) using a (first-order, I believe) upwind scheme in age and standard ODE solvers in time.
* Vaccination is implemented as short campaigns at specific ages with a rate chosen to achieve a target coverage in a given age window (think: in three/six months, mimicking duration of the campaign). Alternatives can easily be added.
* Mortality and YLL are obtained by combining age-specific infection deaths with conditional life expectancy.


**Input and output**
* Input data is the contact data in _contact_matrix_data_120.txt_. Notice that some other pathogen and population specific paramters are defined in the code (to be streamlined).
* Output is written to _mortality_by_scenario_demo.csv_ and _output/yll_by_scenario_demo.csv_.
* various plots and heatmaps are also ouputted. 


**Dependencies**
* DifferentialEquations, DiffEqCallbacks, LinearAlgebra, CSV, DataFrames, Plots, QuadGK, Distributions, NLsolve.


**Status**
* This code is work in progress and come with no warranty whatsoever. 
