### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ c3b6bce8-ab7e-11eb-292e-3bb1464bfad2
begin
    import Pkg
    Pkg.activate(mktempdir())
	
    Pkg.add([
        Pkg.PackageSpec(name="Plots"),
        Pkg.PackageSpec(name="PlutoUI"),
		Pkg.PackageSpec(name="StatsBase"),
		Pkg.PackageSpec(name="SymPy")
    ])
	
    using Statistics, Plots, PlutoUI, LinearAlgebra, StatsBase, SymPy
end

# ╔═╡ cc0bd675-2f1d-4963-b5d6-6899f542c2d5
md"Define variables for SymPy"

# ╔═╡ 786fdaa3-db0e-408c-b1cb-d839735f7e38
s11,s12,s13,s14,s15,s21,s22,s23,s24,s25,s31,s32,s33,s34,s35,s41,s42,s43,s44,s45,s51,s52,s53,s54,s55=symbols("s11,s12,s13,s14,s15,s21,s22,s23,s24,s25,s31,s32,s33,s34,s35,s41,s42,s43,s44,s45,s51,s52,s53,s54,s55")

# ╔═╡ 78c1efb9-823d-4103-b3df-cba410804543
let
	vars = ""
	for i=1:5, j=1:5
		vars *= "s$i$j,"
	end
	vars
end

# ╔═╡ 1f867b35-90f6-40ac-ba80-750d3bd927e1
md"To avoid typing all equtions manually we can generate those. Hopefully this also allows to avoid typos"

# ╔═╡ d00cf809-6f0c-4057-b2ac-ad2c81eba6ae
begin
	eqs = ""
	for i=1:5, j=1:5
		eq = ".25 * .9*(s$(i-1)$j + s$(i+1)$j + s$i$(j-1) + s$i$(j+1)) - s$i$j "
		if (i in [1,5]) & (j in [1,5])
			eq *= " -.5 "
		elseif (i == 1) | (j == 1) | (i == 5) | (j == 5)
			eq *= " -.25 "
		end
		eq = "("*eq*")"
		eq *= ","
		eqs *= eq
	end
	eqs = replace(eqs, "0"=>"1")
	eqs = replace(eqs, "6"=>"5")
	eqs
end

# ╔═╡ 7da62397-7f49-4109-82a9-470c37fb31ae
begin
	equations = [
		(.25 * .9*(s11 + s21 + s11 + s12) - s11  -.5) ,
		(10 + .9*s52 - s12) ,
		(.25 * .9*(s13 + s23 + s12 + s14) - s13  -.25) ,
		(5 + .9*s34 - s14) ,
		(.25 * .9*(s15 + s25 + s14 + s15) - s15  -.5) ,
		(.25 * .9*(s11 + s31 + s21 + s22) - s21  -.25) ,
		(.25 * .9*(s12 + s32 + s21 + s23) - s22) ,
		(.25 * .9*(s13 + s33 + s22 + s24) - s23) ,
		(.25 * .9*(s14 + s34 + s23 + s25) - s24) ,
		(.25 * .9*(s15 + s35 + s24 + s25) - s25  -.25) ,
		(.25 * .9*(s21 + s41 + s31 + s32) - s31  -.25) ,
		(.25 * .9*(s22 + s42 + s31 + s33) - s32) ,
		(.25 * .9*(s23 + s43 + s32 + s34) - s33) ,
		(.25 * .9*(s24 + s44 + s33 + s35) - s34 ),
		(.25 * .9*(s25 + s45 + s34 + s35) - s35  -.25) ,
		(.25 * .9*(s31 + s51 + s41 + s42) - s41  -.25) ,
		(.25 * .9*(s32 + s52 + s41 + s43) - s42) ,
		(.25 * .9*(s33 + s53 + s42 + s44) - s43) ,
		(.25 * .9*(s34 + s54 + s43 + s45) - s44) ,
		(.25 * .9*(s35 + s55 + s44 + s45) - s45  -.25) ,
		(.25 * .9*(s41 + s51 + s51 + s52) - s51  -.5) ,
		(.25 * .9*(s42 + s52 + s51 + s53) - s52  -.25) ,
		(.25 * .9*(s43 + s53 + s52 + s54) - s53  -.25) ,
		(.25 * .9*(s44 + s54 + s53 + s55) - s54  -.25) ,
		(.25 * .9*(s45 + s55 + s54 + s55) - s55  -.5)
	]
	res = solve(equations, [s11,s12,s13,s14,s15,s21,s22,s23,s24,s25,s31,s32,s33,s34,s35,s41,s42,s43,s44,s45,s51,s52,s53,s54,s55])
end

# ╔═╡ 040aafc7-ef88-494a-9df4-cc5c5155a793
md"Visualising the result as heatmap makes it more readable:"

# ╔═╡ 326c9837-5770-4da0-afda-9f9950b56563
function make_anno(M::Matrix; labels=nothing, c=:green)
	m,n = size(M)
	xs = repeat(1:n, m)
	ys = [(i ÷ n)+1 for i=0:m*n-1]
	if labels == nothing
		vs = [round(M[i, j], digits=2) for (i, j) = zip(ys, xs)]
	else
		vs = [labels[i,j]  for (i, j) = zip(ys, xs)]
	end
	
	(xs, ys, vs, c)
end

# ╔═╡ 52eaab79-825d-487c-beb3-a647710037ee
begin
	vals = []
	for k = sort(collect(keys(res)), by=(s -> string(s)) )
		push!(vals, float(res[k]))
	end
	v = Matrix(transpose(reshape(vals, 5,5)))
	anno = make_anno(v)
	heatmap(v, annotations=anno, yflip=true)
end

# ╔═╡ 0f099f62-8867-41fb-9306-b06a8f9a1f2d
md"## Recycler 2000"

# ╔═╡ 430275d6-99cf-49fa-a10e-8e6ef64249c5
v_high, v_low = symbols("v_high, v_low")

# ╔═╡ 74625ccb-effa-4297-8eee-d9a39030656d
md"γ - discount factor"

# ╔═╡ 3494fc1f-2362-4c62-a30b-d7f48ec8e06d
@bind γ Slider(0.:0.1:1., default=0.9, show_value=true)

# ╔═╡ 338afec1-81fd-493a-8901-639f95240a0a
md"α ⧋ p( high | high, search )"

# ╔═╡ a6146066-bbf1-4b93-a870-89a13906e4e7
@bind α Slider(0.:0.1:1., default=0.8, show_value=true)

# ╔═╡ 9218b0c8-1ffa-472b-b14e-ca7afe29a06f
md"β ⧋ p( low | low, search)"

# ╔═╡ a407e55c-ce71-4356-a457-e3c1adaa666e
@bind β Slider(0.:0.1:1., default=0.5, show_value=true)

# ╔═╡ 812e3169-2cc6-40da-9783-3062a5ebf84d
md"`r_search`"

# ╔═╡ 4e756355-028e-4d76-a400-33e03fe5edc3
@bind r_search Slider(1:1:10, default=4, show_value=true)

# ╔═╡ 10a1d717-2d92-4c83-8270-18e8aebe6420
md"`r_wait` (< `r_search`)"

# ╔═╡ 87bb0511-34f5-4be3-927d-137ec5263285
@bind r_wait Slider(1:1:10, show_value=true)

# ╔═╡ a9373263-42cb-49c0-8ab3-9e6a3718fa65
md"Random policy"

# ╔═╡ 043e16a5-aee9-4921-99c7-093f66540dd4
begin
	rec_equations = [
		(1/2 * (r_search + γ*(α*v_high + (1-α)*v_low)) + 
		 1/2 * (r_wait   + γ*v_high)) - v_high,
		
		(1/3 * (β * (r_search + γ*v_low) + (1-β)*(-3 + γ*v_high)) + # remove γ*v_high to make it episodic 
		 1/3 * (r_wait + γ*v_low) + 
		 1/3 * γ*v_high) - v_low
	]
end

# ╔═╡ 6e024e65-da0c-450b-a9aa-431dd9a27aa8
rec_equations

# ╔═╡ 9bf77bb4-3025-412e-a0a5-f0074201d570
rec_res = solve(rec_equations, [v_high, v_low])

# ╔═╡ 8aa0196b-2ed6-4fec-9e08-9175f2fa30b9
md"Let's consider how the state value function depend on policy for _high_ state actions π(search|high) - probability of atking search action in _high_ state"

# ╔═╡ 65a39833-2103-4a90-b8af-cb4f8ea7da8c
let
	
	psl = 0.3
	pwl = 0.3
	prl = 1 - psl - pwl
	
	v_hs = []
	v_ls = []
	for psh = 0.1:0.1:1.
		pwh = 1-psh
		rec_equations = [
			(psh * (r_search + γ*(α*v_high + (1-α)*v_low)) + 
			 pwh * (r_wait   + γ*v_high)) - v_high,

			(psl * (β * (r_search + γ*v_low) + (1-β)*(-3)) +
			 pwl * (r_wait + γ*v_low) + 
			 prl * γ*v_high) - v_low
		]
		rec_res = solve(rec_equations, [v_high, v_low])
		push!(v_hs, float(rec_res[v_high]))
		push!(v_ls, float(rec_res[v_low]))
	end
	p = plot(legend=:bottomright, xlabel="π(search|high)", ylabel="state values", xlim=[0,1])
	plot!(p, 0.1:0.1:1., v_hs, label = "v_high")
	plot!(p, 0.1:0.1:1., v_ls, label = "v_low")
end

# ╔═╡ 395a73e0-1244-444a-9af1-8ec6ef9aaa23
let
	
	psh = 1.
	pwh = 0.
	
	v_hs = []
	v_ls = []
	for psl = 0.:0.1:1.
		pwl = (1 - psl)*1/2#1/3
		prl = (1 - psl)*1/2#2/3
		rec_equations = [
			(psh * (r_search + γ*(α*v_high + (1-α)*v_low)) + 
			 pwh * (r_wait   + γ*v_high)) - v_high,

			(psl * (β * (r_search + γ*v_low) + (1-β)*(-3)) +
			 pwl * (r_wait + γ*v_low) + 
			 prl * γ*v_high) - v_low
		]
		rec_res = solve(rec_equations, [v_high, v_low])
		push!(v_hs, float(rec_res[v_high]))
		push!(v_ls, float(rec_res[v_low]))
	end
	p = plot(legend=:bottomright, xlabel="π(search|low)", ylabel="state values", xlim=[0,1])
	plot!(p, 0.:0.1:1., v_hs, label = "v_high")
	plot!(p, 0.:0.1:1., v_ls, label = "v_low")
end

# ╔═╡ 18db857c-a0b0-40e8-974d-fdbf36749ec5
md"Ex 3.19: $q_{\pi}(s, a) = \mathbb(E)[R_{t+1} + v_{\pi}(S_{t+1})] = \sum_r \sum_{s'} p(s', r|s, a) (r + \gamma.v_{\pi}(s'))$ "

# ╔═╡ Cell order:
# ╟─c3b6bce8-ab7e-11eb-292e-3bb1464bfad2
# ╟─cc0bd675-2f1d-4963-b5d6-6899f542c2d5
# ╠═786fdaa3-db0e-408c-b1cb-d839735f7e38
# ╟─78c1efb9-823d-4103-b3df-cba410804543
# ╟─1f867b35-90f6-40ac-ba80-750d3bd927e1
# ╠═d00cf809-6f0c-4057-b2ac-ad2c81eba6ae
# ╠═7da62397-7f49-4109-82a9-470c37fb31ae
# ╟─040aafc7-ef88-494a-9df4-cc5c5155a793
# ╠═52eaab79-825d-487c-beb3-a647710037ee
# ╟─326c9837-5770-4da0-afda-9f9950b56563
# ╟─0f099f62-8867-41fb-9306-b06a8f9a1f2d
# ╟─430275d6-99cf-49fa-a10e-8e6ef64249c5
# ╟─74625ccb-effa-4297-8eee-d9a39030656d
# ╟─3494fc1f-2362-4c62-a30b-d7f48ec8e06d
# ╟─338afec1-81fd-493a-8901-639f95240a0a
# ╟─a6146066-bbf1-4b93-a870-89a13906e4e7
# ╟─9218b0c8-1ffa-472b-b14e-ca7afe29a06f
# ╟─a407e55c-ce71-4356-a457-e3c1adaa666e
# ╟─812e3169-2cc6-40da-9783-3062a5ebf84d
# ╟─4e756355-028e-4d76-a400-33e03fe5edc3
# ╟─10a1d717-2d92-4c83-8270-18e8aebe6420
# ╠═87bb0511-34f5-4be3-927d-137ec5263285
# ╟─a9373263-42cb-49c0-8ab3-9e6a3718fa65
# ╠═043e16a5-aee9-4921-99c7-093f66540dd4
# ╠═6e024e65-da0c-450b-a9aa-431dd9a27aa8
# ╠═9bf77bb4-3025-412e-a0a5-f0074201d570
# ╟─8aa0196b-2ed6-4fec-9e08-9175f2fa30b9
# ╠═65a39833-2103-4a90-b8af-cb4f8ea7da8c
# ╟─395a73e0-1244-444a-9af1-8ec6ef9aaa23
# ╟─18db857c-a0b0-40e8-974d-fdbf36749ec5
