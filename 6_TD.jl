### A Pluto.jl notebook ###
# v0.14.8

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

# ╔═╡ 2ac84074-c6ac-11eb-3a87-2b08d5ffa4fc
begin
    import Pkg
    Pkg.activate(mktempdir())
	
    Pkg.add([
        Pkg.PackageSpec(name="Plots"),
        Pkg.PackageSpec(name="PlutoUI"),
		Pkg.PackageSpec(name="StatsBase"),
		Pkg.PackageSpec(name="Distributions")
    ])
	
    using Statistics, Plots, PlutoUI, LinearAlgebra, StatsBase, Distributions
end

# ╔═╡ 7fbbaff3-c5f7-46d5-aa9d-2cb0f1fddc50
begin
	abstract type AbstractEnv end
	abstract type AbstractAgent end
	abstract type AbstractPolicy end
end

# ╔═╡ 893783fe-a5d7-4c5e-ba6d-764e9967a60f
md"# Chapter 6. Temporal-Difference Learning"

# ╔═╡ 788836c3-f7a9-47a1-804e-2ea0127e7b02
md"## Random Walk example"

# ╔═╡ db31c17f-37bd-4a5c-8ea0-aafa76919f46
md"""
![random_walk.jpg](6_rw.jpg)
"""

# ╔═╡ 1a1023ec-9e7e-4015-acb0-04fefc9852e3
function random_walk()
	state = 3# rand(1:5)
	history = [(0, state)]
	while (state > 0) & (state < 6)
		if rand() > 0.5
			state += 1
		else
			state -= 1
		end
		reward = (state == 6) ? 1 : 0
		push!(history, (reward, state))
	end
	
	return history
end

# ╔═╡ 7fa89833-aff3-4fad-8144-f099d54b0e4d
function mrp_td(α=0.1, init=0.5, γ=1.)
	V = zeros(5) .+ init
	vs = Array{Float64}(undef, (101, 5))
	vs[1, :] = V
	for i = 1: 100
		history = random_walk()
		(_, s) = history[1]
		for (r, s′) = history[2:end]
			Vs′ = (s′ in [0,6]) ? 0 : V[s′]
			V[s] += α * (r + γ*Vs′ - V[s])
			s = s′
		end
		vs[i+1, :] = V
	end
	vs, V
end

# ╔═╡ 1d382eee-1616-4c45-81aa-6b18a3223a48
function mrp_mc(α=0.1, init=0.5, γ=1.)
	V = zeros(5) .+ init
	vs = Array{Float64}(undef, (101, 5))
	vs[1, :] = V
	for i = 1: 100
		history = random_walk()
		(r, s) = history[end]
		G = r
		for t = length(history)-1:-1:1
			(r, s) = history[t]	
			# every visit MC
			V[s] += α * (G - V[s])
			G += r
		end
		vs[i+1, :] = V
	end
	vs, V
end

# ╔═╡ c8973041-c9c8-4e8e-bc96-2d9691fef86d
let 
	n_runs = 100
	V_true = collect(1:5) ./ 6
	rp = plot(ylabel="rmse", ylim=[0., 0.25], xlabel="episodes")
	for (α, c) = zip(0.05:0.05:0.15, ["blue", "blue2", "blue4"])
		rmse = zeros(101, 1)
		for i = 1:n_runs
			vs, V = mrp_td(α)
			rmse += (sqrt.(mean((vs .- reshape(V_true, 1, :)).^2, dims=2)) - rmse) / i
		end
				
		plot!(rp, rmse, label="TD, α=$α", colour=c)
	end
	
	for (α, c)= zip(0.01:0.01:0.04, ["red", "red1", "red2", "red4"])
		rmse = zeros(101, 1)
		for i = 1:n_runs
			vs, V = mrp_mc(α)
			rmse += (sqrt.(mean((vs .- reshape(V_true, 1, :)).^2, dims=2)) - rmse) / i
		end
		plot!(rp, rmse, label="MC, α=$α", colour=c)
	end
	rp
end

# ╔═╡ 31289e15-e217-496a-b2e8-0b99c0ecc019
let 
	n_runs = 100
	V_true = collect(1:5) ./ 6
	rp = plot(ylabel="rmse", ylim=[0., 0.25], xlabel="episodes")
	for α = 0.01:0.01:0.1
		rmse = zeros(101, 1)
		for i = 1:n_runs
			vs, V = mrp_td(α)
			rmse += (sqrt.(mean((vs .- reshape(V_true, 1, :)).^2, dims=2)) - rmse) / i
		end
				
		plot!(rp, rmse, label="TD, α=$α")
	end
	
	rp
end

# ╔═╡ 5b552b82-9a34-40c3-9853-7305488ae94d
let 
	n_runs = 100
	V_true = collect(1:5) ./ 6
	rp = plot(ylabel="rmse", ylim=[0., 0.25], xlabel="episodes")
	
	for α = 0.002:0.001:0.02
		rmse = zeros(101, 1)
		for i = 1:n_runs
			vs, V = mrp_mc(α)
			rmse += (sqrt.(mean((vs .- reshape(V_true, 1, :)).^2, dims=2)) - rmse) / i
		end
		plot!(rp, rmse, label="MC, α=$α")
	end
	rp
end

# ╔═╡ e364341b-ee70-4feb-948b-e1435ca7d503
md"""
V $(@bind init_val_td Slider(0. : 0.1 : 1. , show_value=true, default=0.5))
"""

# ╔═╡ 63cccf72-13df-46c4-8513-dcf1f8e25c83
let 
	n_runs = 100
	V_true = collect(1:5) ./ 6
	rp = plot(title="TD", ylabel="rmse", ylim=[0., 0.25], xlabel="episodes")
	for α = 0.025:0.025:0.15
		rmse = zeros(101, 1)
		for i = 1:n_runs
			vs, V = mrp_td(α, init_val_td)
			rmse += (sqrt.(mean((vs .- reshape(V_true, 1, :)).^2, dims=2)) - rmse) / i
		end
				
		plot!(rp, rmse, label="TD, α=$α")
	end
	
	rp
end

# ╔═╡ 4f2d1b03-8372-4d2a-9ae3-a3aeb4d2c6f3
md"""
V $(@bind init_val_mc Slider(0. : 0.1 : 1. , show_value=true, default=0.5))
"""

# ╔═╡ bd265c0f-1da2-46e8-8f5d-c656686288de
let 
	n_runs = 100
	V_true = collect(1:5) ./ 6
	rp = plot(title="MC", ylabel="rmse", ylim=[0., 0.25], xlabel="episodes")
	for α = 0.01:0.01:0.05
		rmse = zeros(101, 1)
		for i = 1:n_runs
			vs, V = mrp_td(α, init_val_mc)
			rmse += (sqrt.(mean((vs .- reshape(V_true, 1, :)).^2, dims=2)) - rmse) / i
		end
				
		plot!(rp, rmse, label="TD, α=$α")
	end
	
	rp
end

# ╔═╡ 5a5b2f79-f3ad-4cb2-998f-a21a16836127
md"## SARSA"

# ╔═╡ 934e4b2e-cbb1-4d20-9b52-a433a0456bee
@enum GridworldAction left=1 right=2 up=3 down=4

# ╔═╡ cb5acea5-e28f-40d8-bfe1-6a8410f2978e
begin
	struct WindyGridworld <: AbstractEnv
		m::Int64
		n::Int64
		special_states::Dict
	end
	function (world::WindyGridworld)(a::GridworldAction, state::Tuple)
		x, y = state
		m, n = world.m, world.n
		if haskey(world.special_states, state)
			return world.special_states[state]
		end
		if a == left
			if y-1 < 1
				return -1, state
			else
				return 0, (x, y-1)
			end
		elseif a == right
			if y+1 > n
				return -1, state
			else
				return 0, (x, y+1)
			end
		elseif a == up
			if x-1 < 1
				return -1, state
			else
				return 0, (x-1, y)
			end
		elseif a == down
			if x+1 > m
				return -1, state
			else
				return 0, (x+1, y)
			end
		end
	end
end

# ╔═╡ 6c9a108b-5ac0-43c8-84bf-2459c9674587
mutable struct GridworldAgent <: AbstractAgent
	v::Array
	q::Array
	state::Tuple
	actions
end

# ╔═╡ 4b5de8db-13fc-4f0f-a585-ce2b2de912f2
function state2idx(state::Tuple)::Tuple
	player_sum, dealer_card, usable_ace = state
	return player_sum-3, dealer_card, Int(usable_ace)+1
end

# ╔═╡ f4472d2e-bcaf-4736-b3e4-9c862acdcb89
function show_v(V::Array; legend=:none)
	left = heatmap(V[9:end,:,1], title="No usable ace", ylabel="Player sum", legend=:none, yticks=(1:10, 12:21))
	
	right= heatmap(V[9:end,:,2], title="Usable ace", legend=legend, yticks=:none)
	plot(left, right, xlabel="Dealer card")
end

# ╔═╡ 67053afd-1686-406b-acf1-790bdfa28374
function show_v3d(V::Array)
	left  = plot(1:10, 12:21, V[9:end, :, 1], st=:surface, title="No Ace", legend=:none)
	right = plot(1:10, 12:21, V[9:end, :, 2], st=:surface, title="With Ace", legend=:none)
	plot(left, right, layout=(1,2))
end

# ╔═╡ 6867a8a6-ea82-40c3-84c1-d37bdea8c79c
function show_actions(π::Array)
	no_ace_act = (π[8:end, :, 1, 1] .> .5)
	left = heatmap(no_ace_act, legend=:none, title="No Ace", yticks=(1:11, 11:21), ylabel="Player sum", xlabel="Dealer card")
	annotate!([(8,1,text("hit", :white, :left)), (8,11, text("stick", :black, :left))])
	ace_act = (π[8:end, :, 2, 1] .> .5)
	right = heatmap(ace_act, legend=:none, title="With Ace", yticks=(1:11, 11:21), xlabel="Dealer card")
	annotate!([(8,1,text("hit", :white, :left)), (8,11, text("stick", :black, :left))])
	plot(left, right)
end

# ╔═╡ 2390d0cf-6cad-4059-953f-e6b04479ca87
function off_policy_control(max_iter=100)
	# evaluete naive policy
	π = zeros(21-3,10,2,2)
	π[1:21-5, :, :, 2] .+= 1.
	π[21-4:21-3, :, :, 1] .+= 1.
	
	@assert all(sum(π, dims=4) .== 1) "Policy should be given as valid action probability distribution"
	env = Blackjack()
	γ = 1
	Q  = zeros(21-3, 10, 2, 2)
	C  = zeros(size(Q))
	history = []
	for i = 1:max_iter
		agent = create_agent()
		# behaviour policy (random)
		b = zeros(21-3, 10, 2, 2) .+ 0.5
		history = episode(agent, env, b)
		visited_states = map(x -> x.s, history)
		G = 0
		W = 1
		T = length(history)
		for i = T:-1:1
			s, a, r = history[i]
			G = γ*G + r
			
			idx = (state2idx(s)..., Int(a))
			# Q-values update
			C[idx...] += W
			Q[idx...] += (G - Q[idx...])*W/C[idx...]
			# policy update
			pidx = idx[1:3]
			a_star = argmax(Q[pidx..., :])
			π[pidx..., :] .= 0.
			π[pidx..., a_star] += 1.
			@assert all(sum(π, dims=4) .== 1)
			if π[idx...] == 0.
				break
			end
			W *= 1 / b[idx...]
		end
	end
	Q, π, C
end

# ╔═╡ 70bb38e8-ee11-4659-a3f5-ac6625676540
md"#### Example 5.5. Infinite variance"

# ╔═╡ b35c60b8-c952-4a7e-bff3-1d296ba25ba3
md"To be continued..."

# ╔═╡ b8af68b9-b2a9-4b0a-aa2b-b14e09243f59
TableOfContents()

# ╔═╡ Cell order:
# ╟─2ac84074-c6ac-11eb-3a87-2b08d5ffa4fc
# ╟─7fbbaff3-c5f7-46d5-aa9d-2cb0f1fddc50
# ╟─893783fe-a5d7-4c5e-ba6d-764e9967a60f
# ╟─788836c3-f7a9-47a1-804e-2ea0127e7b02
# ╟─db31c17f-37bd-4a5c-8ea0-aafa76919f46
# ╠═1a1023ec-9e7e-4015-acb0-04fefc9852e3
# ╟─7fa89833-aff3-4fad-8144-f099d54b0e4d
# ╟─1d382eee-1616-4c45-81aa-6b18a3223a48
# ╟─c8973041-c9c8-4e8e-bc96-2d9691fef86d
# ╠═31289e15-e217-496a-b2e8-0b99c0ecc019
# ╠═5b552b82-9a34-40c3-9853-7305488ae94d
# ╟─e364341b-ee70-4feb-948b-e1435ca7d503
# ╟─63cccf72-13df-46c4-8513-dcf1f8e25c83
# ╟─4f2d1b03-8372-4d2a-9ae3-a3aeb4d2c6f3
# ╟─bd265c0f-1da2-46e8-8f5d-c656686288de
# ╟─5a5b2f79-f3ad-4cb2-998f-a21a16836127
# ╠═934e4b2e-cbb1-4d20-9b52-a433a0456bee
# ╠═cb5acea5-e28f-40d8-bfe1-6a8410f2978e
# ╠═6c9a108b-5ac0-43c8-84bf-2459c9674587
# ╠═4b5de8db-13fc-4f0f-a585-ce2b2de912f2
# ╟─f4472d2e-bcaf-4736-b3e4-9c862acdcb89
# ╟─67053afd-1686-406b-acf1-790bdfa28374
# ╟─6867a8a6-ea82-40c3-84c1-d37bdea8c79c
# ╟─2390d0cf-6cad-4059-953f-e6b04479ca87
# ╟─70bb38e8-ee11-4659-a3f5-ac6625676540
# ╟─b35c60b8-c952-4a7e-bff3-1d296ba25ba3
# ╠═b8af68b9-b2a9-4b0a-aa2b-b14e09243f59
