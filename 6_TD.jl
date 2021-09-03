### A Pluto.jl notebook ###
# v0.15.1

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
md"## TD Prediction: Random Walk example"

# ╔═╡ db31c17f-37bd-4a5c-8ea0-aafa76919f46
md"""
![random_walk.jpg](https://raw.githubusercontent.com/arampacha/rlbook.jl/main/images/6_rw.svg)
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

# ╔═╡ f8e93801-c271-47b8-a46b-f3a2eb7fe51e
md"## TD Control"

# ╔═╡ 5a5b2f79-f3ad-4cb2-998f-a21a16836127
md"""
### Sarsa

Sarsa is on-policy learning algorithm
"""

# ╔═╡ 9deae8d7-e96b-44bc-9bbd-d58d6fc7b0d1
md"""
##### Sarsa algorithm for estimating Q ≈ q*

>**Algorithm parameters:** step size $α ∈ (0, 1]$, small $ε > 0$\
>Initialize $Q(s, a)$, for all $s ∈ \mathcal{S}^+$, $a \in \mathcal{A}(s)$, arbitrarily except that $Q(terminal , ·) = 0$\
>Loop for each episode:\
>$\quad$Initialize $S$\
>$\quad$Choose $A$ from $S$ using policy derived from $Q$ (e.g., >ε-greedy)\
>$\quad$Loop for each step of episode:\
>$\quad \quad$Take action A, observe R, S 0\
>$\quad \quad$Choose $A_0$ from $S_0$ using policy derived from $Q$ (e.g., ε-greedy)\
>$\quad \quad$$Q(S,A) ← Q(S,A) + α [R + γQ(S', A') - Q(S, A)]$\
>$\quad \quad$$S ← S', A ← A'$\
>$\quad$until $S$ is terminal
"""

# ╔═╡ 934e4b2e-cbb1-4d20-9b52-a433a0456bee
@enum GridworldAction left=1 right=2 up=3 down=4

# ╔═╡ 5cf46ff6-0323-444f-a402-160f90da6c9a
action_map = Dict([
		(left,  ( 0,-1)),
		(right, ( 0, 1)),
		(up,    ( 1, 0)),
		(down,  (-1, 0))
		])

# ╔═╡ cb5acea5-e28f-40d8-bfe1-6a8410f2978e
begin
	struct WindyGridworld <: AbstractEnv
		height::Int64
		width::Int64
		wind::Vector
		terminal_states::Set
	end
	function (world::WindyGridworld)(state::Tuple, a::GridworldAction)
		y, x = state
		
		if ((y,x) in world.terminal_states)
			return 0, (y,x), true
		end
		
		m, n = world.height, world.width
		
		dy, dx = action_map[a]
		
		x, y = clamp(x+dx, 1, n), clamp(y+dy+world.wind[x], 1, m)
		
		return -1, (y,x), ((y,x) in world.terminal_states)
		
	end
end

# ╔═╡ 9b9b2dd3-3811-4ef4-9c42-85d0e93d4da2
let
	env = WindyGridworld(7, 10, zeros(Int64, 10), Set([(4,8)]))
	i = 0
	s = (4,1)
	finished=false
	history = []
	while !(finished) & (i< 100)
		i +=1
		r, s, finished = env(s, rand([right,left]))
		push!(history, s)
	end
	finished
	history
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
	heatmap(V, title="Value", legend=:none, yticks=:none, xticks=:none)
end

# ╔═╡ 67053afd-1686-406b-acf1-790bdfa28374
function show_v3d(V::Array)
	left  = plot(1:10, 12:21, V[9:end, :, 1], st=:surface, title="No Ace", legend=:none)
	right = plot(1:10, 12:21, V[9:end, :, 2], st=:surface, title="With Ace", legend=:none)
	plot(left, right, layout=(1,2))
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

# ╔═╡ 7e30e439-41f1-4041-bd3b-8e4fe9d63d7f
sa2idx(s, a) = (s..., Int(a))

# ╔═╡ fd53d80a-d48f-43cc-a12d-451d59f44de0
sa2idx((1,1), left)

# ╔═╡ c7518919-a1f1-4fd0-9349-3c90aa5b1227
begin
	function onehot(i, n)
		res = zeros(n)
		res[i] = 1.
		return res
	end
end

# ╔═╡ a8da32bd-a670-4a34-9256-a347a326be09
function sarsa(env::AbstractEnv, actions, n_iter::Int64=100, α::Float64=0.5, ε::Float64=0.1, γ=1.; start=nothing)
	max_steps = 1000
	m,n = env.height, env.width
	na = length(instances(actions))
	Q = zeros(m, n, na)
	π = similar(Q)
	# starting with random policy
	π .= 1. /na
	steps_per_episode = []
	episodes = []
	@assert all(sum(π, dims=3) .== 1.)
	for i = 1:n_iter
		s = (start == nothing) ? (rand(1:m), rand(1:n)) : start
		a = actions(sample(1:na, ProbabilityWeights(π[s..., :])))
		finished = false
		j = 1
		while !(finished) && (j < max_steps)
			push!(episodes, i)
			j += 1
			r, s′, finished = env(s, a)
			a′ = actions(sample(1:na, ProbabilityWeights(π[s′..., :])))
			Q′ = (finished ? 0 : Q[sa2idx(s′, a′)...])
			Q[sa2idx(s,a)...] += α * (r + γ*Q′ - Q[sa2idx(s,a)...])
			
			#ε-greedy policy based on Q
			π[s..., :] = ((1-ε)*onehot(argmax(Q[s..., :]), na) +
						  ε .* ones(na) ./ na)
			
			s, a = s′, a′
		end
		push!(steps_per_episode, j)
	end
	Q, π, episodes
end

# ╔═╡ ae892e0b-6c9d-40b1-85b7-bc80715fa340
function sarsa_step!(
		Q::Array, 
		π::Array, 
		env::AbstractEnv,
		actions::DataType;
		α::Float64=0.5,
		ε::Float64=0.1,
		γ::Float64=1.,
		start=nothing,
		max_steps=1000
	)
	
	m, n = env.height, env.width
	na = length(instances(actions))
	
	s = (start == nothing) ? (rand(1:m), rand(1:n)) : start
	a = actions(sample(1:na, ProbabilityWeights(π[s..., :])))
	
	finished = false
	j = 1
	total_reward = 0
	
	while !(finished) && (j < max_steps)
		j += 1
		r, s′, finished = env(s, a)
		total_reward += r
		a′ = actions(sample(1:na, ProbabilityWeights(π[s′..., :])))
		Q′ = (finished ? 0 : Q[sa2idx(s′, a′)...])
		Q[sa2idx(s,a)...] += α * (r + γ*Q′ - Q[sa2idx(s,a)...])

		#ε-greedy policy based on Q
		π[s..., :] = ((1-ε)*onehot(argmax(Q[s..., :]), na) +
					  ε .* ones(na) ./ na)
		
		s, a = s′, a′
	end
	
	h = (n=j, r = total_reward)
	return Q, π, h
end

# ╔═╡ ca57335d-1e44-476c-9c27-df8d252fea71
function run_policy(π, env, start)
	s = start
	path = [s, ]
	actions = []
	finished = false
	i = 0
	while (! finished) && (i<100)
		a = GridworldAction(argmax(π[s..., :]))
		r, s, finished = env(s, a)
		push!(path, s)
		push!(actions, a)
		i += 1
	end
	path, actions
end

# ╔═╡ ed4c917d-bd8f-40c1-b421-2f4daf9835f8
md"### Q-learning"

# ╔═╡ 958ae35d-b2a2-45e3-b478-2f9fffb52a6d
begin
	struct Cliff <: AbstractEnv
		height::Int64
		width::Int64
		special_states::Dict
		terminal_states::Set
	end
	function (world::Cliff)(state::Tuple, a::GridworldAction)
		y, x = state
		
		if ((y,x) in world.terminal_states)
			return 0, (y,x), true
		end
		
		m, n = world.height, world.width
		
		dy, dx = action_map[a]
		
		x = clamp(x+dx, 1, n)
		y = clamp(y+dy, 1, m)
		if (y,x) in keys(world.special_states)
			return world.special_states[(y,x)]
		end
		return -1, (y,x), ((y,x) in world.terminal_states)
		
	end
end

# ╔═╡ 9206164f-a71a-4878-a3e8-38a889f309f2
let
	special_states = Dict(
		[((1,x), (-100, (1,1), false)) for x = 2:11]
	)
	env = Cliff(4,12,special_states, Set([(1,12)]))
	s = (1,1)
	env(s, up), env(s, right)
end

# ╔═╡ b31c6091-0018-40ff-98c2-be16b59a2583
function q_learning(env::AbstractEnv, actions, n_iter::Int64=100, α::Float64=0.5, ε::Float64=0.1, γ=1.; start=nothing)
	max_steps = 1000
	m,n = env.height, env.width
	na = length(instances(actions))
	Q = zeros(m, n, na)
	π = similar(Q)
	# starting with random policy
	π .= 1. /na
	@assert all(sum(π, dims=3) .== 1.)
	
	steps_per_episode = []
	episodes = []
	for i = 1:n_iter
		s = (start == nothing) ? (rand(1:m), rand(1:n)) : start
		finished = false
		j = 1
		while !(finished) && (j < max_steps)
			push!(episodes, i)
			j += 1
			a = actions(sample(1:na, ProbabilityWeights(π[s..., :])))
			r, s′, finished = env(s, a)
			
			Q′ = (finished ? 0 : maximum(Q[s′..., :]))
			Q[sa2idx(s,a)...] += α * (r + γ*Q′ - Q[sa2idx(s,a)...])
			
			#ε-greedy policy based on Q
			π[s..., :] = ((1-ε)*onehot(argmax(Q[s..., :]), na) +
						  ε .* ones(na) ./ na)
			
			s = s′
		end
		push!(steps_per_episode, j)
	end
	Q, π, episodes
end

# ╔═╡ 59ea89fb-c80a-4472-b592-fb0e076c042e
function ql_step!(
		Q::Array, 
		π::Array, 
		env::AbstractEnv,
		actions::DataType;
		α::Float64=0.5,
		ε::Float64=0.1,
		γ::Float64=1.,
		start=nothing,
		max_steps=1000
	)
	
	m, n = env.height, env.width
	na = length(instances(actions))
	s = (start == nothing) ? (rand(1:m), rand(1:n)) : start
	a = actions(sample(1:na, ProbabilityWeights(π[s..., :])))
	
	finished = false
	j = 1
	total_reward = 0
	
	while !(finished) && (j < max_steps)
		j += 1
		a = actions(sample(1:na, ProbabilityWeights(π[s..., :])))
		r, s′, finished = env(s, a)
		total_reward += r
		
		Q′ = (finished ? 0 : maximum(Q[s′..., :]))
		Q[sa2idx(s,a)...] += α * (r + γ*Q′ - Q[sa2idx(s,a)...])
		
		#ε-greedy policy based on Q
		π[s..., :] = ((1-ε)*onehot(argmax(Q[s..., :]), na) +
					  ε .* ones(na) ./ na)
		
		s = s′
	end
	
	h = (n=j, r = total_reward)
	return Q, π, h
end

# ╔═╡ abf8614d-be09-41a8-93b5-3a118274d8b3
md"### Expected Sarsa"

# ╔═╡ 095457b7-0b9d-4e34-ac36-08b438f08b17
function esarsa(env::AbstractEnv, actions, n_iter::Int64=100, α::Float64=0.5, ε::Float64=0.1, γ=1.; start=nothing)
	max_steps = 1000
	m,n = env.height, env.width
	na = length(instances(actions))
	Q = zeros(m, n, na)
	π = similar(Q)
	# starting with random policy
	π .= 1. /na
	@assert all(sum(π, dims=3) .== 1.)
	
	steps_per_episode = []
	episodes = []
	for i = 1:n_iter
		s = (start == nothing) ? (rand(1:m), rand(1:n)) : start
		finished = false
		j = 1
		while !(finished) && (j < max_steps)
			push!(episodes, i)
			j += 1
			a = actions(sample(1:na, ProbabilityWeights(π[s..., :])))
			r, s′, finished = env(s, a)
			
			Q′ = (finished ? 0 : sum(π[s′..., :] .* (Q[s′..., :])))
			Q[sa2idx(s,a)...] += α * (r + γ*Q′ - Q[sa2idx(s,a)...])
			
			#ε-greedy policy based on Q
			π[s..., :] = ((1-ε)*onehot(argmax(Q[s..., :]), na) +
						  ε .* ones(na) ./ na)
			
			s = s′
		end
		push!(steps_per_episode, j)
	end
	Q, π, episodes
end

# ╔═╡ cd65cbbf-3b75-4cfc-a1f6-af1a566b3ef6
function esarsa_step!(
		Q::Array, 
		π::Array, 
		env::AbstractEnv,
		actions::DataType;
		α::Float64=0.5,
		ε::Float64=0.1,
		γ::Float64=1.,
		start=nothing,
		max_steps=1000
	)
	
	m, n = env.height, env.width
	na = length(instances(actions))
	s = (start == nothing) ? (rand(1:m), rand(1:n)) : start
	a = actions(sample(1:na, ProbabilityWeights(π[s..., :])))
	
	finished = false
	j = 1
	total_reward = 0
	
	while !(finished) && (j < max_steps)
		j += 1
		a = actions(sample(1:na, ProbabilityWeights(π[s..., :])))
		r, s′, finished = env(s, a)
		total_reward += r
		
		Q′ = (finished ? 0 : sum(π[s′..., :] .* (Q[s′..., :])))
		Q[sa2idx(s,a)...] += α * (r + γ*Q′ - Q[sa2idx(s,a)...])
		
		#ε-greedy policy based on Q
		π[s..., :] = ((1-ε)*onehot(argmax(Q[s..., :]), na) +
					  ε .* ones(na) ./ na)
		
		s = s′
	end
	
	h = (n=j, r = total_reward)
	return Q, π, h
end

# ╔═╡ e185420b-c6b6-45c4-8e5f-7c21455a400f
md"### Maximization bias"

# ╔═╡ e810049e-2b75-48a6-8bf1-e98432e84e68
begin
	struct Env67 <: AbstractEnv
	end
	function (world::Env67)(state::String, a::Int64)
		
		if state == "A"
			if a == 1
				return 0, "B", false
			else
				return 0, "T", true
			end
		end
		
		if state == "B"
			r = randn() - 0.1
			return r, "T", true
		end
	end
	
	
end

# ╔═╡ d4603152-17d9-414e-852d-9a7b09b01d46
let 
	env = Env67()
	env("B", 2)
end

# ╔═╡ 429845a8-81a8-40d7-b032-f1abbb8b8c36
function ql_step!(
		Q::Dict, 
		π::Dict, 
		env::AbstractEnv;
		# actions::Dict;
		α::Float64=0.5,
		ε::Float64=0.1,
		γ::Float64=1.,
		start=nothing,
		max_steps=1000
	)
	
	n = length(keys(Q))
	s = "A"
	a = sample(1:length(π[s]), ProbabilityWeights(π[s]))
	fa = a
	finished = false
	j = 1
	total_reward = 0
	
	while !(finished) && (j < max_steps)
		if j > 1
			a = sample(1:length(π[s]), ProbabilityWeights(π[s]))
		end
		r, s′, finished = env(s, a)
		total_reward += r
		
		Q′ = (finished ? 0 : maximum(Q[s′]))
		Q[s][a] += α * (r + γ*Q′ - Q[s][a])

		#ε-greedy policy based on Q
		na = length(Q[s])
		π[s] = ((1-ε)*onehot(argmax(Q[s]), na) +
				ε .* ones(na) ./ na)
		
		s = s′
		j += 1
	end
	
	h = (n=j, r = total_reward, a=fa)
	return Q, π, h
end

# ╔═╡ a70269b1-b3e1-4114-8a71-330927b38b2f
function sarsa_step!(
		Q::Dict, 
		π::Dict, 
		env::AbstractEnv;
		α::Float64=0.5,
		ε::Float64=0.1,
		γ::Float64=1.,
		start=nothing,
		max_steps=1000
	)
	
	n = length(keys(Q))
	s = "A"
	a = sample(1:length(π[s]), ProbabilityWeights(π[s]))
	fa = a
	finished = false
	j = 1
	total_reward = 0
	
	while !(finished) && (j < max_steps)
		r, s′, finished = env(s, a)
		total_reward += r
		a′ = (finished ? 
			0 : sample(1:length(π[s′]), ProbabilityWeights(π[s′])))
		Q′ = (finished ? 0 : Q[s′][a′])
		Q[s][a] += α * (r + γ*Q′ - Q[s][a])

		#ε-greedy policy based on Q
		na = length(Q[s])
		π[s] = ((1-ε)*onehot(argmax(Q[s]), na) +
				ε .* ones(na) ./ na)
		
		s, a = s′, a′
	end
	
	h = (n=j, r = total_reward, a=fa)
	return Q, π, h
end

# ╔═╡ aca031f4-23d5-443b-9f8f-16d641bda22a
let
	special_states = Dict(
		[((1,x), (-100, (1,1), false)) for x = 2:11]
	)
	env = Cliff(4,12,special_states, Set([(1,12)]))
	
	n_iter = 500
	
	m,n = env.height, env.width
	na = length(instances(GridworldAction))
	
	# SARSA
	Q = zeros(m, n, na)
	π = similar(Q)
	# starting with random policy
	π .= 1. /na
	@assert all(sum(π, dims=3) .== 1.)
	
	steps_per_episode = []
	rewards = []
	for i = 1:n_iter
		Q, π, h = sarsa_step!(Q, π, env, GridworldAction; start=(1,1))
		push!(steps_per_episode, h.n)
		push!(rewards, h.r)
	end
	
	mean_rewards_sarsa = [mean(rewards[1:i]) for i = 1:length(rewards)]
	p = plot(rewards, legend=:none, ylabel="episode", xlabel="timestep", ylim=[-100,0], colour=:blue, linestyle=:dash, alpha=0.4)
	plot!(p, mean_rewards_sarsa, legend=:none, colour=:blue, linewidth=2)
	
	
	# Q-learning
	Q = zeros(m, n, na)
	π = similar(Q)
	# starting with random policy
	π .= 1. /na
	@assert all(sum(π, dims=3) .== 1.)
	
	steps_per_episode = []
	rewards = []
	for i = 1:n_iter
		Q, π, h = ql_step!(Q, π, env, GridworldAction; start=(1,1))
		push!(steps_per_episode, h.n)
		push!(rewards, h.r)
	end
	
	mean_rewards_ql = [mean(rewards[1:i]) for i = 1:length(rewards)]
	plot!(p, rewards, legend=:none, colour=:red, linestyle=:dash, alpha=0.4)
	plot!(p, mean_rewards_ql, legend=:none, colour=:red, linewidth=2)
end

# ╔═╡ a37be2bf-f1de-47a5-bb48-1ae580ddf587
function esarsa_step!(
		Q::Dict, 
		π::Dict, 
		env::AbstractEnv;
		# actions::Dict;
		α::Float64=0.5,
		ε::Float64=0.1,
		γ::Float64=1.,
		start=nothing,
		max_steps=1000
	)
	
	n = length(keys(Q))
	s = "A"
	a = sample(1:length(π[s]), ProbabilityWeights(π[s]))
	fa = a
	finished = false
	j = 1
	total_reward = 0
	
	while !(finished) && (j < max_steps)
		if j > 1
			a = sample(1:length(π[s]), ProbabilityWeights(π[s]))
		end
		r, s′, finished = env(s, a)
		total_reward += r
		
		Q′ = (finished ? 0 : sum(π[s′] .* (Q[s′])))
		Q[s][a] += α * (r + γ*Q′ - Q[s][a])
		
		#ε-greedy policy based on Q
		na = length(Q[s])
		π[s] = ((1-ε)*onehot(argmax(Q[s]), na) +
				ε .* ones(na) ./ na)
		
		s = s′
		j += 1
	end
	
	h = (n=j, r = total_reward, a=fa)
	return Q, π, h
end

# ╔═╡ 2a489290-c4b3-4d09-98b3-c3ba445e41f2
function dql_step!(
		Q::Dict, 
		π::Dict, 
		env::AbstractEnv;
		# actions::Dict;
		α::Float64=0.5,
		ε::Float64=0.1,
		γ::Float64=1.,
		start=nothing,
		max_steps=1000
	)
	n = length(keys(Q))
	s = "A"
	a = sample(1:length(π[s]), ProbabilityWeights(π[s]))
	fa = a
	finished = false
	j = 1
	total_reward = 0
	
	while !(finished) && (j < max_steps)
		if j > 1
			a = sample(1:length(π[s]), ProbabilityWeights(π[s]))
		end
		r, s′, finished = env(s, a)
		total_reward += r
		
		(main, est) = (rand() < 0.5) ? (1,2) : (2,1) 
		Q′ = (finished ? 0 : maximum(Q[s′][main, :]))
		Q[s][main, a] += α * (r + γ*Q′ - Q[s][est, a])
		# if rand() < 0.5
		# 	Q′ = (finished ? 0 : maximum(Q[s′][1, :]))
		# 	Q[s][1, a] += α * (r + γ*Q′ - Q[s][2, a])
		# else
		# 	Q′ = (finished ? 0 : maximum(Q[s′][2, :]))
		# 	Q[s][2, a] += α * (r + γ*Q′ - Q[s][1, a])
		# end

		#ε-greedy policy based on Q
		na = size(Q[s])[end]
		π[s] = ((1-ε)*onehot(argmax(dropdims(sum(Q[s], dims=1), dims=1)), na) +
				ε .* ones(na) ./ na)
		# v1 = Q[s][1, 1]
		# if all(Q[s][1, :] .== v1)
		# 	id = rand(1:size(Q[s])[end])
		# else
		# 	id = argmax(Q[s][1, :])
		# end	
		# π[s] = ((1-ε)*onehot(id, na) +
		# 		ε .* ones(na) ./ na)
		s = s′
		j += 1
	end
	
	h = (n=j, r = total_reward, a=fa)
	return Q, π, h
end

# ╔═╡ 03b88572-cc28-492e-b644-dc505d0d84b1
function e67_experiment(f, double=true, n=300; debug=false)
	env = Env67()
	na = 10
	if double
		Q = Dict([
			("A", zeros(2,2)),
			("B", zeros(2,na)),
		])
	else
		Q = Dict([
			("A", zeros(2)),
			("B", zeros(na)),
		])
	end
	π = Dict([(k, ones(size(v)[end]) ./ size(v)[end]) for (k,v) in Q])
	steps = Array{Float64}(undef, n)
	rewards = Array{Float64}(undef, n)
	first_actions = Array{Float64}(undef, n)
	for i = 1:n
		Q, π, h = f(Q, π, env; α=0.1)
		
		steps[i] = h.n
		rewards[i] = h.r
		first_actions[i] = h.a
	end
	if debug
		return Q, π, rewards
	end
	return first_actions, rewards
end

# ╔═╡ 965d23da-47f5-4568-8bc8-6207eaa04c65
let
	ap = plot(ylabel="left action %", xlabel="episode", ylim=[0,1])
	rp = plot(ylabel="average reward", xlabel="episode")
	N = 1000
	T = 300
	actions = zeros(N,T)
	rewards = zeros(N,T)
	for i = 1:N
		actions[i, :], rewards[i, :] = e67_experiment(ql_step!, false, T)
	end
	la_ql = dropdims(mean(Int.(actions .== 1), dims=1), dims=1)
	ap = plot!(ap, la_ql, label="Q-learning")
	rp = plot!(rp, dropdims(mean(rewards, dims=1), dims=1), label="Q-learning")
	
# 	actions = zeros(N,T)
# 	d_rewards = zeros(N,T)
	
# 	for i = 1:N
# 		actions[i, :], d_rewards[i, :] = e67_experiment(dql_step!, T)
# 	end
# 	la_dql = dropdims(mean(Int.(actions .== 1), dims=1), dims=1)
# 	ap = plot!(ap, la_dql, label="Double Q-learning")
# 	rp = plot!(rp, dropdims(mean(d_rewards, dims=1), dims=1), label="Doble Q-learning")
	
	ap = hline!(ap, [0.05], linestyle=:dash, alpha=0.5, label=:none)
	plot(ap, rp, layout=(1,2))
end

# ╔═╡ 3d8c1ed5-aff8-4bc7-af58-bcd60c190238
let
	ap = plot(ylabel="left action %", xlabel="episode", ylim=[0,1])
	N = 1000
	T = 300
	actions = zeros(N,T)
	rewards = zeros(N,T)
	
	for i = 1:N
		actions[i, :], rewards[i, :] = e67_experiment(sarsa_step!, false, T)
	end
	la_dql = dropdims(mean(Int.(actions .== 1), dims=1), dims=1)
	ap = plot!(ap, la_dql, label="Sarsa")
	rp = plot(dropdims(mean(rewards, dims=1), dims=1))
	
	ap = hline!(ap, [0.05], linestyle=:dash, alpha=0.5, label=:none)
	plot(ap, rp)
	
end

# ╔═╡ eca1fd82-25a1-45db-b8b6-b5db21507792
let
	ap = plot(ylabel="left action %", xlabel="episode", ylim=[0,1])
	N = 1000
	T = 300
	actions = zeros(N,T)
	rewards = zeros(N,T)
	
	for i = 1:N
		actions[i, :], rewards[i, :] = e67_experiment(esarsa_step!, false, T)
	end
	la_dql = dropdims(mean(Int.(actions .== 1), dims=1), dims=1)
	ap = plot!(ap, la_dql, label="Expected Sarsa")
	rp = plot(dropdims(mean(rewards, dims=1), dims=1))
	
	ap = hline!(ap, [0.05], linestyle=:dash, alpha=0.5, label=:none)
	plot(ap, rp)
	
end

# ╔═╡ a3d111f1-999e-4dbc-b586-d12293cf0cb8


# ╔═╡ ef138518-d771-4f9a-98ad-7bd3ef673438
md"### Double Q-Learning"

# ╔═╡ 49918ad5-1b50-406b-ae28-ee05a6412ca2
let
	ap = plot(ylabel="left action %", xlabel="episode", ylim=[0,1])
	N = 1000
	T = 300
	actions = zeros(N,T)
	rewards = zeros(N,T)
	
	for i = 1:N
		actions[i, :], rewards[i, :] = e67_experiment(dql_step!, true, T)
	end
	la_dql = dropdims(mean(Int.(actions .== 1), dims=1), dims=1)
	ap = plot!(ap, la_dql, label="Double Q-learning")
	rp = plot(dropdims(mean(rewards, dims=1), dims=1))
	plot(ap, rp)
	
end

# ╔═╡ dd96b3c8-8f93-4089-9c05-3212aea07b95
let
	q, pi, rs = e67_experiment(dql_step!, true, 20; debug=true)
	q, rs
end

# ╔═╡ 03618872-d71a-4256-868b-6504fda1aae9
function test()
	na = 10
	Q = Dict([
		("A", zeros(2, 2)),
		("B", zeros(2, na)),
	])
	π = Dict([(k, ones(size(v)[end]) ./ size(v)[end]) for (k,v) in Q])
	env = Env67()
	α=0.1
	ε=0.1
	γ=1.
	# start=nothing
	max_steps=1000
	# create a doubled Q
	# Q = Dict([(k, zeros(2, length(v)) .+ 1.) for (k,v) in Q])
	n = length(keys(Q))
	
	hs = []
	for i = 1:1000
		s = "A"
		a = sample(1:length(π[s]), ProbabilityWeights(π[s]))
		fa = a
		finished = false
		j = 1
		total_reward = 0
		while !(finished) && (j < max_steps)
			if j > 1
				a = sample(1:length(π[s]), ProbabilityWeights(π[s]))
			end
			r, s′, finished = env(s, a)
			total_reward += r

			(main, est) = (rand() < 0.5) ? (1,2) : (2,1) 
			Q′ = (finished ? 0 : maximum(Q[s′][main, :]))
			try 
				Q[s][main, a] += α * (r + γ*Q′ - Q[s][est, a])
			catch
				return j, a, s
			end
			# if rand() < 0.5
			# 	Q′ = (finished ? 0 : maximum(Q[s′][1, :]))
			# 	Q[s][1, a] += α * (r + γ*Q′ - Q[s][2, a])
			# else
			# 	Q′ = (finished ? 0 : maximum(Q[s′][2, :]))
			# 	Q[s][2, a] += α * (r + γ*Q′ - Q[s][1, a])
			# end

			#ε-greedy policy based on Q
			na = size(Q[s])[end]
			π[s] = ((1-ε)*onehot(argmax(dropdims(sum(Q[s], dims=1), dims=1)), na) +
					ε .* ones(na) ./ na)
			# v1 = Q[s][1, 1]
			# if all(Q[s][1, :] .== v1)
			# 	id = rand(1:size(Q[s])[end])
			# else
			# 	id = argmax(Q[s][1, :])
			# end	
			# π[s] = ((1-ε)*onehot(id, na) +
			# 		ε .* ones(na) ./ na)
			s = s′
			j += 1
		end

		h = (n=j, r = total_reward, a=fa)
		push!(hs,h)
	end
	# Q = Dict([(k, dropdims(mean(v, dims=1), dims=1)) for (k,v) = Q])
	# Q = Dict([(k, v[1, :]) for (k,v) = Q])
	Q, π, hs
end

# ╔═╡ 2ccc1982-b869-4fdb-87fd-85165e8afd75
test()

# ╔═╡ b8c8478e-8723-4648-add2-450ac141b959
let
	a = zeros(5)
	size(a)[end]
end

# ╔═╡ b35c60b8-c952-4a7e-bff3-1d296ba25ba3
md"To be continued..."

# ╔═╡ b8af68b9-b2a9-4b0a-aa2b-b14e09243f59
TableOfContents()

# ╔═╡ 811fe1e4-b950-4317-b78c-24332ae3a855
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

# ╔═╡ fe7bfff1-8620-4a64-aba6-d1e934bb0f18
begin
	arrows = ["←", "→", "↑", "↓"]	
	
	function get_arrows(Q::Array)
		m,n,_ = size(Q)
		actions = [arrows[argmax(Q[i, j, :])] for i=1:m, j=1:n]
	end
	
end

# ╔═╡ 5f27e743-1eb4-43c5-97fb-319ce3304f98
function annotated_heatmap(Q::Array, show_policy::Bool=false; kwargs...)
	if show_policy
		labels = get_arrows(Q)
	else
		labels=nothing
	end
	m,n,_ = size(Q)
	V = [maximum(Q[i, j, :]) for i=1:m, j=1:n]
	anno = make_anno(V, labels=labels)
	
	p = plot(title="Value", yticks=:none, xticks=:none)
	heatmap!(p, V, yflip=false, annotations=anno, kwargs...)
end

# ╔═╡ 15414fc1-3a3a-4ce8-922e-b506fea7ba3c
begin
	m, n = 7,10
	# wind = zeros(Int64, n)
	wind = [0,0,0,1,1,1,2,2,1,0]
	# wind = [0,0,0,0,0,1,1,1,0,0]
	env = WindyGridworld(m, n, wind, Set([(4,8)]))
	Q, π, steps = sarsa(env, GridworldAction, 500, start=(4,1))
	
	p1 = annotated_heatmap(Q, true)
	p2 = plot(steps, legend=:none, ylabel="episode", xlabel="timestep")
	p1 = annotate!(p1, [(1,4, text("▤", :green)), (8,4, text("x", :red))])
	# learned policy
	path, actions = run_policy(π, env, (4,1))
	p1 = plot!(p1, [x[2] for x in path], [x[1] for x in path], linewidth=3, color=:blue, legend=:none, arrow=true)
	plot(p1,p2, layout=(2,1))
end

# ╔═╡ 5dfa9376-dd47-488b-8e9a-a941e7b49b6d
let
	special_states = Dict(
		[((1,x), (-100, (1,1), false)) for x = 2:11]
	)
	env = Cliff(4,12,special_states, Set([(1,12)]))
	Q, π, steps = sarsa(env, GridworldAction, 1000, start=(1,1))
	
	p1 = annotated_heatmap(Q, true)
	p2 = plot(steps, legend=:none, ylabel="episode", xlabel="timestep")
	
	# learned policy
	path, actions = run_policy(π, env, (1,1))
	p1 = plot!(p1, [x[2] for x in path], [x[1] for x in path], linewidth=3, color=:blue, legend=:none, arrow=true)
	
	plot(p1,p2, layout=(2,1))
end

# ╔═╡ 6d8ceca5-93fc-4987-afd2-4c61c79b7f80
let
	special_states = Dict(
		[((1,x), (-100, (1,1), false)) for x = 2:11]
	)
	env = Cliff(4,12,special_states, Set([(1,12)]))
	Q, π, steps = q_learning(env, GridworldAction, 1000, start=(1,1))
	
	p1 = annotated_heatmap(Q, true)
	p2 = plot(steps, legend=:none, ylabel="episode", xlabel="timestep")
	
	# learned policy
	path, actions = run_policy(π, env, (1,1))
	p1 = plot!(p1, [x[2] for x in path], [x[1] for x in path], linewidth=3, color=:blue, legend=:none, arrow=true)
	
	plot(p1,p2, layout=(2,1))
end

# ╔═╡ bbd07a54-9217-4bbf-9ed6-361f5fb976e6
let
	special_states = Dict(
		[((1,x), (-100, (1,1), false)) for x = 2:11]
	)
	env = Cliff(4,12,special_states, Set([(1,12)]))
	Q, π, steps = esarsa(env, GridworldAction, 100, 0.9, start=(1,1))
	
	p1 = annotated_heatmap(Q, true)
	p2 = plot(steps, legend=:none, ylabel="episode", xlabel="timestep")
	
	# learned policy
	path, actions = run_policy(π, env, (1,1))
	p1 = plot!(p1, [x[2] for x in path], [x[1] for x in path], linewidth=3, color=:blue, legend=:none, arrow=true)
	
	plot(p1,p2, layout=(2,1))
end

# ╔═╡ Cell order:
# ╟─2ac84074-c6ac-11eb-3a87-2b08d5ffa4fc
# ╠═7fbbaff3-c5f7-46d5-aa9d-2cb0f1fddc50
# ╟─893783fe-a5d7-4c5e-ba6d-764e9967a60f
# ╠═788836c3-f7a9-47a1-804e-2ea0127e7b02
# ╟─db31c17f-37bd-4a5c-8ea0-aafa76919f46
# ╠═1a1023ec-9e7e-4015-acb0-04fefc9852e3
# ╠═7fa89833-aff3-4fad-8144-f099d54b0e4d
# ╠═1d382eee-1616-4c45-81aa-6b18a3223a48
# ╟─c8973041-c9c8-4e8e-bc96-2d9691fef86d
# ╠═31289e15-e217-496a-b2e8-0b99c0ecc019
# ╠═5b552b82-9a34-40c3-9853-7305488ae94d
# ╟─e364341b-ee70-4feb-948b-e1435ca7d503
# ╟─63cccf72-13df-46c4-8513-dcf1f8e25c83
# ╟─4f2d1b03-8372-4d2a-9ae3-a3aeb4d2c6f3
# ╟─bd265c0f-1da2-46e8-8f5d-c656686288de
# ╟─f8e93801-c271-47b8-a46b-f3a2eb7fe51e
# ╟─5a5b2f79-f3ad-4cb2-998f-a21a16836127
# ╟─9deae8d7-e96b-44bc-9bbd-d58d6fc7b0d1
# ╟─934e4b2e-cbb1-4d20-9b52-a433a0456bee
# ╠═cb5acea5-e28f-40d8-bfe1-6a8410f2978e
# ╟─5cf46ff6-0323-444f-a402-160f90da6c9a
# ╟─9b9b2dd3-3811-4ef4-9c42-85d0e93d4da2
# ╟─6c9a108b-5ac0-43c8-84bf-2459c9674587
# ╟─4b5de8db-13fc-4f0f-a585-ce2b2de912f2
# ╟─f4472d2e-bcaf-4736-b3e4-9c862acdcb89
# ╟─67053afd-1686-406b-acf1-790bdfa28374
# ╟─2390d0cf-6cad-4059-953f-e6b04479ca87
# ╟─7e30e439-41f1-4041-bd3b-8e4fe9d63d7f
# ╟─fd53d80a-d48f-43cc-a12d-451d59f44de0
# ╠═a8da32bd-a670-4a34-9256-a347a326be09
# ╠═ae892e0b-6c9d-40b1-85b7-bc80715fa340
# ╟─c7518919-a1f1-4fd0-9349-3c90aa5b1227
# ╠═15414fc1-3a3a-4ce8-922e-b506fea7ba3c
# ╠═ca57335d-1e44-476c-9c27-df8d252fea71
# ╟─ed4c917d-bd8f-40c1-b421-2f4daf9835f8
# ╟─958ae35d-b2a2-45e3-b478-2f9fffb52a6d
# ╟─9206164f-a71a-4878-a3e8-38a889f309f2
# ╠═5dfa9376-dd47-488b-8e9a-a941e7b49b6d
# ╠═b31c6091-0018-40ff-98c2-be16b59a2583
# ╠═59ea89fb-c80a-4472-b592-fb0e076c042e
# ╠═aca031f4-23d5-443b-9f8f-16d641bda22a
# ╠═6d8ceca5-93fc-4987-afd2-4c61c79b7f80
# ╟─abf8614d-be09-41a8-93b5-3a118274d8b3
# ╠═095457b7-0b9d-4e34-ac36-08b438f08b17
# ╠═cd65cbbf-3b75-4cfc-a1f6-af1a566b3ef6
# ╠═bbd07a54-9217-4bbf-9ed6-361f5fb976e6
# ╟─e185420b-c6b6-45c4-8e5f-7c21455a400f
# ╠═e810049e-2b75-48a6-8bf1-e98432e84e68
# ╠═d4603152-17d9-414e-852d-9a7b09b01d46
# ╟─429845a8-81a8-40d7-b032-f1abbb8b8c36
# ╠═a70269b1-b3e1-4114-8a71-330927b38b2f
# ╠═a37be2bf-f1de-47a5-bb48-1ae580ddf587
# ╟─2a489290-c4b3-4d09-98b3-c3ba445e41f2
# ╠═03b88572-cc28-492e-b644-dc505d0d84b1
# ╠═965d23da-47f5-4568-8bc8-6207eaa04c65
# ╠═3d8c1ed5-aff8-4bc7-af58-bcd60c190238
# ╠═eca1fd82-25a1-45db-b8b6-b5db21507792
# ╠═a3d111f1-999e-4dbc-b586-d12293cf0cb8
# ╟─ef138518-d771-4f9a-98ad-7bd3ef673438
# ╠═49918ad5-1b50-406b-ae28-ee05a6412ca2
# ╠═dd96b3c8-8f93-4089-9c05-3212aea07b95
# ╠═03618872-d71a-4256-868b-6504fda1aae9
# ╠═2ccc1982-b869-4fdb-87fd-85165e8afd75
# ╠═b8c8478e-8723-4648-add2-450ac141b959
# ╟─b35c60b8-c952-4a7e-bff3-1d296ba25ba3
# ╠═b8af68b9-b2a9-4b0a-aa2b-b14e09243f59
# ╟─811fe1e4-b950-4317-b78c-24332ae3a855
# ╟─fe7bfff1-8620-4a64-aba6-d1e934bb0f18
# ╟─5f27e743-1eb4-43c5-97fb-319ce3304f98
