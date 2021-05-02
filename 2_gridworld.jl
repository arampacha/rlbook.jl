### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ f2cc03f2-aab2-11eb-2cc1-b77ff8eaf5b3
begin
    import Pkg
    Pkg.activate(mktempdir())
	
    Pkg.add([
        Pkg.PackageSpec(name="Plots"),
        Pkg.PackageSpec(name="PlutoUI"),
		Pkg.PackageSpec(name="StatsBase"),
    ])
	
    using Statistics, Plots, PlutoUI, LinearAlgebra, StatsBase
end

# ╔═╡ 71cf2d28-ba98-422e-8839-f50452be5cdb
abstract type Envirenment end

# ╔═╡ 37db68da-32ca-49f7-acb8-c6afa5f1fd5f
@enum Action left=1 right=2 up=3 down=4

# ╔═╡ bc3c5dc0-976f-4158-950b-61155904a9ba
begin
	struct Gridworld <: Envirenment
		m::Int64
		n::Int64
		special_states::Dict
	end
	function (world::Gridworld)(a::Action, state::Tuple)
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

# ╔═╡ 299d4065-e294-44ea-8fd6-8bd0e641bc2c
begin
	special_states = Dict([
			((1,2), (10, (5,2))),
			((1,4), ( 5, (3,4))),
			])
	world = Gridworld(5,5, special_states)
end

# ╔═╡ 9af358f5-0466-4e73-a401-9e45ef197dbd
world(up, (1,1))

# ╔═╡ 9cb61687-6aa9-4632-93b5-d4208cef1db1
let
	ss1 = (1,2)
	ss2 = (1,4)
	@assert world(up,   (1,1)) == (-1, (1,1))
	@assert world(down, (1,1)) == ( 0, (2,1))
	@assert world(down, ss1) == world(up, ss1) == world(left, ss1) == world(down, ss1) == world.special_states[ss1]
	@assert world(down, ss2) == world(up, ss2) == world(left, ss2) == world(down, ss2) == world.special_states[ss2] 
end

# ╔═╡ ac443d5e-f8f9-423f-8b8d-8516d6bab2bd
function move(a::Action, state::Tuple, world::Gridworld)
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

# ╔═╡ df197549-817e-4618-afe4-e96e1a46d68e
move(up, (1,1), world)

# ╔═╡ 25b8e0fe-34bd-4251-8156-54b6daadbfa5
move(down, (1,1), world)

# ╔═╡ a5ecc150-f644-491f-98dd-60f670c74888
move(down, (1,2), world)

# ╔═╡ 8d1ff438-8d08-4134-9265-c41cfe609197
let
	ss1 = (1,2)
	ss2 = (1,4)
	@assert move(up,   (1,1), world) == (-1, (1,1))
	@assert move(down, (1,1), world) == ( 0, (2,1))
	@assert move(down, ss1, world) == move(up, ss1, world) == move(left, ss1, world) == move(down, ss1, world) == world.special_states[ss1]
	@assert move(down, ss2, world) == move(up, ss2, world) == move(left, ss2, world) == move(down, ss2, world) == world.special_states[ss2] 
end

# ╔═╡ e8323dd8-3ae0-4509-a8ae-f5db3a38e4a4
mutable struct Agent
	q::Array{Float64,3}
	v::Array{Float64,2}
	state::Tuple
	
	Agent(m::Integer,n::Integer, state::Tuple, actions, init::Float64=1.) = new([init for i = 1:m, j=1:n, k=1:length(instances(actions))],[init for i=1:m, j=1:n], state)
	
	
	Agent(env::Gridworld, actions, init=1.) = Agent(env.m, env.n, (rand(1:env.m), rand(1:env.n)), actions, init)
end

# ╔═╡ f9d31ad6-3446-481d-a7f5-faafeb1d87c9
begin
	abstract type AbstractPolicy end
	struct RandomPolicy <: AbstractPolicy end
end

# ╔═╡ 5cb50a3c-01ce-487b-b451-e7ca56933aae
function get_action(agent::Agent, π::RandomPolicy)
	return rand([left, right, up, down])
end

# ╔═╡ 601b3875-45de-470e-a3f1-a9c1c56ecf4d
up

# ╔═╡ 687dbaf2-cd43-4dc5-861d-6d0962dfb921
md"### Exact solution"

# ╔═╡ 075a6137-9a52-4c06-a863-dce8fae98359


# ╔═╡ a48e130f-2876-44cc-bfdd-746251836b55
struct GreedyPolicy <: AbstractPolicy
	ε::Float64
	GreedyPolicy(ε) = ((ε < 0.) | (ε > 1.)) ? error("ε outside [0,1]") : new(ε)
	GreedyPolicy() = GreedyPolicy(0.1)
end

# ╔═╡ 84de4363-ee9a-4dc2-9650-5f21d427f454
GreedyPolicy(0.1)

# ╔═╡ 5132b64c-3ef1-45d6-9c43-f64a2efa4fb0
function get_action(agent::Agent, π::GreedyPolicy)
	if rand() < π.ε
		return rand(instances(Action))
	else
		return Action(argmax(agent.q[agent.state..., :]))
	end
end

# ╔═╡ fa747c20-d3d2-43e0-a9e3-2d65981e6cf9
function softmax(x::Vector)
	return exp.(x .- maximum(x)) ./ sum(exp.(x .- maximum(x)))
end

# ╔═╡ a0047a8d-cb1f-44e4-bfe5-a294b2264001
actions = [a for a = instances(Action)]

# ╔═╡ b4c4ee9c-6dc9-47e2-b87e-525c198751d1
sample(actions, ProbabilityWeights(softmax(zeros(4)), 1))

# ╔═╡ f7459692-fdd5-4e5b-bc3c-f637d86d0fe9
struct SamplingPolicy <: AbstractPolicy
	τ::Float64
	SamplingPolicy(τ) = (τ ≤ 0.) ? error("τ must be > 0") : new(τ)
	SamplingPolicy() = SamplingPolicy(1.)
end

# ╔═╡ f545a6f8-e8ea-4510-a32e-a6f1306deaec
SamplingPolicy()

# ╔═╡ 266aab70-e346-4a64-ae3a-993cd94ab534
function get_action(agent::Agent, π::SamplingPolicy)
	probs = softmax(agent.q[agent.state..., :]/π.τ)
	Action(sample(1:5, ProbabilityWeights(probs, 1)))
end

# ╔═╡ fe96ce8c-26f8-432e-abee-9917a3c417a2
function step!(agent::Agent, env::Envirenment, π::AbstractPolicy=RandomPolicy())
	γ = 0.9
	a = get_action(agent, π)
	
	r, new_state = env(a, agent.state)
	
	agent.q[agent.state..., Int(a)] = r + γ * agent.v[new_state...]
	agent.v[agent.state...] = r + γ * agent.v[new_state...]
	agent.state = new_state
end

# ╔═╡ 8034730a-0532-425c-8009-573edda39bec
get_action(Agent(world, Action), GreedyPolicy())

# ╔═╡ 58cb61b9-25ab-4235-a74c-cf701a4c5411
get_action(Agent(world, Action), SamplingPolicy())

# ╔═╡ bca44f3f-cf45-481f-b383-d3abdd36b4fc
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

# ╔═╡ 9c5bf918-e071-4b2d-af03-1eede3b31439
arrows = ["←", "→", "↑", "↓"]

# ╔═╡ 017f1aaa-ad51-4698-b6db-5a2cc26dcfe7
function get_arrows(agent::Agent)
	m,n = size(agent.v)
	actions = [arrows[argmax(agent.q[i, j, :])] for i=1:m, j=1:n]
end

# ╔═╡ bae52d01-015e-4029-8c9b-d9b032abc43c
function annotated_heatmap!(p, agent, show_policy=false; kwargs...)
	if show_policy
		labels = get_arrows(agent)
	else
		labels=nothing
	end
	anno = make_anno(agent.v, labels=labels)
	heatmap(agent.v, yflip=true, annotations=anno, kwargs...)
end

# ╔═╡ 85ab20ca-cea0-4487-a817-5447313f28d4
let
	agent = Agent(world, Action)
	history = [agent.state]
	for t in 1:100_000
		step!(agent, world)
		push!(history, agent.state)
	end
	
	p = plot()
	annotated_heatmap!(p, agent)
end

# ╔═╡ 313ac0dc-6c44-4cc1-a6af-2551dc645890
let
	agent = Agent(world, Action)
	history = [agent.state]
	for t in 1:100_000
		step!(agent, world, GreedyPolicy())
		push!(history, agent.state)
	end
	
	p = plot()
	annotated_heatmap!(p, agent)
end

# ╔═╡ 9380a8a6-421b-42d8-824f-78c07bc9d20c
let
	agent = Agent(world, Action)
	history = [agent.state]
	for t in 1:100_000
		step!(agent, world, SamplingPolicy(1.5))
		push!(history, agent.state)
	end
	
	p = plot()
	annotated_heatmap!(p, agent)
end

# ╔═╡ c64e734f-6c6b-4dd2-835a-28c8b505f4bc
begin
	agent = Agent(world, Action)
	history = [agent.state]
	for t in 1:100_000
		step!(agent, world, SamplingPolicy())
		if t % 1000 == 0
			agent.state = (rand(1:world.m), rand(1:world.n))
		end
		push!(history, agent.state)
	end
	
	p = plot()
	annotated_heatmap!(p, agent)
end

# ╔═╡ 30f462d0-5b8a-4fe7-a28d-492327652590
annotated_heatmap!(plot(), agent, true)

# ╔═╡ 11fefeff-1026-4537-9d50-912627d95b50
get_arrows(agent)

# ╔═╡ 06f2a1e3-38d6-4f59-bbd9-17dea644d088
md"## Maze"

# ╔═╡ b93b1dd5-0d09-4c2b-a26d-c414a065073f
begin
	struct Maze <: Envirenment
		m::Int64
		n::Int64
		special_states::Dict
		impossible_actions::Dict
	end
	function (world::Maze)(a::Action, state::Tuple)
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

# ╔═╡ 7b2fe3d6-6004-4cc2-b2d5-8a73f1ee1038


# ╔═╡ 9e3394fe-3a55-4a5a-9f09-b9083541c6b7


# ╔═╡ Cell order:
# ╠═f2cc03f2-aab2-11eb-2cc1-b77ff8eaf5b3
# ╠═71cf2d28-ba98-422e-8839-f50452be5cdb
# ╠═37db68da-32ca-49f7-acb8-c6afa5f1fd5f
# ╠═bc3c5dc0-976f-4158-950b-61155904a9ba
# ╠═299d4065-e294-44ea-8fd6-8bd0e641bc2c
# ╠═9af358f5-0466-4e73-a401-9e45ef197dbd
# ╠═9cb61687-6aa9-4632-93b5-d4208cef1db1
# ╟─ac443d5e-f8f9-423f-8b8d-8516d6bab2bd
# ╟─df197549-817e-4618-afe4-e96e1a46d68e
# ╟─25b8e0fe-34bd-4251-8156-54b6daadbfa5
# ╟─a5ecc150-f644-491f-98dd-60f670c74888
# ╟─8d1ff438-8d08-4134-9265-c41cfe609197
# ╠═e8323dd8-3ae0-4509-a8ae-f5db3a38e4a4
# ╠═f9d31ad6-3446-481d-a7f5-faafeb1d87c9
# ╠═5cb50a3c-01ce-487b-b451-e7ca56933aae
# ╠═601b3875-45de-470e-a3f1-a9c1c56ecf4d
# ╠═fe96ce8c-26f8-432e-abee-9917a3c417a2
# ╠═85ab20ca-cea0-4487-a817-5447313f28d4
# ╟─687dbaf2-cd43-4dc5-861d-6d0962dfb921
# ╠═075a6137-9a52-4c06-a863-dce8fae98359
# ╠═a48e130f-2876-44cc-bfdd-746251836b55
# ╟─84de4363-ee9a-4dc2-9650-5f21d427f454
# ╠═5132b64c-3ef1-45d6-9c43-f64a2efa4fb0
# ╠═fa747c20-d3d2-43e0-a9e3-2d65981e6cf9
# ╠═b4c4ee9c-6dc9-47e2-b87e-525c198751d1
# ╠═a0047a8d-cb1f-44e4-bfe5-a294b2264001
# ╠═8034730a-0532-425c-8009-573edda39bec
# ╠═313ac0dc-6c44-4cc1-a6af-2551dc645890
# ╠═f7459692-fdd5-4e5b-bc3c-f637d86d0fe9
# ╠═f545a6f8-e8ea-4510-a32e-a6f1306deaec
# ╠═266aab70-e346-4a64-ae3a-993cd94ab534
# ╠═58cb61b9-25ab-4235-a74c-cf701a4c5411
# ╠═9380a8a6-421b-42d8-824f-78c07bc9d20c
# ╠═bae52d01-015e-4029-8c9b-d9b032abc43c
# ╠═bca44f3f-cf45-481f-b383-d3abdd36b4fc
# ╠═c64e734f-6c6b-4dd2-835a-28c8b505f4bc
# ╠═30f462d0-5b8a-4fe7-a28d-492327652590
# ╟─9c5bf918-e071-4b2d-af03-1eede3b31439
# ╠═017f1aaa-ad51-4698-b6db-5a2cc26dcfe7
# ╠═11fefeff-1026-4537-9d50-912627d95b50
# ╟─06f2a1e3-38d6-4f59-bbd9-17dea644d088
# ╠═b93b1dd5-0d09-4c2b-a26d-c414a065073f
# ╠═7b2fe3d6-6004-4cc2-b2d5-8a73f1ee1038
# ╠═9e3394fe-3a55-4a5a-9f09-b9083541c6b7
