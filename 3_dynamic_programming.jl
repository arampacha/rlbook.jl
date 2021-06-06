### A Pluto.jl notebook ###
# v0.14.7

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

# ╔═╡ f2cc03f2-aab2-11eb-2cc1-b77ff8eaf5b3
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

# ╔═╡ 37db68da-32ca-49f7-acb8-c6afa5f1fd5f
@enum Action left=1 right=2 up=3 down=4

# ╔═╡ f9d31ad6-3446-481d-a7f5-faafeb1d87c9
begin
	abstract type AbstractPolicy end
end

# ╔═╡ fa747c20-d3d2-43e0-a9e3-2d65981e6cf9
function softmax(x::Vector)
	mx = maximum(x)
	return exp.(x .- mx) ./ sum(exp.(x .- mx))
end

# ╔═╡ a0047a8d-cb1f-44e4-bfe5-a294b2264001
actions = [a for a = instances(Action)]

# ╔═╡ b0e44de6-3b3c-455c-8261-7b80bbfb4c40
actions

# ╔═╡ b4c4ee9c-6dc9-47e2-b87e-525c198751d1
sample(actions, ProbabilityWeights(softmax(zeros(4)), 1))

# ╔═╡ f7459692-fdd5-4e5b-bc3c-f637d86d0fe9
struct SamplingPolicy <: AbstractPolicy
	τ::Float64
	SamplingPolicy(τ) = (τ ≤ 0.) ? error("τ must be > 0") : new(τ)
	SamplingPolicy() = SamplingPolicy(1.)
end

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

# ╔═╡ b85630be-0de6-4a3b-8e65-dd09c6b8cd09
md"## Policy evaluation"

# ╔═╡ f7153663-096e-4560-9a5e-9d82a26cb653
function max_idx(itr)
	max_val = first(itr)
	idxs = [1]
	for i = 2:length(itr)
		if itr[i] > max_val
			idxs = [i]
			max_val = itr[i]
		elseif itr[i] == max_val
			push!(idxs, i)
		end
	end
	idxs
end	

# ╔═╡ ce1e9402-d7d8-42e9-bd17-23989a40976e
md"
```
function get_action(agent::Agent, π::GreedyPolicy)
	if rand() < π.ε
		return rand(instances(Action))
	else
		return Action(argmax(agent.q[agent.state..., :]))
	end
end
```

"

# ╔═╡ 14c52ff7-ed4d-4984-adfa-01fd33104a88
md"
```
function iter_policy_evaluation(π; thres=1e-4)
	V = 
	max_iter = 100_000
	for _ in 1:max_iter
		Δ = 0
		for s in states
			v = V[s]
			V(s) = # Bellman
			Δ = max(Δ, abs(v-V[s]))
		end
	end
	V
end
```
"

# ╔═╡ e25f1ecc-279e-4987-aa3b-1edf87c99baa
md"k ∈ [1,100]"

# ╔═╡ 1feed7d1-eb2c-4c39-9a51-3f6929441ccf
@bind k1 Slider(1:1:100, show_value=true)

# ╔═╡ dd0457e6-8d02-4994-96dc-bf977ace13f9
md"## Policy iteration"

# ╔═╡ e7335f7e-c5df-457d-b7b5-8cd20ba00328
function get_arrows(policy::Array)
	m,n,_ = size(policy)
	actions = [arrows[argmax(policy[i, j, :])] for i=1:m, j=1:n]
end

# ╔═╡ 69653c0a-c2ff-4fe9-bb6b-fc84188032e9
@bind k2 Slider(2:1:20, show_value=true)

# ╔═╡ bc784903-90dc-42a0-9617-65b5813a9ec9
md"### Car Rental"

# ╔═╡ ae927a87-1169-479e-bdf6-77b78b4f9892
car_actions = -5:5

# ╔═╡ 4e7a19e2-71a8-4dc2-84f3-7a1f67bdf812
md"Poisson distribution"

# ╔═╡ 293096c6-b4b8-4fd6-b546-2b5d9f3bc180
@bind lambda1 Slider(0:1:10, default=3, show_value=true)

# ╔═╡ 9ee1d663-7e3d-4c14-ae15-227895bd3952
let
	λ = lambda1
	K = 0:20
	p = plot(K, [λ^k/factorial(k) * exp(-λ) for k=K], xlim=(0,20))
	title!(p, "PDF of poisson distribution, λ = $λ")
end

# ╔═╡ e8bb6fae-00e9-4fc5-ae4d-13dc6145ba81
@bind run2 CheckBox()

# ╔═╡ 5df84a7f-a025-47af-aee1-861b86fba7f7
# let
# 	env = CarRental(20)
# 	agent = RentalAgent(20, car_actions)
	
# 	policy = zeros(20,20,length(agent.actions))
# 	policy[:, :, 6] .= 1.
# 	policy_evaluation(policy, agent, env)
# end

# ╔═╡ 3a58fc77-8e56-4fa3-9751-bd46f240d249
@bind k3 Slider(2:10, show_value=true)

# ╔═╡ 41298949-f106-4618-a358-c567445bc123
# let
# 	env = CarRental(20)
# 	agent = RentalAgent(20, car_actions)
	
# 	policy = zeros(20,20,length(agent.actions))
# 	policy[:, :, 6] .= 1.
# 	stable = false
# 	for i = 1:k3
# 		v, n = policy_evaluation(policy, agent, env)

# 		policy, stable = policy_update(policy, agent, env)
# 		if stable
# 			break
# 		end
# 	end
# 	if stable
# 		title = "The policy is stable"
# 	else
# 		title = "The policy is unstable"
# 	end
# 	heatmap(agent.v; colorbar=:none, title=title)
# end

# ╔═╡ 53cc743b-bf03-43a9-9e2d-8ea326eec091
md"## Value Iteration"

# ╔═╡ d8ba47f1-c0a3-4445-8972-632a53895b80
@bind k4 Slider(1:41, show_value=true)

# ╔═╡ a4f22ea9-3de0-4c5b-b266-e0d04f60f0a1
@bind ph Slider(0.05:0.05:0.95, show_value=true, default=0.4)

# ╔═╡ 536c684c-feb0-4a18-88b2-cefef417479f
@bind s_cap Slider(10:10:90, show_value=true, default=50)

# ╔═╡ ba1ad05b-d77f-4eb8-915f-066e6bd8c5fa
function gamble(ph::Float64, policy::Function, n::Integer=10_000)
	res = []
	for i = 1:n
		cap = s_cap
		while (cap > 0) & (cap < 100)
			bet = policy(cap)
			if rand() < ph
				cap += bet
			else
				cap -= bet
			end
		end
		push!(res, Int(cap == 100))
	end
	mean(res)
end

# ╔═╡ 6cad08a4-df22-4db4-8446-73657fc369b5
let
	res_minbet_40 = gamble(0.4, x -> 1)
	res_maxbet_40 = gamble(0.4, x -> min(x, 100-x))
	
	md"Probability of wining:
	pₕ=0.4  - minbet $res_minbet_40 maxbet $res_maxbet_40
	"
end

# ╔═╡ 42c95284-fa39-4811-85f3-e2eca96e0cfe
let
	
	res_minbet_55 = gamble(0.55, x -> 1)
	res_maxbet_55 = gamble(0.55, x -> min(x, 100-x))
	
	md"Probability of wining:
	pₕ=0.55 - minbet $res_minbet_55 maxbet $res_maxbet_55
	"
end

# ╔═╡ 8d9c6117-e514-4a12-9c4b-8de70f445b53
md"Family of optimal strategies for pₕ < 0.5 includes maximal bets at 25, 50, 75. For p\_h > 0.5 always betting minimal is optimal."

# ╔═╡ a419de6a-fa5c-439e-ac64-ed576e0da6be
md"## utils"

# ╔═╡ 3b0047ae-4713-470f-a145-6a97cf93f090
abstract type AbstractAgent end

# ╔═╡ f11b080e-b689-4b5b-8525-e36bd2f08658
mutable struct RentalAgent <: AbstractAgent
	max_cars::Integer
	actions::Vector
	v::Matrix
	q:: Array
	state::Tuple
	
	RentalAgent(max_n, actions) = new(max_n, [a for a = actions], zeros(max_n, max_n), zeros(max_n, max_n, length(actions)), (max_n, max_n))
end

# ╔═╡ 2d60f367-c5cf-4bac-9bd6-50c6646598ba
let
	agent = RentalAgent(20, car_actions)
	agent.v
	s = (2, 3)
	[agent.actions[i] for i=1:length(agent.actions) if (agent.actions[i] >= -s[2]) & (agent.actions[i] <= s[1])]
end

# ╔═╡ 01a7c860-da17-44c5-a96b-9ba3e81a066f
function valid_action(i::Integer, agent::AbstractAgent, state::Tuple)
	true
end

# ╔═╡ 43036305-86fb-4311-9671-98419a547d9b
abstract type AbstractEnv end

# ╔═╡ bc3c5dc0-976f-4158-950b-61155904a9ba
begin
	struct Gridworld <: AbstractEnv
		m::Int64
		n::Int64
		special_states::Dict
		default_reward::Number
		out_of_bound_reward::Number
		Gridworld(m,n,special_states,default_reward,out_of_bound_reward) = new(m,n,special_states,default_reward,out_of_bound_reward)
		Gridworld(m,n,special_states) = new(m,n,special_states, 0, -1)
		Gridworld(m,n,special_states,default_reward) = new(m,n,special_states,default_reward, -1)
	end
	function (env::Gridworld)(a::Action, state::Tuple)
		x, y = state
		m, n = env.m, env.n
		if haskey(env.special_states, state)
			return env.special_states[state]
		end
		if a == left
			if y-1 < 1
				return env.out_of_bound_reward, state
			else
				return env.default_reward, (x, y-1)
			end
		elseif a == right
			if y+1 > n
				return env.out_of_bound_reward, state
			else
				return env.default_reward, (x, y+1)
			end
		elseif a == up
			if x-1 < 1
				return env.out_of_bound_reward, state
			else
				return env.default_reward, (x-1, y)
			end
		elseif a == down
			if x+1 > m
				return env.out_of_bound_reward, state
			else
				return env.default_reward, (x+1, y)
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

# ╔═╡ 59d7151c-09a5-4531-ad14-433bdbb4269b
begin 
	special_states1 = Dict([
			((1,1), (0, (1,1))),
			((4,4), (0, (4,4)))
			])
	gridworld1 = Gridworld(4,4, special_states1, -1)
	gridworld1(right, (1,1))
end

# ╔═╡ 1f27ca59-901d-4751-a3ae-20d3a4563fe5
struct CarRental <:AbstractEnv
	max_cars::Integer
	CarRental(max_cars) = new(max_cars)
	function (env::CarRental)(action, state)
		action = clamp(action, -state[2], state[1])
		new_state = (
			clamp(state[1] - action,0,env.max_cars),
			clamp(state[2] + action,0,env.max_cars)
		)
		
		a_rented = rand(Poisson(3))
		b_rented = rand(Poisson(4))
		a_returned = rand(Poisson(3))
		b_returned = rand(Poisson(2))
		
		new_state = (
			clamp(new_state[1] - a_rented + a_returned, 0, env.max_cars),
			clamp(new_state[2] - b_rented + b_returned, 0, env.max_cars)
		)
		
		if min(state...) ≤ 0
			return (0, (20,20))
		end
		return ((a_rented+b_rented)*10 - abs(action)*2, new_state)
	end
	function (env::CarRental)(action, state, flag)
		
		action = clamp(action, -state[2], state[1])
		new_state = (
			clamp(state[1] - action,0,env.max_cars),
			clamp(state[2] + action,0,env.max_cars)
		)
		
		if flag == "mean"
			a_rented = 3
			b_rented = 4
			a_returned = 3
			b_returned = 2

			new_state = (
				clamp(new_state[1] - a_rented + a_returned, 0, env.max_cars),
				clamp(new_state[2] - b_rented + b_returned, 0, env.max_cars)
			)

			if min(new_state...) ≤ 0
				return (0, (0,0))
			end
			return ((a_rented+b_rented)*10 - abs(action)*2, new_state)
		elseif flag == "all"
			pax, pbx = Truncated(Poisson(3), 0,10),Truncated(Poisson(4), 0,10)
			pay, pby = Truncated(Poisson(3), 0,10),Truncated(Poisson(2), 0,10)
			ret = []
			for ax = 0:10, bx = 0:10, ay=0:10, by=0:10
				p = pdf(pax, ax)*pdf(pbx, bx)*pdf(pay, ay)*pdf(pby, by)
				new_state = (
					clamp(new_state[1] - ax + ay, 0, env.max_cars),
					clamp(new_state[2] - bx + by, 0, env.max_cars)
				)
				if min(new_state...) ≤ 0
					r, new_state =  0, (0,0)
				else
					r = (ax+bx)*10 - abs(action)*2
				end
				push!(ret, (p, r, new_state))
			end
			return ret
		elseif flag == "simple"
			pax = [0.1, 0.4, 0.4, 0.1]
			pay = [0.1, 0.4, 0.4, 0.1]
			pbx = [0.1, 0.1, 0.3, 0.4, 0.1]
			pby = [0.2, 0.4, 0.3, 0.1]
			ret = []
			for ax = 0:3, bx = 0:4, ay = 0:3, by = 0:3
				p = pax[ax+1]*pbx[bx+1]*pay[ay+1]*pby[by+1]
				new_state = (
					clamp(new_state[1] - ax + ay, 0, env.max_cars),
					clamp(new_state[2] - bx + by, 0, env.max_cars)
				)
				if min(new_state...) ≤ 0
					r, new_state =  0, (0,0)
				else
					r = (ax+bx)*10 - abs(action)*2
				end
				push!(ret, (p, r, new_state))
			end
			return ret
		else
			throw("Invalid flag")
		end
	end
end

# ╔═╡ e8323dd8-3ae0-4509-a8ae-f5db3a38e4a4
mutable struct Agent <: AbstractAgent
	q::Array{Float64,3}
	v::Array{Float64,2}
	state::Tuple
	actions::Vector
	
	Agent(m::Integer,n::Integer, state::Tuple, actions, init::Float64=1.) = new([init for i = 1:m, j=1:n, k=1:length(instances(actions))],[init for i=1:m, j=1:n], state, [a for a = instances(Action)])
	
	
	Agent(env::Gridworld, actions, init=1.) = Agent(env.m, env.n, (rand(1:env.m), rand(1:env.n)), actions, init)
	
	Agent(env::CarRental, actions, init=0.) = Agent(env.max_cars, env.max_cars, (env.max_cars, env.max_cars), actions, init)
end

# ╔═╡ 2021299e-c672-4414-9749-e80a0cc90cee
begin
	struct RandomPolicy <: AbstractPolicy
		RandomPolicy() = new()
		function (policy::RandomPolicy)(agent::Agent)
			rand(agent.actions)
		end
	end
end

# ╔═╡ 5cb50a3c-01ce-487b-b451-e7ca56933aae
function get_action(agent::Agent, π::RandomPolicy)
	return rand(agent.actions)
end

# ╔═╡ a48e130f-2876-44cc-bfdd-746251836b55
struct GreedyPolicy <: AbstractPolicy
	ε::Float64
	GreedyPolicy(ε) = ((ε < 0.) | (ε > 1.)) ? error("ε outside [0,1]") : new(ε)
	GreedyPolicy() = GreedyPolicy(0.1)
	function (policy::GreedyPolicy)(agent::Agent)
		if rand() < π.ε
			return rand(agent.actions)
		else
			return agent.actions[argmax(agent.q[agent.state..., :])]
	end
end
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

# ╔═╡ 266aab70-e346-4a64-ae3a-993cd94ab534
function get_action(agent::Agent, π::SamplingPolicy)
	probs = softmax(agent.q[agent.state..., :]/π.τ)
	Action(sample(1:5, ProbabilityWeights(probs, 1)))
end

# ╔═╡ 8034730a-0532-425c-8009-573edda39bec
get_action(Agent(world, Action), GreedyPolicy())

# ╔═╡ 58cb61b9-25ab-4235-a74c-cf701a4c5411
get_action(Agent(world, Action), SamplingPolicy())

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
	heatmap(agent.v, yflip=true, annotations=anno; kwargs...)
end

# ╔═╡ 97f374ff-e713-4dea-b30a-eec1f4a767dd
function get_action_probs(agent::Agent, π::RandomPolicy)
	return ones(length(actions))./length(actions)
end

# ╔═╡ 997228c8-7367-4ba6-8c37-e70c36b5c361
function get_action_probs(agent::Agent, π::GreedyPolicy)
	n = length(agent.actions)
	q = agent.q[agent.state..., :]
	idxs = max_idx(q)
	greedy_probs = zeros(n)
	greedy_probs[idxs] .= 1. /length(idxs)
	if π.ε == 0
		return greedy_probs
	else
		return ones(n).*(π.ε/n) + (1-π.ε).*greedy_probs
	end
end

# ╔═╡ 738b38b4-35ba-493d-84bc-42b3a43d2cea
get_action_probs(Agent(world, Action), GreedyPolicy())

# ╔═╡ eff3da75-14db-4d98-8d22-3cd9fbc694c7
function policy_evaluation(policy::Array, agent::Agent, env::Gridworld, max_iter=1000)
	m,n = size(agent.v)
	all_states = reshape([(i,j) for i = 1:m, j = 1:n], 1,m*n)
	n = 0
	for _ = 1:max_iter
		n += 1
		Δ = 0
		V = copy(agent.v)
		for s in all_states
			v = V[s...]
			agent.state = s
			a_probs = policy[s..., :]
			
			possible_actions = filter(i -> a_probs[i] > 1e-5, 1:length(a_probs))
			new_v = 0
			for i = possible_actions
				r, s′ = env(actions[i], s)
				if s in keys(env.special_states)
					vs′ = 0
				else
					vs′ = agent.v[s′...]
				end
				agent.q[s..., i] = (r + 1. *vs′)
				new_v += a_probs[i] * (r + 1. *vs′)
			end
			
			V[s...] = new_v
			Δ = max(Δ, abs(v-new_v))
		end
		agent.v = V
		if Δ < 0.001
			break
		end
	end
	agent.v, n
end

# ╔═╡ c7c05fbe-2ae8-41fd-ab42-5068444b6043
function policy_update(policy::Array, agent::Agent, env::Gridworld, γ=0.9)
	stable = true
	m,n,_ = size(policy)
	all_states = reshape([(i,j) for i = 1:m, j = 1:n], 1,m*n)
	for s in all_states
		old_action = policy[s..., :]
		new_action = zeros(size(old_action))
		action_values = similar(new_action)
		for (i, a) in enumerate(agent.actions)
			r, s′ = env(a, s)
			if s in keys(env.special_states)
				vs′ = 0 # for terminal states
			else
				vs′ = agent.v[s′...]
			end
			
			action_values[i] = r + γ * vs′
		end
		idx = max_idx(action_values)
		new_action[idx] .= 1/length(idx)
		if maximum(new_action - old_action) > 1e-5
			stable = false
		end
		policy[s..., :] = new_action
	end
	return policy, stable
end

# ╔═╡ f3c9a7b1-2777-4072-a7f4-22c35874458b
function annotated_heatmap!(p, agent::Agent, policy::Array; kwargs...)
	labels = get_arrows(agent)
	anno = make_anno(agent.v, labels=labels)
	heatmap(agent.v, yflip=true, annotations=anno; kwargs...)
end

# ╔═╡ 186ef7b4-2da7-46b0-b548-bd8d8fc0c5af
let
	agent = Agent(gridworld1, Action, 0.)
	max_iter = 100
	m,n = size(agent.v)
	all_states = reshape([(i,j) for i = 1:m, j = 1:n], 1,m*n)
	n = 0
	for _ = 1:k1
		n += 1
		Δ = 0
		V = copy(agent.v)
		for s in all_states
			v = V[s...]
			agent.state = s
			a_probs = get_action_probs(agent, RandomPolicy())
			
			possible_actions = filter(i -> a_probs[i] != 0., 1:length(a_probs))
			new_v = 0
			for i = possible_actions
				r, s′ = gridworld1(actions[i], s)
				if s in keys(gridworld1.special_states)
					vs′ = 0
				else
					vs′ = agent.v[s′...]
				end
				agent.q[s..., i] = (r + 1. *vs′)
				new_v += a_probs[i] * (r + 1. *vs′)
			end
			
			V[s...] = new_v
			agent.v = V
			Δ = max(Δ, abs(v-new_v))
		end
		# agent.v = V
		if Δ < 0.001
			break
		end
	end
	left = annotated_heatmap!(plot(), agent; aspect_ratio=1, colorbar=:none)
	right = annotated_heatmap!(plot(), agent, true; aspect_ratio=1, colorbar=:none)
	plot(left, right)
end
		

# ╔═╡ 55193f21-566a-4b53-b326-c90ef3f69bc6
let
	env = CarRental(20)
	length(env(0, (20,20), "all"))
end

# ╔═╡ 588db6bb-cb41-4365-a933-24e456ceab60
function is_terminal(s::Tuple, env::CarRental)
	return min(s...) ≤ 0
end

# ╔═╡ a03873ab-975e-4cd0-9382-99926e9ea96a
function policy_evaluation(policy::Array, agent::RentalAgent, env::CarRental, max_iter=1000, γ=0.9)
	m,n = size(agent.v)
	all_states = reshape([(i,j) for i = 1:m, j = 1:n], 1,m*n)
	n = 0
	for _ = 1:max_iter
		n += 1
		Δ = 0
		V = copy(agent.v)
		for s in all_states
			v = V[s...]
			agent.state = s
			a_probs = policy[s..., :]
			
			possible_actions = [i for i=1:length(agent.actions) if (a_probs[i] > 0) & (agent.actions[i] >= -s[2]) & (agent.actions[i] <= s[1])]
			new_v = 0
			for i = possible_actions
				r, s′ = env(agent.actions[i], s, "mean")
				vs′ = agent.v[s...]
				agent.q[s..., i] = (r + γ *vs′)
				new_v += a_probs[i] * (r + γ*vs′)
			end
			
			V[s...] = new_v
			Δ = max(Δ, abs(v-new_v))
		end
		agent.v = V
		if Δ < 0.001
			break
		end
	end
	agent.v, n
end

# ╔═╡ 565d50a3-ab61-48f8-93d0-2b7c23b3b47e
let
	agent = Agent(gridworld1, Action, 0.)
	policy = ones(size(agent.q)) ./ length(agent.actions)
	
	v, n = policy_evaluation(policy, agent, gridworld1)
	annotated_heatmap!(plot(), agent; colorbar=:none)
end

# ╔═╡ 4e930947-bec0-47a9-865e-c907c2545504
function policy_update(policy::Array, agent::RentalAgent, env::CarRental, γ=0.9)
	stable = true
	m,n,_ = size(policy)
	all_states = reshape([(i,j) for i = 1:m, j = 1:n], 1,m*n)
	for s in all_states
		old_action = policy[s..., :]
		new_action = zeros(size(old_action))
		action_values = similar(new_action)
		for (i, a) in enumerate(agent.actions)
			r, s′ = env(a, s)
			if (s′[1] == 0) | (s′[2] == 0)
				vs′ = 0
			else
				vs′ = agent.v[s′...]
			end
			action_values[i] = r + γ * vs′
		end
		idx = max_idx(action_values)
		new_action[idx] .= 1/length(idx)
		if maximum(new_action - old_action) > 1e-5
			stable = false
		end
		policy[s..., :] = new_action
	end
	return policy, stable
end

# ╔═╡ 998f5fbc-e44a-40fb-a296-f805dabd816a
let
	agent = Agent(gridworld1, Action, 0.)
	policy = ones(size(agent.q)) ./ length(agent.actions)
	stable = false
	n = 0
	for i = 1:k2
		n += 1
		v, n = policy_evaluation(policy, agent, gridworld1)

		policy, stable = policy_update(policy, agent, gridworld1, 1.)
		if stable
			break
		end
	end
	if stable
		title = "The policy is stable"
	else
		title = "The policy is unstable"
	end
	annotated_heatmap!(plot(), agent, policy; colorbar=:none, title=title)
	# annotated_heatmap!(plot(), agent; colorbar=:none, title=title)
end

# ╔═╡ 13f2f7a6-d3a5-4fa4-ab19-03d4c2f91f18
function is_terminal(state::Tuple, env::AbstractEnv)
	false
end

# ╔═╡ 554cb669-1f95-496b-9aa1-6ab59a62e952
struct GamblingEnv <: AbstractEnv
	ph::Float64
	s_max::Number
	GamblingEnv(ph::Float64) = new(ph, 100)
	function (env::GamblingEnv)(a::Number, s::Tuple)
		s = s[1]
		if rand() < env.ph
			new_s = s + a
		else
			new_s = s - a
		end
		return Int(new_s == env.s_max), (new_s, )
	end
	function (env::GamblingEnv)(a::Number, s::Tuple, flag::String)
		s = s[1]
		if flag == "all"
			win_r = Int(s + a == env.s_max)
			return [(env.ph, win_r, (s+a, )), ((1-env.ph), 0, (s-a, ))]
		else
			throw("Incrorrect flag")
		end
	end
end

# ╔═╡ 9eb29cfd-abff-42c7-a6e2-708e57d3de3f
mutable struct GamblingAgent <:AbstractAgent
	v::Array
	q::Array
	state::Tuple
	actions
	
	GamblingAgent(env::GamblingEnv) = new(zeros(env.s_max-1), zeros(env.s_max-1, env.s_max ÷ 2), (50,), 1:env.s_max ÷ 2)
end

# ╔═╡ c7efa000-006a-466a-aa6b-bac75db68130
function valid_actions(agent::GamblingAgent)
	return 1:min(agent.state[1], length(agent.v)+1 - agent.state[1])
end

# ╔═╡ 2d00665b-8033-45cb-a199-55d57979f11e
function get_valid_states(agent::GamblingAgent)
	[(i, ) for i = 1:size(agent.v)[1]]
end

# ╔═╡ c5664612-0d17-43c4-8985-29ef14ce6d39
function is_terminal(s::Tuple, env::GamblingEnv)
	return (s[1] == 0) || (s[1] == env.s_max)
end

# ╔═╡ 67dd5195-7088-4dab-8153-f9553b3695d8
let
	if run2
	env = CarRental(20)
	agent = RentalAgent(20, car_actions)
	γ = 0.9
	policy = zeros(20,20,length(agent.actions))
	policy[:, :, 6] .= 1.
	
	max_iter = 10
	m,n = size(agent.v)
	all_states = reshape([(i,j) for i = 1:m, j = 1:n], 1,m*n)
	n = 0
	foo = nothing
	for _ = 1:max_iter
		n += 1
		Δ = 0
		V = agent.v
		for s in all_states
			v = V[s...]
			agent.state = s
			a_probs = policy[s..., :]
			
			possible_actions = [i for i=1:length(agent.actions) if (a_probs[i] > 0) & (agent.actions[i] >= -s[2]) & (agent.actions[i] <= s[1])]
			new_v = 0
			for i = possible_actions
				outcomes = env(agent.actions[i], s, "simple")
				gt = 0
				for (p,r,s′) = outcomes
					if is_terminal(s′, env)
						vs′ = 0
					else
						vs′ = agent.v[s′...]
					end
					gt += p*(r + γ *vs′)	
				end
				agent.q[s..., i] = gt
				new_v += a_probs[i] * gt
			end
			
			V[s...] = new_v
			Δ = max(Δ, abs(v-new_v))
		end
		agent.v = V
		if Δ < 0.001
			break
		end
	end
	agent.v, n, foo
	# heatmap(agent.v)
	end
end

# ╔═╡ d61ceca5-f015-4ed9-981d-92f0079e749c
function value_iteration(agent::AbstractAgent, env::AbstractEnv, max_iter=1000, γ=0.9)
	
	m,n = size(agent.v)
	all_states = reshape([(i,j) for i = 1:m, j = 1:n], 1,m*n)
	Δ = 0
	n = 0
	for _ = 1:max_iter
		n += 1
		V = agent.v
		for s in all_states
			v = V[s...]
			agent.state = s
			
			possible_actions = [i for i = 1:length(agent.actions) if valid_action(i, agent.actions, state)]
			new_v = nothing
			for i = possible_actions
				r, s′ = env(actions[i], s)
				if s in keys(env.special_states)
					vs′ = 0
				else
					vs′ = agent.v[s′...]
				end
				gt = (r + γ *vs′) # expected return
				agent.q[s..., i] = gt
				if (new_v == nothing) || (gt > new_v)
					new_v += (r + 1. *vs′)
				end
			end
			@assert !(new_v == nothing)
			V[s...] = new_v
			Δ = max(Δ, abs(v-new_v))
		end
		agent.v = V
		if Δ < 0.001
			break
		end
	end
	agent.v, n
	
	policy = zeros(size(agent.q))
	for s = all_states
		optimal_action = nothing
		possible_actions = [i for i = 1:length(agent.actions) if valid_action(i, agent.actions, state)]
		v = nothing
		for i = possible_actions
			r, s′ = env(agent.actions[i], s)
			if is_terminal(s′, env)
				vs′ = 0
			else
				vs′ = agent.v[s′...]
			end
			gt = (r + γ *vs′)
			if (v == nothing) || (gt > v)
				optimal_action = i
				v = gt
			end
		end
		@assert !(optimal_action == nothing)
		policy[s..., i] = 1
	end
	policy
end

# ╔═╡ bb745adc-330e-4c4e-b956-3d5410a428eb
let
	env = GamblingEnv(0.5)
	env(50, (50,))
	agent = GamblingAgent(env)
	agent.state = (80,)
	valid_actions(agent)
end

# ╔═╡ e3b6f21e-f388-40fe-80ca-1bd6c47c080b
let
	env = GamblingEnv(ph)
	agent = GamblingAgent(env)
	all_states = get_valid_states(agent)
	γ = 1.
	n = 0
	vp = plot(title = "Value function estimate", ylabel="E(V)", xlabel="capital", legend=:topleft)
	max_iter = k4
	if k4 > 40
		max_iter = 1000
	end
	for step = 1:max_iter
		Δ = 0
		n += 1
		V = agent.v
		for s = all_states
			v = V[s...]
			agent.state = s
			
			possible_actions = valid_actions(agent)
			new_v = nothing
			for i = possible_actions
				outcomes = env(agent.actions[i], s, "all")
				gt = 0 # expected return
				for (p, r, s′) = outcomes
					if is_terminal(s′, env)
						vs′ = 0
					else
						vs′ = agent.v[s′...]
					end
					gt += p*(r + γ *vs′)
				end
				agent.q[s..., i] = gt
				if (new_v == nothing) || (gt > new_v)
					new_v = gt
				end
			end
			@assert !(new_v == nothing)
			V[s...] = new_v
			Δ = max(Δ, abs(v-new_v))
		end
		agent.v = V
		if Δ < 1e-5
			break
		end
		if step in (1,2,3,10,32,100,500)
			plot!(vp, agent.v, label="step $step")
		end
	end
	agent.v, n
	vp
	policy = zeros(size(agent.q))
	for s = all_states
		agent.state = s
		optimal_action = nothing
		possible_actions = valid_actions(agent)
		v = nothing
		for i = possible_actions
			outcomes = env(agent.actions[i], s, "all")
				gt = 0 # expected return
				for (p, r, s′) = outcomes
					if is_terminal(s′, env)
						vs′ = 0
					else
						vs′ = agent.v[s′...]
					end
					gt += p*(r + γ *vs′)
				end
			if (v == nothing) || (gt > v)&(abs(gt - v) > 1e-5)
				optimal_action = i
				v = gt
			end
		end
		@assert !(optimal_action == nothing)
		policy[s..., optimal_action] = 1
	end
	p_star = [argmax(policy[i, :]) for i = 1:size(policy)[1]]
	pp = scatter(p_star, title="Policy", xlabel="capital", ylabel="stake", label=:none, markersize=2)
	plot!(pp, p_star, alpha=0.3, label=:none)
	l = @layout [a; b]
	plot(vp, pp, layout=l)
end

# ╔═╡ 9e3394fe-3a55-4a5a-9f09-b9083541c6b7
mutable struct Position
	x::Integer
	y::Integer
end

# ╔═╡ ffb8607f-8b56-4549-ac1a-c5e90ede5912
begin
	function Base.:+(a::Position, b::Position)
		return Position(a.x+b.x, a.y+b.y)
	end
	function Base.:-(a::Position, b::Position)
		return Position(a.x-b.x, a.y-b.y)
	end
	function to_tuple(c::Position)
		return c.x, c.y
	end
end

# ╔═╡ 3ea4a4cc-12e1-4a83-8c2b-c87721e715d4
Position(1,1) + Position(1,2), Position(1,1) - Position(2,2)

# ╔═╡ Cell order:
# ╠═f2cc03f2-aab2-11eb-2cc1-b77ff8eaf5b3
# ╠═37db68da-32ca-49f7-acb8-c6afa5f1fd5f
# ╠═bc3c5dc0-976f-4158-950b-61155904a9ba
# ╟─299d4065-e294-44ea-8fd6-8bd0e641bc2c
# ╠═9af358f5-0466-4e73-a401-9e45ef197dbd
# ╟─9cb61687-6aa9-4632-93b5-d4208cef1db1
# ╟─ac443d5e-f8f9-423f-8b8d-8516d6bab2bd
# ╟─df197549-817e-4618-afe4-e96e1a46d68e
# ╟─25b8e0fe-34bd-4251-8156-54b6daadbfa5
# ╟─a5ecc150-f644-491f-98dd-60f670c74888
# ╟─8d1ff438-8d08-4134-9265-c41cfe609197
# ╟─e8323dd8-3ae0-4509-a8ae-f5db3a38e4a4
# ╠═f9d31ad6-3446-481d-a7f5-faafeb1d87c9
# ╠═2021299e-c672-4414-9749-e80a0cc90cee
# ╠═b0e44de6-3b3c-455c-8261-7b80bbfb4c40
# ╠═5cb50a3c-01ce-487b-b451-e7ca56933aae
# ╠═a48e130f-2876-44cc-bfdd-746251836b55
# ╟─84de4363-ee9a-4dc2-9650-5f21d427f454
# ╠═5132b64c-3ef1-45d6-9c43-f64a2efa4fb0
# ╠═fa747c20-d3d2-43e0-a9e3-2d65981e6cf9
# ╠═b4c4ee9c-6dc9-47e2-b87e-525c198751d1
# ╠═a0047a8d-cb1f-44e4-bfe5-a294b2264001
# ╠═8034730a-0532-425c-8009-573edda39bec
# ╠═f7459692-fdd5-4e5b-bc3c-f637d86d0fe9
# ╠═266aab70-e346-4a64-ae3a-993cd94ab534
# ╠═58cb61b9-25ab-4235-a74c-cf701a4c5411
# ╠═bae52d01-015e-4029-8c9b-d9b032abc43c
# ╠═bca44f3f-cf45-481f-b383-d3abdd36b4fc
# ╟─9c5bf918-e071-4b2d-af03-1eede3b31439
# ╠═017f1aaa-ad51-4698-b6db-5a2cc26dcfe7
# ╟─b85630be-0de6-4a3b-8e65-dd09c6b8cd09
# ╠═59d7151c-09a5-4531-ad14-433bdbb4269b
# ╠═f7153663-096e-4560-9a5e-9d82a26cb653
# ╟─ce1e9402-d7d8-42e9-bd17-23989a40976e
# ╠═97f374ff-e713-4dea-b30a-eec1f4a767dd
# ╠═997228c8-7367-4ba6-8c37-e70c36b5c361
# ╠═738b38b4-35ba-493d-84bc-42b3a43d2cea
# ╟─14c52ff7-ed4d-4984-adfa-01fd33104a88
# ╟─e25f1ecc-279e-4987-aa3b-1edf87c99baa
# ╠═1feed7d1-eb2c-4c39-9a51-3f6929441ccf
# ╠═186ef7b4-2da7-46b0-b548-bd8d8fc0c5af
# ╠═eff3da75-14db-4d98-8d22-3cd9fbc694c7
# ╠═565d50a3-ab61-48f8-93d0-2b7c23b3b47e
# ╟─dd0457e6-8d02-4994-96dc-bf977ace13f9
# ╠═c7c05fbe-2ae8-41fd-ab42-5068444b6043
# ╟─e7335f7e-c5df-457d-b7b5-8cd20ba00328
# ╟─f3c9a7b1-2777-4072-a7f4-22c35874458b
# ╠═69653c0a-c2ff-4fe9-bb6b-fc84188032e9
# ╠═998f5fbc-e44a-40fb-a296-f805dabd816a
# ╟─bc784903-90dc-42a0-9617-65b5813a9ec9
# ╟─ae927a87-1169-479e-bdf6-77b78b4f9892
# ╠═1f27ca59-901d-4751-a3ae-20d3a4563fe5
# ╟─4e7a19e2-71a8-4dc2-84f3-7a1f67bdf812
# ╟─9ee1d663-7e3d-4c14-ae15-227895bd3952
# ╟─293096c6-b4b8-4fd6-b546-2b5d9f3bc180
# ╠═55193f21-566a-4b53-b326-c90ef3f69bc6
# ╠═f11b080e-b689-4b5b-8525-e36bd2f08658
# ╟─2d60f367-c5cf-4bac-9bd6-50c6646598ba
# ╠═588db6bb-cb41-4365-a933-24e456ceab60
# ╠═e8bb6fae-00e9-4fc5-ae4d-13dc6145ba81
# ╠═67dd5195-7088-4dab-8153-f9553b3695d8
# ╠═a03873ab-975e-4cd0-9382-99926e9ea96a
# ╠═5df84a7f-a025-47af-aee1-861b86fba7f7
# ╟─4e930947-bec0-47a9-865e-c907c2545504
# ╟─3a58fc77-8e56-4fa3-9751-bd46f240d249
# ╠═41298949-f106-4618-a358-c567445bc123
# ╟─53cc743b-bf03-43a9-9e2d-8ea326eec091
# ╠═01a7c860-da17-44c5-a96b-9ba3e81a066f
# ╠═13f2f7a6-d3a5-4fa4-ab19-03d4c2f91f18
# ╠═d61ceca5-f015-4ed9-981d-92f0079e749c
# ╠═9eb29cfd-abff-42c7-a6e2-708e57d3de3f
# ╠═554cb669-1f95-496b-9aa1-6ab59a62e952
# ╠═c5664612-0d17-43c4-8985-29ef14ce6d39
# ╠═c7efa000-006a-466a-aa6b-bac75db68130
# ╠═bb745adc-330e-4c4e-b956-3d5410a428eb
# ╠═2d00665b-8033-45cb-a199-55d57979f11e
# ╠═d8ba47f1-c0a3-4445-8972-632a53895b80
# ╟─a4f22ea9-3de0-4c5b-b266-e0d04f60f0a1
# ╠═e3b6f21e-f388-40fe-80ca-1bd6c47c080b
# ╠═536c684c-feb0-4a18-88b2-cefef417479f
# ╠═ba1ad05b-d77f-4eb8-915f-066e6bd8c5fa
# ╟─6cad08a4-df22-4db4-8446-73657fc369b5
# ╟─42c95284-fa39-4811-85f3-e2eca96e0cfe
# ╟─8d9c6117-e514-4a12-9c4b-8de70f445b53
# ╟─a419de6a-fa5c-439e-ac64-ed576e0da6be
# ╠═3b0047ae-4713-470f-a145-6a97cf93f090
# ╠═43036305-86fb-4311-9671-98419a547d9b
# ╟─9e3394fe-3a55-4a5a-9f09-b9083541c6b7
# ╟─ffb8607f-8b56-4549-ac1a-c5e90ede5912
# ╟─3ea4a4cc-12e1-4a83-8c2b-c87721e715d4
