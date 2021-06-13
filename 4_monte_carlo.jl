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
md"# Chapter 5. Monte Carlo Methods"

# ╔═╡ 8bcc1011-6a64-4e33-b24f-8c26c690fa91
md"In this chapter we are learning about Monte Carlo Methods in Monte Carlo style - by exploring Blackjack game."

# ╔═╡ 788836c3-f7a9-47a1-804e-2ea0127e7b02
md"## Blackjack problem formulation"

# ╔═╡ df8ad838-abe9-4f43-8f70-944f861ff275
md"""
Game description (Example 5.1 from the [book](http://incompleteideas.net/book/the-book.html))

> The object of the popular casino card game of blackjack is to obtain cards the sum of whose numerical values is as great as possible without exceeding 21. All face cards count as 10, and an ace can count either as 1 or as 11. We consider the version in which each player competes independently against the dealer. The game begins with two cards dealt to both dealer and player. One of the dealer’s cards is face up and the other is face down. If the player has 21 immediately (an ace and a 10-card), it is called a natural. He then wins unless the dealer also has a natural, in which case the game is a draw. If the player does not have a natural, then he can request additional cards, one by one (hits), until he either stops (sticks) or exceeds 21 (goes bust). If he goes bust, he loses; if he sticks, then it becomes the dealer’s turn. If the dealer goes bust, then the player wins; otherwise, the outcome -- win, lose, or draw -- is determined by whose final sum is closer to 21.
"""

# ╔═╡ 9b3e4e24-5b53-49e1-87e9-37938da0b83d
@enum BlackjackAction stick=1 hit=2

# ╔═╡ 00504589-f2bd-4496-bd03-740cd5a1c7a1
ACE = 1

# ╔═╡ 0d899fa7-53d0-42f4-ab26-454c2a9a6079
md"The dealer hits or sticks according to a fixed strategy without choice: he sticks on any sum of 17 or greater, and hits otherwise."

# ╔═╡ 75d91ff3-c9c4-4680-93bf-aaeb1daaf8de
function add_card(total, new_card, usable_ace)
	if new_card == ACE
		if total < 11
			return total + 11, true
		else
			return total + 1, usable_ace
		end
	else
		total += new_card
		if total > 21 && usable_ace
			return total - 10, false
		else
			return total, usable_ace
		end
	end
end

# ╔═╡ bd4f3e10-d152-470e-aac3-582dd5821085
function dealersum(first_card::Int, pw)
	if first_card == ACE
		usable_ace = true
		total = 11
	else
		usable_ace = false
		total = first_card
	end
	while total < 17
		new_card = sample(ACE:10, pw)
		total, uasble_ace = add_card(total, new_card, usable_ace)
	end
	return total
end

# ╔═╡ 546812ec-ea48-423a-8d32-d61083decc44
check_sum(x) = x > 21 ? 0 : x

# ╔═╡ f86c7c26-a34a-4de2-9626-f7e37a7cabbf
check_sum(21), check_sum(15), check_sum(22)

# ╔═╡ 8fb8cda9-3e6c-4eb8-bf01-69509008af92
md"""
The `Blackjack` envirenment is responcible for state transitions and rewards. Given a state and an action the envirenment returns 3-tuple of reward, new state and a boolian indicator of the episode termnation.
"""

# ╔═╡ 7ccbae20-160b-45be-814c-ef96b85fc40a
begin
	struct Blackjack <: AbstractEnv
		w
		Blackjack() = new(fweights(map(x -> x == 10 ? 4 : 1, ACE:10)))
	end
	
	function (env::Blackjack)(state::Tuple, a::BlackjackAction)
		player_sum, dealer_card, usable_ace = state
		final_state = (nothing, nothing, nothing)
		if a == hit
			new_card = sample(ACE:10, env.w)
			player_sum, usable_ace = add_card(player_sum, new_card, usable_ace)
			if player_sum > 21
				return -1, final_state, true
			else
				return 0, (player_sum, dealer_card, usable_ace), false
			end
		else (a == stick)
			dealer_sum = dealersum(dealer_card, env.w)
			player_sum = check_sum(player_sum)
			dealer_sum = check_sum(dealer_sum)
			if player_sum == dealer_sum 
				r = 0 #draw
			elseif player_sum > dealer_sum
				r = 1 #win
			else
				r =-1 #loose
			end
			return r, final_state, true
		end
	end
end

# ╔═╡ fa715a39-d413-4c89-b243-90f28e5b1d80
let
	env = Blackjack()
	env((19, 1, false), hit)
end

# ╔═╡ 6c9a108b-5ac0-43c8-84bf-2459c9674587
mutable struct BlackjackAgent <: AbstractAgent
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

# ╔═╡ 00cf6607-c4c6-49c9-a508-d32f54e7060f
function create_agent()
	v = zeros(10, 10, 2)
	q = zeros(10, 10, 2, 2)
	pw = fweights(map(x -> x == 10 ? 4 : 1, ACE:10))
	cards = sample(ACE:10, pw, 2)
	usable_ace = (ACE in cards)
	cards = replace(cards, (ACE=>1))
	player_sum = sum(cards)
	if usable_ace
		player_sum += 10
	end
	dealer_card = sample(ACE:10, pw)
	state = (player_sum, dealer_card, usable_ace)
	
	BlackjackAgent(v, q, state, [stick, hit])
end

# ╔═╡ c7284971-c9b8-40fe-9541-ebb910dbfab2
create_agent().state

# ╔═╡ ced93e92-0ca0-4359-b038-6feb08258dda
md"## Value function estimation"

# ╔═╡ 6a31eaa3-9309-4b31-ac57-0e3204ff8bfa
md"First we will apply Monte Carlo policy evaluation to simple deterministic policy defined below:"

# ╔═╡ 30278c80-38ea-4ee1-9ff8-7b924020dd87
function simple_policy(state::Tuple)
	if state[1] in [20, 21]
		return stick
	else
		return hit
	end
end

# ╔═╡ 3bbe3335-68c5-4efe-9800-e9505b8a63b9
function episode(agent::BlackjackAgent, env::Blackjack, π::Function)
	history = []
	finished = false
	state = agent.state
	i = 0
	while !(finished) && (i < 100)
		i += 1
		action = π(state)
		reward, new_state, finished = env(state, action)
		push!(history, (s=state, a=action, r=reward))
		state = new_state
	end
	history
end

# ╔═╡ fe2df29d-3296-4eec-aee9-5d91b086d73b
md"""
Number of episodes $(@bind n_episodes_eval Select(["10000", "100000","500000", "1000000"]))

Every visit $(@bind ev_eval CheckBox())
"""

# ╔═╡ e0f41a91-2cfc-432d-b977-e4f8e73047de
md"""
Show 3D plot $(@bind show_3d_1 CheckBox())
"""

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

# ╔═╡ dcd5c9db-6663-4cc9-bdbd-b107131e763a
md"## Action values estimation"

# ╔═╡ b70bcf37-2b1e-4621-a378-d9ab72de50c4
md"""
Estimatin _action_ values $$q_π(s,a)$$ (rather then _state_ values) is particularly useful in a case when we don't have a model of the envirenment. State values alone are not enaugh to find optimal policy in this case as we cannot perform lookahead to determine preferred next state. Action values alone on the other hand provide an actionable signal, we can directly select the action with highest value for a given state.

To estimate $$q_π(s,a)$$ we need to make sure each state-action pair is reachable.
"""

# ╔═╡ d78418eb-07fe-4f9b-a073-c79cccb0bbb9
md"""
### Exploring starts

One way to ensure that all state-action pairs are visited during simulation is _exploring starts_. Each episode is initialised with both state and action selected at random.
"""

# ╔═╡ b972b736-e97d-496b-9913-466dc9ee4664
function episode_es(agent::BlackjackAgent, env::Blackjack, π::Array)
	"Episode with exploring statrt"
	history = []
	finished = false
	# randomize initial state
	state = state = (rand(4:21), rand(1:10), rand() > 0.5)
	i = 0
	while !(finished) && (i < 100)
		i += 1
		if i == 1
			# randomize first action
			action = BlackjackAction(rand(1:2))
		else
			action = BlackjackAction(argmax(π[state2idx(state)..., :]))
		end
		reward, new_state, finished = env(state, action)
		push!(history, (s=state, a=action, r=reward))
		state = new_state
	end
	history
end

# ╔═╡ 89aa9ab7-7afc-4569-9f71-9881f1666bf3
let
	π = zeros(21-3,10,2,2)
	π[1:21-5, :, :, 2] .= 1
	π[21-4:21-3, :, :, 1] .= 1
	@assert all(sum(π, dims=4) .== 1)
	episode_es(create_agent(), Blackjack(), π)
end

# ╔═╡ f08c8021-11e1-4443-8e86-0f2a664cecf1
md"""
## Monte Carlo Control

To find optimal policy we apply the idea of generalized policy iteration to the Monte Carlo methods. After each episode the action values are updated and greedy policy is selected.
"""

# ╔═╡ 40d598a9-3e7d-4929-ba1f-82fd594255aa
md"### Exploring starts"

# ╔═╡ 614e7c44-1173-4dda-854b-c6915a453249
function mces(max_iter=100)
	# π = zeros(21-3, 10, 2, 2) .+ 0.5
	π = zeros(21-3,10,2,2)
	π[1:21-5, :, :, 2] .= 1
	π[21-4:21-3, :, :, 1] .= 1
	@assert all(sum(π, dims=4) .== 1)
	env = Blackjack()
	γ = 1
	Q = zeros(21-3, 10, 2, 2)
	Nv = zeros(size(Q))
	history = []
	for i = 1:max_iter
		agent = create_agent()
		history = episode_es(agent, env, π)
		visited_states = map(x -> (x.s, x.a), history)
		G = 0
		T = length(history)
		for i = T:-1:1
			s, a, r = history[i]
			G = γ*G + r
			if !((s, a) in visited_states[1:i-1])
				idx = (state2idx(s)..., Int(a))
				# Q-values update
				try
					Nv[idx...] += 1
					Q[idx...] += (G - Q[idx...])/Nv[idx...]
				catch
					return history, idx
				end
				# policy update
				idx = idx[1:3]
				a_star = argmax(Q[idx..., :])
				π[idx..., :] .= 0.
				π[idx..., a_star] = 1.
				# @assert all(sum(π, dims=4) .== 1)
			end
		end
	end
	Q, π, Nv
end

# ╔═╡ ca28c10a-1495-4704-b521-e11653e5848a
md"Estimates obtained after 10000 episodes are quite noisy. It's necessary to run at least for 500000 episodes to obtain more reliable policy."

# ╔═╡ e239eae0-4468-4860-80cd-855dc546878f
md"""
Number of episodes $(@bind n_episodes_mcse Select(["10000", "100000","500000", "1000000"]))
"""

# ╔═╡ 6636140b-cbb1-4a05-b724-50ed1188efe5
Q, π, Nv = mces(parse(Int, n_episodes_mcse));

# ╔═╡ 4ee8e017-3c54-438e-94aa-03722a8cb28b
@bind to_show Select(["value", "policy"])

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

# ╔═╡ f923eae6-9d21-4313-86f3-0047dba919f9
let
	gr()
	if to_show == "value"
		V = maximum(Q, dims=4)
		show_v(V)
	elseif to_show == "policy"
		show_actions(π)
	end
end

# ╔═╡ 8d316cc9-f389-4ec9-96cb-0bd217e18760
md"""
Show 3D plot $(@bind show_3d_2 CheckBox())
"""

# ╔═╡ b2a4e78d-d1d2-401d-a318-5a198fe8d9db
let
	if show_3d_2
		plotly()
		V = maximum(Q, dims=4)
		show_v3d(V)
	end
end

# ╔═╡ 51f042b7-f45c-48a3-b2d1-7530e6fcb35a
md"## MC Control without explorin starts"

# ╔═╡ 5232d559-dd4c-4340-92a1-621cc7bafbb4
md"### ε-soft policies"

# ╔═╡ 9dc66787-41c8-4369-9665-4afa17906d68
function episode(agent::BlackjackAgent, env::Blackjack, π::Array; first_action=nothing)
	history = []
	finished = false
	state = agent.state
	i = 0
	while !(finished) && (i < 100)
		i += 1
		if i > 1 || first_action == nothing
			action = BlackjackAction(sample(1:2, ProbabilityWeights(π[state2idx(state)..., :])))
		else
			action = first_action
		end
		reward, new_state, finished = env(state, action)
		push!(history, (s=state, a=action, r=reward))
		state = new_state
	end
	history
end

# ╔═╡ a2ed8727-152f-45c2-8b7e-087bab8639ce
episode(create_agent(), Blackjack(), simple_policy)

# ╔═╡ 234840e5-a678-40ec-b37f-a35f9412fff2
function evaluate_policy(π; max_iter=100, every_visit=false)
	env = Blackjack()
	γ = 1
	V = zeros(21-3, 10, 2)
	Nv = zeros(size(V))
	history = []
	for i = 1:max_iter
		history = episode(create_agent(), env, π)
		visited_states = map(x -> (x.s, x.a), history)
		G = 0
		T = length(history)
		for i = T:-1:1
			s, a, r = history[i]
			G = γ*G + r
			if every_visit || !((s,a) in visited_states[1:i-1])
				idx = state2idx(s)
				try
					Nv[idx...] += 1
					V[idx...] += (G - V[idx...])/Nv[idx...]
				catch
					return history
				end
			end
		end
	end
	V
end

# ╔═╡ 2efbed6f-a676-4e02-ab10-127ce9b45ae0
V = evaluate_policy(simple_policy, max_iter=parse(Int, n_episodes_eval), every_visit=ev_eval);

# ╔═╡ fd00aa9a-d22a-4471-873d-c5deccacf50d
let 
	gr()
	show_v(V)
end

# ╔═╡ 9f3a3b21-7040-47dc-84cc-b4cf04b6a674
let
	if show_3d_1
		plotly()
		show_v3d(V)
	end
end

# ╔═╡ 5bbbce00-22dc-4241-a1d6-fbebd4152bf9
let
	π = zeros(21-3, 10, 2, 2) .+ 0.5
	episode(create_agent(), Blackjack(), π)
end

# ╔═╡ 25d9d69b-c0bb-4456-bf1e-fbd78e0c4992
function mc_eps(max_iter=100)
	# π = zeros(21-3, 10, 2, 2) .+ 0.5
	π = zeros(21-3,10,2,2) .+ 0.05
	π[1:21-5, :, :, 2] .+= 0.9
	π[21-4:21-3, :, :, 1] .+= 0.9
	@assert all(sum(π, dims=4) .== 1)
	env = Blackjack()
	γ = 1
	Q = zeros(21-3, 10, 2, 2)
	Nv = zeros(size(Q))
	history = []
	for i = 1:max_iter
		agent = create_agent()
		history = episode(agent, env, π)
		visited_states = map(x -> (x.s, x.a), history)
		G = 0
		T = length(history)
		for i = T:-1:1
			s, a, r = history[i]
			G = γ*G + r
			if !((s,a) in visited_states[1:i-1])
				idx = (state2idx(s)..., Int(a))
				# Q-values update
				try
					Nv[idx...] += 1
					Q[idx...] += (G - Q[idx...])/Nv[idx...]
				catch
					return history, idx
				end
				# policy update
				idx = idx[1:3]
				a_star = argmax(Q[idx..., :])
				# the policy remains ε-greedy
				π[idx..., :] .= 0.05
				π[idx..., a_star] += 0.9
				@assert all(sum(π, dims=4) .== 1)
			end
		end
	end
	Q, π, Nv
end

# ╔═╡ 21012628-f419-4cd9-a074-47e68559c4b1
md"""
Number of episodes $(@bind n_episodes_mc_eps Select(["10000", "100000","500000", "1000000"]))
"""

# ╔═╡ adde6b40-fd5c-4e73-82bf-8853c48d5a2c
Q_eps, π_eps, nv_eps = mc_eps(parse(Int, n_episodes_mc_eps));

# ╔═╡ c490ac19-58eb-4151-a0c6-dc46e6e8bc93
@bind to_show2 Select(["value", "policy"])

# ╔═╡ 30cb932a-80a3-4716-9cae-024441551ce9
let
	V = maximum(Q_eps, dims=4)
	gr()	
	if to_show2 == "value"
		show_v(V)
	elseif to_show2 == "policy"
		show_actions(π_eps)
	end
end

# ╔═╡ 0a824970-9201-4e42-a26e-5534847aaf2a
md"### Off-policy predictions via Importance Sampling"

# ╔═╡ 5879427c-a30b-47df-8744-68b615547c0e
function off_policy_evaluation(max_iter=100, weighted=true)
	# evaluete naive policy
	π = zeros(21-3,10,2,2)
	π[1:21-5, :, :, 2] .+= 1.
	π[21-4:21-3, :, :, 1] .+= 1.
	@assert all(sum(π, dims=4) .== 1) "Policy should be given as valid action probability distribution"
	env = Blackjack()
	γ = 1
	Q  = zeros(21-3, 10, 2, 2)
	C  = zeros(size(Q))
	Nv = zeros(size(Q))
	history = []
	for i = 1:max_iter
		agent = create_agent()
		# behaviour policy (random)
		b = zeros(21-3, 10, 2, 2) .+ 0.5
		history = episode(agent, env, b)
		visited_states = map(x -> (x.s, x.a), history)
		G = 0
		W = 1
		T = length(history)
		for i = T:-1:1
			s, a, r = history[i]
			G = γ*G + r
			
			idx = (state2idx(s)..., Int(a))
			# Q-values update
			C[idx...] += W
			Nv[idx...] += 1
			if weighted
				Q[idx...] += (G - Q[idx...])*W/C[idx...]
			else
				Q[idx...] += (W*G - Q[idx...]) / Nv[idx...]
			end
			W *= π[idx...] / b[idx...]
			if W == 0
				break
			end
		end
	end
	Q, π, C
end

# ╔═╡ 8ef62951-72c7-4456-b709-be5e5a50ecef
Q_op, _, _ = off_policy_evaluation(100_000);

# ╔═╡ 14f59c06-196a-4f9a-81c6-41790df98724
let
	V = maximum(Q_op, dims=4)
	plotly()
	show_v3d(V)
end

# ╔═╡ 1749beee-a955-4b0e-9152-e5076300ace3
function example_54(weighted=true)
	# evaluete naive policy
	π = zeros(21-3,10,2,2)
	π[1:21-5, :, :, 2] .+= 1.
	π[21-4:21-3, :, :, 1] .+= 1.
	@assert all(sum(π, dims=4) .== 1) "Policy should be given as valid action probability distribution"
	# behaviour policy (random)
	b = zeros(21-3, 10, 2, 2) .+ 0.5
	
	env = Blackjack()
	γ = 1
	Q  = zeros(21-3, 10, 2, 2)
	C  = zeros(size(Q))
	Nv = zeros(size(Q))
	estimates = []
	idx = (state2idx((13, 2, true))..., 2)
	for i = 1:10_000
		agent = create_agent()
		agent.state = (13, 2, true)
		
		history = episode(agent, env, b; first_action=hit)
		_, _, r = history[end]
		G = r
		W = π[idx...] / b[idx...]
		C[idx...] += W
		Nv[idx...] += 1
		# Q-values update
		if weighted
			Q[idx...] += (G - Q[idx...]) *W/C[idx...]
		else
			Q[idx...] += (W*G - Q[idx...]) / Nv[idx...]
		end
		push!(estimates, Q[idx...])
	end
	estimates
end

# ╔═╡ 9e85cd41-6c86-4c9d-bc4c-849d4e3a9d9b
let
	gr()
	p = plot(xscale=:log)
	
	res_w = mean([(example_54() .+ 0.27726).^2 for _ = 1:100]) 
	plot!(p, res_w, label="weighted", colour=:red)
	
	res_o = mean([(example_54(false) .+ 0.27726).^2 for _ = 1:100]) 
	plot!(p, res_o, label="ordinary", colour=:green)
end

# ╔═╡ 6dff459e-7abe-434a-9109-8c14f8dfdc81
# let
# 	gr()
# 	p = plot(xscale=:log)
	
# 	res_w = (mean([example_54() for _ = 1:100])  .+ 0.27726).^2
# 	plot!(p, res_w, label="weighted", colour=:red)
	
# 	res_o =  (mean([example_54(false) for _ = 1:100])  .+ 0.27726).^2
# 	plot!(p, res_o, label="ordinary", colour=:green)
# end

# ╔═╡ 5bfd01a2-5add-49b0-a7e7-7f7d1d5244e7
md"### Off-policy MC control"

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

# ╔═╡ 3e408223-5ca2-4b46-8ff2-44419d9f71a7
md"""
Number of episodes $(@bind n_episodes_mc_opc Select(["10000", "100000","500000", "1000000"]))
"""

# ╔═╡ 0ec098ea-50df-4d87-b273-fe96deab9b2d
Q_opc, π_opc, c_opc = off_policy_control(parse(Int, n_episodes_mc_opc));

# ╔═╡ 1de87fc2-b4bf-41f6-b58b-58e2082b1e2e
@bind to_show3 Select(["value", "policy"])

# ╔═╡ e0ef7ba3-e5bd-40bc-be8d-7b52c0eefcfe
let
	V = maximum(Q_opc, dims=4)
	gr()	
	if to_show3 == "value"
		show_v(V)
	elseif to_show3 == "policy"
		show_actions(π_opc)
	end
end

# ╔═╡ 43c0d6f2-2de6-4be7-bd17-b6b7b7491ae0
md"Looks like even more episodes needed to converge"

# ╔═╡ 70bb38e8-ee11-4659-a3f5-ac6625676540
md"#### Ex. 5.5"

# ╔═╡ b2274f85-a4c7-497c-89b0-e79c0a59bc91
begin
	struct Ex55 <: AbstractEnv
		p::Float64
		Ex55() = new(0.1)

		function (env::Ex55)(s, a)
			if a == 1
				return 0,2
			elseif a == 2
				if rand() < env.p
					return 1,2
				else
					return 0,1
				end
			end
		end
	end
end

# ╔═╡ 38fec5ca-5ce9-48e3-a044-07174ea2288b
function ex55_episode()
	s = 1
	env = Ex55()
	history = []
	while s == 1
		a = Int(rand() > 0.5) + 1
		r, new_s = env(s, a)
		push!(history, (s=s, a=a, r=r))
		s = new_s
	end
	history
end

# ╔═╡ 1dbaf1bf-06a8-4340-ad7a-2258b021f9e3
ex55_episode()

# ╔═╡ 4557dc4a-8d2f-49ef-93ed-c52ee5a3970d
let
	a = zeros(1,2)
	a[1,2] = 1.
	a
end

# ╔═╡ 8b339152-5572-4102-9675-eaead4f4e282
function ex55(max_iter=1000, every_visit=false)
	# evaluete naive policy
	π = zeros(1,2)
	π[1,2] = 1.
	env = Ex55()
	γ = 1
	Q  = zeros(1, 2)
	Nv = zeros(size(Q))
	estimates = []
	Ws = []
	for i = 1:max_iter
		# behaviour policy (random)
		b = zeros(1, 2) .+ 0.5
		history = ex55_episode()
		visited_states = map(x -> x.s, history)
		G = 0.
		W = 1.
		T = length(history)
		for i = T:-1:1
			s, a, r = history[i]
			idx = (s,a)
			G = γ*G + r
			
			
			# if W==0
			# 	break
			# end
			
			
			if every_visit || !(s in visited_states[1:i-1])
				# W *= π[idx...] / b[idx...]
				# if W==0
				# 	break
				# end
				# Q-values update
				Nv[idx...] += 1
				Q[idx...] += (G*W - Q[idx...]) / Nv[idx...]
				# return W, Q[idx...], Nv[idx...]
			end
			# return a, π[idx...], b[idx...]
			W *= π[idx...] / b[idx...]
			if W == 0
				break
			end
			
			
		end
		push!(estimates, Q[1,2])
	end
	Q, π, Nv
	estimates
end

# ╔═╡ 0f5aee9e-b070-4132-b26d-7446fccd9cd9
ex55()

# ╔═╡ 7175fb27-9bc9-451c-b46c-f7c1f6d59243
h = ex55_episode()

# ╔═╡ f0318384-68c7-4c2f-a17d-3472832ad779
let  
	Q = zeros(1,2)
	Nv = zeros(1,2)
	ret = nothing
	visited_states = map(x -> x.s, h)
	
	G = 0.
	W = 1.
	T = length(h)
	for i = T:-1:1
		s, a, r = h[i]
		idx = (s,a)
		G = G + r
		ret = (s,a,r)
		if false || !(s in visited_states[1:i-1])
			# W *= π[idx...] / b[idx...]
# 			# if W==0
# 			# 	break
# 			# end
# 			# Q-values update
			Nv[idx...] += 1
			Q[idx...] += (G*W - Q[idx...]) / Nv[idx...]
# 			# return W, Q[idx...], Nv[idx...]
		end
# 		# return a, π[idx...], b[idx...]
# 		W *= π[idx...] / b[idx...]
# 		if W == 0
# 			break
# 		end

	
	end
	Q, Nv
# 	push!(estimates, Q[1,2])
end

# ╔═╡ 8718cbdb-d6a8-4e2b-a1a1-67216ddaae86
let
	res = [ex55(10_000) for _ = 1:5]
	res
	# plot(res[1], label=:none, ylim=[-1, 10], xscale=:log)
end

# ╔═╡ 10bcbbf3-2c1b-4c1d-87a0-251b4b9ca430


# ╔═╡ b35c60b8-c952-4a7e-bff3-1d296ba25ba3
md"To be continued..."

# ╔═╡ b8af68b9-b2a9-4b0a-aa2b-b14e09243f59
TableOfContents()

# ╔═╡ Cell order:
# ╟─2ac84074-c6ac-11eb-3a87-2b08d5ffa4fc
# ╟─7fbbaff3-c5f7-46d5-aa9d-2cb0f1fddc50
# ╟─893783fe-a5d7-4c5e-ba6d-764e9967a60f
# ╟─8bcc1011-6a64-4e33-b24f-8c26c690fa91
# ╟─788836c3-f7a9-47a1-804e-2ea0127e7b02
# ╟─df8ad838-abe9-4f43-8f70-944f861ff275
# ╠═9b3e4e24-5b53-49e1-87e9-37938da0b83d
# ╟─00504589-f2bd-4496-bd03-740cd5a1c7a1
# ╟─0d899fa7-53d0-42f4-ab26-454c2a9a6079
# ╠═bd4f3e10-d152-470e-aac3-582dd5821085
# ╟─75d91ff3-c9c4-4680-93bf-aaeb1daaf8de
# ╟─546812ec-ea48-423a-8d32-d61083decc44
# ╟─f86c7c26-a34a-4de2-9626-f7e37a7cabbf
# ╟─8fb8cda9-3e6c-4eb8-bf01-69509008af92
# ╠═7ccbae20-160b-45be-814c-ef96b85fc40a
# ╠═fa715a39-d413-4c89-b243-90f28e5b1d80
# ╠═6c9a108b-5ac0-43c8-84bf-2459c9674587
# ╠═4b5de8db-13fc-4f0f-a585-ce2b2de912f2
# ╠═00cf6607-c4c6-49c9-a508-d32f54e7060f
# ╠═c7284971-c9b8-40fe-9541-ebb910dbfab2
# ╟─ced93e92-0ca0-4359-b038-6feb08258dda
# ╟─6a31eaa3-9309-4b31-ac57-0e3204ff8bfa
# ╠═30278c80-38ea-4ee1-9ff8-7b924020dd87
# ╟─3bbe3335-68c5-4efe-9800-e9505b8a63b9
# ╠═a2ed8727-152f-45c2-8b7e-087bab8639ce
# ╠═234840e5-a678-40ec-b37f-a35f9412fff2
# ╟─fe2df29d-3296-4eec-aee9-5d91b086d73b
# ╠═2efbed6f-a676-4e02-ab10-127ce9b45ae0
# ╟─fd00aa9a-d22a-4471-873d-c5deccacf50d
# ╟─e0f41a91-2cfc-432d-b977-e4f8e73047de
# ╟─9f3a3b21-7040-47dc-84cc-b4cf04b6a674
# ╟─f4472d2e-bcaf-4736-b3e4-9c862acdcb89
# ╟─67053afd-1686-406b-acf1-790bdfa28374
# ╟─dcd5c9db-6663-4cc9-bdbd-b107131e763a
# ╟─b70bcf37-2b1e-4621-a378-d9ab72de50c4
# ╟─d78418eb-07fe-4f9b-a073-c79cccb0bbb9
# ╠═b972b736-e97d-496b-9913-466dc9ee4664
# ╟─89aa9ab7-7afc-4569-9f71-9881f1666bf3
# ╟─f08c8021-11e1-4443-8e86-0f2a664cecf1
# ╟─40d598a9-3e7d-4929-ba1f-82fd594255aa
# ╠═614e7c44-1173-4dda-854b-c6915a453249
# ╟─ca28c10a-1495-4704-b521-e11653e5848a
# ╟─e239eae0-4468-4860-80cd-855dc546878f
# ╠═6636140b-cbb1-4a05-b724-50ed1188efe5
# ╟─4ee8e017-3c54-438e-94aa-03722a8cb28b
# ╠═f923eae6-9d21-4313-86f3-0047dba919f9
# ╟─6867a8a6-ea82-40c3-84c1-d37bdea8c79c
# ╟─8d316cc9-f389-4ec9-96cb-0bd217e18760
# ╟─b2a4e78d-d1d2-401d-a318-5a198fe8d9db
# ╟─51f042b7-f45c-48a3-b2d1-7530e6fcb35a
# ╟─5232d559-dd4c-4340-92a1-621cc7bafbb4
# ╠═9dc66787-41c8-4369-9665-4afa17906d68
# ╟─5bbbce00-22dc-4241-a1d6-fbebd4152bf9
# ╠═25d9d69b-c0bb-4456-bf1e-fbd78e0c4992
# ╟─21012628-f419-4cd9-a074-47e68559c4b1
# ╠═adde6b40-fd5c-4e73-82bf-8853c48d5a2c
# ╟─c490ac19-58eb-4151-a0c6-dc46e6e8bc93
# ╠═30cb932a-80a3-4716-9cae-024441551ce9
# ╟─0a824970-9201-4e42-a26e-5534847aaf2a
# ╠═5879427c-a30b-47df-8744-68b615547c0e
# ╠═8ef62951-72c7-4456-b709-be5e5a50ecef
# ╟─14f59c06-196a-4f9a-81c6-41790df98724
# ╠═1749beee-a955-4b0e-9152-e5076300ace3
# ╠═9e85cd41-6c86-4c9d-bc4c-849d4e3a9d9b
# ╟─6dff459e-7abe-434a-9109-8c14f8dfdc81
# ╟─5bfd01a2-5add-49b0-a7e7-7f7d1d5244e7
# ╟─2390d0cf-6cad-4059-953f-e6b04479ca87
# ╟─3e408223-5ca2-4b46-8ff2-44419d9f71a7
# ╠═0ec098ea-50df-4d87-b273-fe96deab9b2d
# ╟─1de87fc2-b4bf-41f6-b58b-58e2082b1e2e
# ╠═e0ef7ba3-e5bd-40bc-be8d-7b52c0eefcfe
# ╟─43c0d6f2-2de6-4be7-bd17-b6b7b7491ae0
# ╟─70bb38e8-ee11-4659-a3f5-ac6625676540
# ╠═b2274f85-a4c7-497c-89b0-e79c0a59bc91
# ╠═38fec5ca-5ce9-48e3-a044-07174ea2288b
# ╠═1dbaf1bf-06a8-4340-ad7a-2258b021f9e3
# ╠═4557dc4a-8d2f-49ef-93ed-c52ee5a3970d
# ╠═8b339152-5572-4102-9675-eaead4f4e282
# ╠═0f5aee9e-b070-4132-b26d-7446fccd9cd9
# ╠═7175fb27-9bc9-451c-b46c-f7c1f6d59243
# ╠═f0318384-68c7-4c2f-a17d-3472832ad779
# ╠═8718cbdb-d6a8-4e2b-a1a1-67216ddaae86
# ╠═10bcbbf3-2c1b-4c1d-87a0-251b4b9ca430
# ╟─b35c60b8-c952-4a7e-bff3-1d296ba25ba3
# ╠═b8af68b9-b2a9-4b0a-aa2b-b14e09243f59
