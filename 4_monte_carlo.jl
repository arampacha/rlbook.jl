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
md"# Value function estimation"

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

# ╔═╡ e0f41a91-2cfc-432d-b977-e4f8e73047de
md"""
Show 3D plot $(@bind show_3d_1 CheckBox())
"""

# ╔═╡ 92ad6a93-4426-4fa5-9535-6e3ca4b68564
@bind ana Select(["No ace", "With ace"])

# ╔═╡ 365da46c-97d2-4214-bc76-61248683a74f
md"""
Show 500000 episodes $(@bind long_run CheckBox())
"""

# ╔═╡ f4472d2e-bcaf-4736-b3e4-9c862acdcb89
function show_v(V::Array; legend=:none)
	left = heatmap(V[9:end,:,1], title="No usable ace", ylabel="Player sum", legend=:none, yticks=(1:10, 12:21))
	
	right= heatmap(V[9:end,:,2], title="Usable ace", legend=legend, yticks=:none)
	plot(left, right, xlabel="Dealer card")
end

# ╔═╡ dcd5c9db-6663-4cc9-bdbd-b107131e763a
md"# Action values estimation"

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
		visited_states = map(x -> x.s, history)
		G = 0
		T = length(history)
		for i = T:-1:1
			s, a, r = history[i]
			G += γ*r
			if !(s in visited_states[1:i-1])
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

# ╔═╡ f967644d-66f4-46bd-8c91-0ade8a233413
@bind ana2 Select(["No ace", "With ace"])

# ╔═╡ b2a4e78d-d1d2-401d-a318-5a198fe8d9db
let
	if show_3d_2
		plotly()
		V = maximum(Q, dims=4)
		id = Int(ana2 == "With ace")+1
		plot(1:10, 11:21, V[8:end, :, id], st=:surface)
	end
end

# ╔═╡ 51f042b7-f45c-48a3-b2d1-7530e6fcb35a
md"## MC Control without explorin starts"

# ╔═╡ 9dc66787-41c8-4369-9665-4afa17906d68
function episode(agent::BlackjackAgent, env::Blackjack, π::Array)
	history = []
	finished = false
	state = agent.state
	i = 0
	while !(finished) && (i < 100)
		i += 1
		action = BlackjackAction(sample(1:2, ProbabilityWeights(π[state2idx(state)..., :])))
		reward, new_state, finished = env(state, action)
		push!(history, (s=state, a=action, r=reward))
		state = new_state
	end
	history
end

# ╔═╡ a2ed8727-152f-45c2-8b7e-087bab8639ce
episode(create_agent(), Blackjack(), simple_policy)

# ╔═╡ 234840e5-a678-40ec-b37f-a35f9412fff2
function evaluate_policy(π; max_iter=100)
	env = Blackjack()
	γ = 1
	V = zeros(21-3, 10, 2)
	Nv = zeros(size(V))
	history = []
	for i = 1:max_iter
		history = episode(create_agent(), env, π)
		visited_states = map(x -> x.s, history)
		G = 0
		T = length(history)
		for i = T:-1:1
			s, a, r = history[i]
			G += γ*r
			if !(s in visited_states[1:i-1])
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
V = evaluate_policy(simple_policy, max_iter=10_000);

# ╔═╡ fd00aa9a-d22a-4471-873d-c5deccacf50d
let 
	gr()
	show_v(V)
end

# ╔═╡ 9f3a3b21-7040-47dc-84cc-b4cf04b6a674
let
	if show_3d_1
		plotly()
		id = Int(ana == "With ace")+1
		plot(1:10, 12:21, V[9:end, :, id], st=:surface)
	end
end

# ╔═╡ e0ee067b-901a-44d7-904c-4857643fb8ab
let 
	if long_run
		V = evaluate_policy(simple_policy, max_iter=500_000)
		gr()
		show_v(V)
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
		history = episode_es(agent, env, π)
		visited_states = map(x -> x.s, history)
		G = 0
		T = length(history)
		for i = T:-1:1
			s, a, r = history[i]
			G += γ*r
			if !(s in visited_states[1:i-1])
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
md"## Off-policy predictions via Importance Sampling"

# ╔═╡ b35c60b8-c952-4a7e-bff3-1d296ba25ba3
md"To be continued..."

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
# ╠═3bbe3335-68c5-4efe-9800-e9505b8a63b9
# ╠═a2ed8727-152f-45c2-8b7e-087bab8639ce
# ╠═234840e5-a678-40ec-b37f-a35f9412fff2
# ╠═2efbed6f-a676-4e02-ab10-127ce9b45ae0
# ╠═fd00aa9a-d22a-4471-873d-c5deccacf50d
# ╟─e0f41a91-2cfc-432d-b977-e4f8e73047de
# ╟─92ad6a93-4426-4fa5-9535-6e3ca4b68564
# ╟─9f3a3b21-7040-47dc-84cc-b4cf04b6a674
# ╟─365da46c-97d2-4214-bc76-61248683a74f
# ╟─e0ee067b-901a-44d7-904c-4857643fb8ab
# ╟─f4472d2e-bcaf-4736-b3e4-9c862acdcb89
# ╟─dcd5c9db-6663-4cc9-bdbd-b107131e763a
# ╟─b70bcf37-2b1e-4621-a378-d9ab72de50c4
# ╟─d78418eb-07fe-4f9b-a073-c79cccb0bbb9
# ╠═b972b736-e97d-496b-9913-466dc9ee4664
# ╟─89aa9ab7-7afc-4569-9f71-9881f1666bf3
# ╟─f08c8021-11e1-4443-8e86-0f2a664cecf1
# ╠═614e7c44-1173-4dda-854b-c6915a453249
# ╟─ca28c10a-1495-4704-b521-e11653e5848a
# ╟─e239eae0-4468-4860-80cd-855dc546878f
# ╠═6636140b-cbb1-4a05-b724-50ed1188efe5
# ╟─4ee8e017-3c54-438e-94aa-03722a8cb28b
# ╠═f923eae6-9d21-4313-86f3-0047dba919f9
# ╠═6867a8a6-ea82-40c3-84c1-d37bdea8c79c
# ╟─8d316cc9-f389-4ec9-96cb-0bd217e18760
# ╟─f967644d-66f4-46bd-8c91-0ade8a233413
# ╟─b2a4e78d-d1d2-401d-a318-5a198fe8d9db
# ╟─51f042b7-f45c-48a3-b2d1-7530e6fcb35a
# ╟─9dc66787-41c8-4369-9665-4afa17906d68
# ╟─5bbbce00-22dc-4241-a1d6-fbebd4152bf9
# ╠═25d9d69b-c0bb-4456-bf1e-fbd78e0c4992
# ╟─21012628-f419-4cd9-a074-47e68559c4b1
# ╠═adde6b40-fd5c-4e73-82bf-8853c48d5a2c
# ╟─c490ac19-58eb-4151-a0c6-dc46e6e8bc93
# ╟─30cb932a-80a3-4716-9cae-024441551ce9
# ╟─0a824970-9201-4e42-a26e-5534847aaf2a
# ╟─b35c60b8-c952-4a7e-bff3-1d296ba25ba3
