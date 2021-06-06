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

# ╔═╡ 9b3e4e24-5b53-49e1-87e9-37938da0b83d
@enum BlackjackAction stick=1 hit=2

# ╔═╡ 546812ec-ea48-423a-8d32-d61083decc44
check_sum(x) = x > 21 ? 0 : x

# ╔═╡ f86c7c26-a34a-4de2-9626-f7e37a7cabbf
check_sum(21), check_sum(15), check_sum(22)

# ╔═╡ 00504589-f2bd-4496-bd03-740cd5a1c7a1
ACE = 1

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
			if player_sum == dealer_sum #draw
				r = 0
			elseif player_sum > dealer_sum
				r = 1
			else
				r = -1
			end
			return r, final_state, true
		end
	end
end

# ╔═╡ 6c9a108b-5ac0-43c8-84bf-2459c9674587
mutable struct BlackjackAgent <: AbstractAgent
	v::Array
	q::Array
	state::Tuple
	actions
end

# ╔═╡ fa715a39-d413-4c89-b243-90f28e5b1d80
let
	env = Blackjack()
	env((19, 1, false), hit)
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
	while !(finished) && (i < 10)
		i += 1
		action = π(state)
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

# ╔═╡ e0f41a91-2cfc-432d-b977-e4f8e73047de
md"""
Show 3D plot $(@bind show_3d_1 CheckBox())
"""

# ╔═╡ 92ad6a93-4426-4fa5-9535-6e3ca4b68564
@bind ana Select(["No ace", "With ace"])

# ╔═╡ 9f3a3b21-7040-47dc-84cc-b4cf04b6a674
let
	if show_3d_1
		plotly()
		id = Int(ana == "With ace")+1
		plot(1:10, 12:21, V[9:end, :, id], st=:surface)
	end
end

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

# ╔═╡ fd00aa9a-d22a-4471-873d-c5deccacf50d
let 
	gr()
	show_v(V)
end

# ╔═╡ e0ee067b-901a-44d7-904c-4857643fb8ab
let 
	if long_run
		V = evaluate_policy(simple_policy, max_iter=500_000)
		gr()
		show_v(V)
	end
end

# ╔═╡ dcd5c9db-6663-4cc9-bdbd-b107131e763a
md"# Action values estimation"

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
		agent.state = (rand(4:21), rand(1:10), rand() > 0.5)
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
				@assert all(sum(π, dims=4) .== 1)
			end
		end
	end
	Q, π, Nv
end

# ╔═╡ e239eae0-4468-4860-80cd-855dc546878f
md"""
Number of episodes $(@bind n_episodes_mcse Select(["10000", "100000","500000"]))
"""

# ╔═╡ 6636140b-cbb1-4a05-b724-50ed1188efe5
Q, π, Nv = mces(parse(Int, n_episodes_mcse));

# ╔═╡ 4ee8e017-3c54-438e-94aa-03722a8cb28b
@bind to_show Select(["value", "policy"])

# ╔═╡ f923eae6-9d21-4313-86f3-0047dba919f9
let
	gr()
	if to_show == "value"
		V = maximum(Q, dims=4)
		show_v(V)
	elseif to_show == "policy"
		no_ace_act = (π[8:end, :, 1, 1] .> .5)
		left = heatmap(no_ace_act, legend=:none, title="No Ace", yticks=(1:11, 11:21))
		ace_act = (π[8:end, :, 2, 1] .> .5)
		right = heatmap(ace_act, legend=:none, title="With Ace", yticks=(1:11, 11:21))
		plot(left, right)
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

# ╔═╡ b35c60b8-c952-4a7e-bff3-1d296ba25ba3
md"To be continued..."

# ╔═╡ Cell order:
# ╟─2ac84074-c6ac-11eb-3a87-2b08d5ffa4fc
# ╟─7fbbaff3-c5f7-46d5-aa9d-2cb0f1fddc50
# ╟─893783fe-a5d7-4c5e-ba6d-764e9967a60f
# ╟─8bcc1011-6a64-4e33-b24f-8c26c690fa91
# ╟─788836c3-f7a9-47a1-804e-2ea0127e7b02
# ╠═9b3e4e24-5b53-49e1-87e9-37938da0b83d
# ╠═bd4f3e10-d152-470e-aac3-582dd5821085
# ╠═75d91ff3-c9c4-4680-93bf-aaeb1daaf8de
# ╟─546812ec-ea48-423a-8d32-d61083decc44
# ╟─f86c7c26-a34a-4de2-9626-f7e37a7cabbf
# ╠═7ccbae20-160b-45be-814c-ef96b85fc40a
# ╟─00504589-f2bd-4496-bd03-740cd5a1c7a1
# ╠═6c9a108b-5ac0-43c8-84bf-2459c9674587
# ╠═fa715a39-d413-4c89-b243-90f28e5b1d80
# ╠═4b5de8db-13fc-4f0f-a585-ce2b2de912f2
# ╠═00cf6607-c4c6-49c9-a508-d32f54e7060f
# ╠═c7284971-c9b8-40fe-9541-ebb910dbfab2
# ╟─ced93e92-0ca0-4359-b038-6feb08258dda
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
# ╟─b972b736-e97d-496b-9913-466dc9ee4664
# ╟─89aa9ab7-7afc-4569-9f71-9881f1666bf3
# ╟─614e7c44-1173-4dda-854b-c6915a453249
# ╟─e239eae0-4468-4860-80cd-855dc546878f
# ╠═6636140b-cbb1-4a05-b724-50ed1188efe5
# ╟─4ee8e017-3c54-438e-94aa-03722a8cb28b
# ╠═f923eae6-9d21-4313-86f3-0047dba919f9
# ╟─8d316cc9-f389-4ec9-96cb-0bd217e18760
# ╟─f967644d-66f4-46bd-8c91-0ade8a233413
# ╠═b2a4e78d-d1d2-401d-a318-5a198fe8d9db
# ╟─b35c60b8-c952-4a7e-bff3-1d296ba25ba3
