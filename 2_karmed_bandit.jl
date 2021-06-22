### A Pluto.jl notebook ###
# v0.14.4

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

# ╔═╡ 48d4dd64-a47d-11eb-14d1-6161cbc1413a
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

# ╔═╡ 3f4a1865-6b69-4485-a285-b4ebee56c015
q = randn(10)

# ╔═╡ a26cf864-d5a4-4d9e-91ad-fbfe4e385d87
function get_reward(q, a; σ=1.)
	return randn() * σ + q[a]
end

# ╔═╡ 351277e6-6777-417b-96b8-023b4ef5707d
get_reward(q, 1)

# ╔═╡ 15f1567e-c69a-4ff6-8276-114628be3018
function experiment(N=100, T=1000, trial=trial, q_mean=0.; trial_kwargs...)
	opt_pcts = []
	average_rewards = []
	for i = 1:N
		q = randn(10) .+ q_mean
		res = trial(q, T; trial_kwargs...)
		push!(opt_pcts, res.opt_pct)
		push!(average_rewards, res.avg_reward)
	end
	return opt_pcts, average_rewards
end

# ╔═╡ baa33a22-49e9-4469-b707-585a786d8124
@bind show0 CheckBox()

# ╔═╡ c311c9d2-cfd1-41b9-a47c-b5d7620e5937
@bind eps1 Slider(0.:0.01:0.2, show_value=true)

# ╔═╡ e084c765-f5e4-4bbb-b6dc-db72207bb6a1
let 
	if show0
		opt_pcts, average_rewards = experiment(ε=eps1)
		p1 = plot(title="Average reward", legend=:bottomright)
		for r = average_rewards
			plot!(p1, r, label=nothing, opacity=0.3)
		end
		plot!(p1, mean(average_rewards), lw=3, label="mean", colour="black")

		p2 = plot(title="Optimal action %", legend=:bottomright)
		for pct = opt_pcts
			plot!(p2, pct, label=nothing, opacity=0.3)
		end
		plot!(p2, mean(opt_pcts), lw=3, label="mean", colour="black")
		plot(p1, p2, layout=(2, 1))
	
	end
		
end

# ╔═╡ 42797c0e-6785-4868-a78a-5f492351476d
@bind show_std CheckBox()

# ╔═╡ 7fb0c7d1-cd78-4996-aca9-c16dd7989d2a
@bind show1 CheckBox()

# ╔═╡ 09e5bc8f-ed94-4d16-b722-27c3d6aaf56f
@bind upd1 Button("Rerun")

# ╔═╡ 770c1609-10aa-4720-b2d0-b89e64d18a9c
let 
	if show1
		upd1
		
		N = 200
		p1, p2 = plot(title="Average reward", legend=:bottomright), plot(title="Optimal action %", legend=:bottomright)
		for eps = [0., .01, .1]
			opt_pcts, average_rewards = experiment(N, ε=eps)
			if eps == 0.
				l = "ε = 0 (greedy)"
			else
				l = "ε = $eps"
			end
			plot!(p1, mean(average_rewards), lw=2, label=l)
			plot!(p2, mean(opt_pcts), lw=2, label=l)
		end
		plot(p1, p2, layout=(2, 1))
	end
end

# ╔═╡ 81afbb4c-84e2-4c40-9983-78915da37df1
@bind show2 CheckBox()

# ╔═╡ b01c2fe5-a4a8-43ea-9c36-4c5b785ea3a0
@bind show_unbiased CheckBox()

# ╔═╡ 6a2bcf9e-f4d4-4e7e-9686-16fe8055c03d
@bind show_opt CheckBox()

# ╔═╡ 8a35ab08-a496-4599-bdd9-4f0a03b8bd01
@bind upd_opt Button("Rerun")

# ╔═╡ 2889dd86-c0fc-4cee-9a2d-6b829cb746e1
let 
	if show_opt
		upd_opt
		
		N = 200
		p1, p2 = plot(title="Average reward", legend=:bottomright), plot(title="Optimal action %", legend=:bottomright)
		for eps = [0., .1], q_init = [5., 0.]
			opt_pcts, average_rewards = experiment(N; ε=eps, q_init=q_init)
			if eps == 0.
				l = "ε=0 (greedy), q₁=$q_init"
			else
				l = "ε=$eps, q₁=$q_init"
			end
			plot!(p1, mean(average_rewards), lw=2, label=l)
			plot!(p2, mean(opt_pcts), lw=2, label=l)
		end
		plot(p1, p2, layout=(2, 1))
	end
end

# ╔═╡ 462bb53b-a215-4cc7-ab97-a5b8260c423e
@bind ta  Slider(5:1000, default=100, show_value=true)

# ╔═╡ 638c891b-b223-4813-a997-318ae75355a1
let

	na = 1:ta 
	p = plot()
	plot!(p, sqrt.(log(ta)./(na .+ 1e-5)), xlabel="action count", ylabel="potential")
	hline!(p, [log(ta)], label="log t")
end

# ╔═╡ d8899170-07d9-49fd-9d03-3c9ceaca3a9d


# ╔═╡ 6eaf46db-dd07-4a47-96b1-6a07c632a79a
@bind pa Slider(0.:0.01:1., show_value=true, default=0.1)

# ╔═╡ e4012f10-c56a-48e1-adfb-07cfc07c4d87
@bind upd_pot Button("Refresh")

# ╔═╡ 01d8ac29-c38e-4641-a6c8-8a05add94890
let
	upd_pot
	t = 1:100
	c = 1
	na = accumulate(+, [Int(rand()<pa) for _=t])
	
	plot(sqrt.(log.(t)./(na .+ 1e-5)), xlabel="t", ylabel="potential")
	plot!(twinx(), na, colour="red", label="action count")
end

# ╔═╡ 8b4765e8-e2dd-4d0f-b86e-9d5afb6e77d2
@bind show_ubc CheckBox()

# ╔═╡ 6cc2ca49-8d5d-411a-82c3-01f11c6743bd
@bind upd_ubc Button("Rerun")

# ╔═╡ 09b803c6-f06d-4813-bade-c0f93321daa3
@bind show_grad CheckBox()

# ╔═╡ 9bae2290-32c3-4b75-91b1-f589e421bcdb
@bind upd_grad Button("Rerun")

# ╔═╡ f720ba6c-c199-48e7-bebb-aaeae7654536
@bind show_ps CheckBox()

# ╔═╡ 655ee62d-cd27-47c8-b3df-622aa539b98f
function softmax(x::Vector)
	return exp.(x .- maximum(x)) ./ sum(exp.(x .- maximum(x)))
end

# ╔═╡ 92a924e6-8189-40e6-9cdf-3fad210267c8
begin
	function randint(low, high)
		return rand(low:high)
	end
	function randint(high)
		return randint(1, high)
	end
end

# ╔═╡ 93b4b77a-8055-4d09-9024-a81efcb7d7f9
randint(10)

# ╔═╡ d4023ff2-cfa3-49a8-9a1c-5eaa3f9b7a0e
function getindex(collection, idx, default)
	try
		collection[idx]
	catch
		default
	end
end

# ╔═╡ 8cee4777-f9b7-4584-a498-3e08903ca9a3
md"Notice how greedy strategy often gets stuck on with local optima and never explores further"

# ╔═╡ cf6cb7ee-84d8-48fa-9573-07dfacee5962
md"Show plot"

# ╔═╡ 0b68c4f7-ba82-4e86-b4ad-0d52e8d510a4
md"ε"

# ╔═╡ 033f1958-f54a-4398-a1fd-a7d1c2dd6da5
md"Plot average rewards with std"

# ╔═╡ 49f80719-524e-426c-b0f5-b8c8e2463692
md"## ε-greedy strategies"

# ╔═╡ 4910f9fb-3a75-40ce-bb1f-e5fcfe3b3da4
md"Show plot"

# ╔═╡ 029983e9-40fc-45a6-b4ef-4af0e5d96327
md"## Exercise 2.5 - non-stationary"

# ╔═╡ eb826b48-6fc2-468e-8bc7-d96861307b71
md"Show non-stationary Q"

# ╔═╡ 34b6a4d6-b292-4dda-9eb6-1c6371773f43
md"Show unbiased EMA"

# ╔═╡ e9831e70-5052-494e-8682-a88a37fbbd61
md"## Optimistic initial q estimates"

# ╔═╡ b2df3274-7981-4fb6-82fa-d58354e02b60
md"Show optimistic initial Q"

# ╔═╡ 4b672fe5-fb95-4d00-b8d0-1621b8d62ec9
md"## Upper confidence bound"

# ╔═╡ 14791977-1866-46f1-8ff8-6a107559f822
md"The plot below shows the action 'potential' as function of action count for different timestep"

# ╔═╡ 871dfe49-59b5-40d5-9067-b41374bb949a
md"t"

# ╔═╡ f4bbe00c-b033-415c-9c8e-e877270adc8f
md"Action probability"

# ╔═╡ 939cb982-427e-4c4f-91fa-dc60acb97a01
md"Show UBC"

# ╔═╡ f35e791f-9abf-4c70-ba2b-67139332b98f
md"## Gradient bandit"

# ╔═╡ bf7d348b-c059-4cc2-9fc6-560f64751104
md"## Parameter study"

# ╔═╡ b88c8d7e-20b2-4ab6-9a4c-492486cac649
md"Plot parameter study"

# ╔═╡ 8bd817d4-9cce-44ee-8595-567119a4cf43
md"## utils"

# ╔═╡ c2417fcc-016d-4be4-a1f0-b32a85b6a5a1
getindex([1,2,3], -1, 0)

# ╔═╡ ef167d48-dd50-4bb3-9ada-d53ec0b9e3a6
function accumulate_mean!(collection, val)
	last = try
		collection[end]
	catch
		0
	end
	push!(collection, last + (val-last)/(length(collection)+1))
end

# ╔═╡ c08545e2-5218-4489-8975-5c3f86c66345
function trial(q, T=1000; ε=0., q_init=0., walk_std=0.)
	k = length(q)
	a_optimal_pct = []
	avg_reward = []
	q_est = zeros(k) .+ q_init
	a_counts = zeros(k)
	for t = 1:T
		
		a_optimal = argmax(q)
		if rand() > ε
			a = argmax(q_est)
		else
			a = randint(k)
		end
		accumulate_mean!(a_optimal_pct, Int(a == a_optimal))
		
		r = get_reward(q, a)
		accumulate_mean!(avg_reward, r)
		
		a_counts[a] += 1
		q_est[a] += (r-q_est[a]) / a_counts[a]
		
		if walk_std ≠ 0.
			q = q + randn(k)*walk_std
		end
	end
	return (q_est=q_est, avg_reward=avg_reward, opt_pct=a_optimal_pct)
end

# ╔═╡ 86e725ba-ff06-4aa4-bd85-0c1b6f30065b
let
	res = trial(q, 1000)
	
	# plot(res.opt_pct, label=nothing)
	res.q_est, q
end

# ╔═╡ 018c6afe-6a60-4aee-a4da-8612955b64d2
function ema_trial(q, T=1000; ε=0., q_init=0., walk_std=0., α=0.1, unbiased=false)
	k = length(q)
	a_optimal_pct = []
	avg_reward = []
	similar
	q_est = zeros(k)
	a_counts = zeros(k)
	
	ō = 0
	
	for t = 1:T
		a_optimal = argmax(q)
		if ε ≠ 0. && rand() > ε
			a = argmax(q_est)
		else
			a = randint(k)
		end
		accumulate_mean!(a_optimal_pct, Int(a == a_optimal))
		
		r = get_reward(q, a)
		accumulate_mean!(avg_reward, r)
		
		a_counts[a] += 1
		
		if unbiased
			ō += α * (1 - ō)
			β = α / ō
			q_est[a] += α*(r-q_est[a])
		else
			q_est[a] += α*(r-q_est[a])
		end
		if walk_std ≠ 0.
			q = q + randn(k)*walk_std
		end
	end
	return (q_est=q_est, avg_reward=avg_reward, opt_pct=a_optimal_pct)
end

# ╔═╡ 0bbdea00-060b-4ce0-a3e2-d3dc5d141649
let
	q_est, opt_pct, avgr = ema_trial(q, 1000)
	plot(opt_pct, label=nothing)
end

# ╔═╡ 731acaab-e910-40bf-915e-7868d742abd1
unbiased_ema_trial = (args...; kwargs...) -> ema_trial(args...; kwargs...,  unbiased=true)

# ╔═╡ 3aabf6b6-f22a-47ee-8729-4a7840c0bebc
let 
	if show2
		show_unbiased
		
		N = 200
		t = 10000
		p1, p2 = plot(title="Average reward", legend=:bottomright), plot(title="Optimal action %", legend=:bottomright)
		opt_pcts, average_rewards = experiment(N, t, trial; ε=0.1, walk_std=0.01)
		plot!(p1, mean(average_rewards), lw=2, label="sample avg")
		plot!(p2, mean(opt_pcts), lw=2, label="sample average")

		opt_pcts, average_rewards = experiment(N, t, ema_trial; ε=0.1, walk_std=0.01)
		plot!(p1, mean(average_rewards), lw=2, label="ema")
		plot!(p2, mean(opt_pcts), lw=2, label="ema")

		if show_unbiased

			opt_pcts, average_rewards = experiment(N, t, unbiased_ema_trial; ε=0.1, walk_std=0.01)
			plot!(p1, mean(average_rewards), lw=2, label="unbiased ema")
			plot!(p2, mean(opt_pcts), lw=2, label="unbiased ema")
		end

		plot(p1, p2, layout=(2, 1))
	end
end

# ╔═╡ d0e247a0-2c40-474c-b359-4d2fc045e882
function ucb_trial(q, T=1000; ε=0., q_init=0., walk_std=0., c=2.)
	k = length(q)
	a_optimal_pct = []
	avg_reward = []
	q_est = zeros(k) .+ q_init
	a_counts = zeros(k)
	for t = 1:T
		
		a_optimal = argmax(q)
		
		potentials = sqrt.(log(t)./(a_counts .+ 1e-5))
		
		a = argmax(q_est + c * potentials)
		accumulate_mean!(a_optimal_pct, Int(a == a_optimal))
		
		r = get_reward(q, a)
		accumulate_mean!(avg_reward, r)
		
		a_counts[a] += 1
		q_est[a] += (r-q_est[a])/a_counts[a]
		
		if walk_std ≠ 0.
			q = q + randn(k)*walk_std
		end
	end
	return (q_est=q_est, avg_reward=avg_reward, opt_pct=a_optimal_pct)
end

# ╔═╡ 286c4c5d-bac8-4a46-9bbb-f1cbb75c9605
let 
	if show_ubc
		upd_ubc
		
		N = 200
		T = 1000
		p1, p2 = plot(title="Average reward", legend=:bottomright), plot(title="Optimal action %", legend=:bottomright)

		opt_pcts, average_rewards = experiment(N, ε=0.1)
		plot!(p1, mean(average_rewards), lw=2, label="ε = 0.1")
		plot!(p2, mean(opt_pcts), lw=2, label="ε = 0.1")
		
		c = 1.
		opt_pcts, average_rewards = experiment(N, T, ucb_trial; c=c)
		plot!(p1, mean(average_rewards), lw=2, label="UCB, c=$c")
		plot!(p2, mean(opt_pcts), lw=2, label="UCB, c=$c")

		plot(p1, p2, layout=(2, 1))
	end
end

# ╔═╡ aa4d7178-5058-4caa-a5a6-1b6c96ede324
function grad_trial(q, T=1000; q_init=0., walk_std=0., α=0.1, baseline=true)
	k = length(q)
	a_optimal_pct = []
	avg_reward = []
	
	q_est = zeros(k) .+ q_init
	a_counts = zeros(k) ./ k
	h = zeros(k)
	
	for t = 1:T
		
		a_optimal = argmax(q)
		π = softmax(h)
		a = sample(1:k, ProbabilityWeights(π, 1))
		accumulate_mean!(a_optimal_pct, Int(a == a_optimal))
		
		r = get_reward(q, a)
		accumulate_mean!(avg_reward, r)
		
		if baseline
			if t == 1
				r̄ = r
			else
				r̄ = avg_reward[t-1] # in book it's mentioned that current reward was included in update for their simulations; this can be achieved by using avg_reward[t]
			end
		else
			r̄ = 0
		end
		# update action probabilities
		for (idx, val) in enumerate(h)
			if idx == a
				h[idx] += α * (r - r̄) * (1 - π[idx])
			else
				h[idx] += α * (r - r̄) * π[idx]
			end
		end
		
		if walk_std ≠ 0.
			q = q + randn(k)*walk_std
		end
	end
	return (q_est=nothing, avg_reward=avg_reward, opt_pct=a_optimal_pct, h=h, q=q)
end

# ╔═╡ 609c08da-2250-4788-ac7d-fe381a5d34b6
let
	res = grad_trial(q, 1000, baseline=true)
	
	res.h, res.q
	# plot(res.opt_pct, label=nothing)
end

# ╔═╡ adffebe6-685a-46ab-93dd-5699f754ce99
let 
	if show_grad
		upd_grad
		
		N = 200
		T = 1000
		q_mean = 4.
		p1, p2 = plot(title="Average reward", legend=:bottomright), plot(title="Optimal action %", legend=:bottomright)

		α = 0.1
		opt_pcts, average_rewards = experiment(N, T, grad_trial, q_mean; α=α)
		plot!(p1, mean(average_rewards), lw=2, label="α=$α")
		plot!(p2, mean(opt_pcts), lw=2, label="α=$α")
		
		opt_pcts, average_rewards = experiment(N, T, grad_trial, q_mean; α=α, baseline=false)
		plot!(p1, mean(average_rewards), lw=2, label="α=$α, w/o baseline")
		plot!(p2, mean(opt_pcts), lw=2, label="α=$α, w/o baseline")

		
		α = 0.4
		opt_pcts, average_rewards = experiment(N, T, grad_trial, q_mean; α=α)
		plot!(p1, mean(average_rewards), lw=2, label="α=$α")
		plot!(p2, mean(opt_pcts), lw=2, label="α=$α")
		
		opt_pcts, average_rewards = experiment(N, T, grad_trial, q_mean; α=α, baseline=false)
		plot!(p1, mean(average_rewards), lw=2, label="α=$α, w/o baseline")
		plot!(p2, mean(opt_pcts), lw=2, label="α=$α, w/o baseline")
		
		
		plot(p1, p2, layout=(2, 1))
		
	end
end

# ╔═╡ 2faf1f5e-9534-4913-ba7f-72ee9a8150ff
let
	if show_ps
		
		N = 100
		T = 1000
		p = plot(xscale=:log, legend=:bottomright)

		# epsilon greedy
		εs = [2. ^i for i = -7:-2]
		ε_avg = []
		for ε in εs
			_, avgr = experiment(N, T, trial; ε=ε)
			push!(ε_avg, mean(avgr)[end])
		end
		plot!(p, εs, ε_avg, label="ε-greedy")


		# optimistic greedy
		q_init = [2. ^i for i = -2:2]
		q_avg = []
		for q0 in q_init
			_, avgr = experiment(N, T, trial; q_init=q0)
			push!(q_avg, mean(avgr)[end])
		end
		plot!(p, q_init, q_avg, label="optimistic")

		# UCB
		cs = [2. ^i for i = -4:2]
		ucb_avg = []
		for c in cs
			_, avgr = experiment(N, T, ucb_trial; c=c)
			push!(ucb_avg, mean(avgr)[end])
		end
		plot!(p, cs, ucb_avg, label="UCB")

		# gradient bandit
		αs = [2. ^i for i = -5:1]
		grad_avg = []
		for α in αs
			_, avgr = experiment(N, T, grad_trial; α=α)
			push!(grad_avg, mean(avgr)[end])
		end
		plot!(p, αs, grad_avg, label="gradient")
		
		# plot
		ylabel!(p, "Average reward, 1000 steps")
		p
	end
end

# ╔═╡ 5f88d99c-1ef3-4223-82bd-3a29a033db54
let
	a = []
	for i = 1:5
		accumulate_mean!(a,i)
	end
	a
end

# ╔═╡ 00623f53-5a44-4c42-8e16-cafe490f642a
function plot_average_rewards(average_rewards, with_std=true, xtra=0)
	p = plot(title="Average rewards")
	
	if xtra ≠ 0
		for x in average_rewards
			plot!(p, x, opacity=0.3)
		end
	end
	
	μ = mean(average_rewards)
	σ = std(average_rewards, mean=μ)
	plot!(p, μ, ribbon=σ, lw=2)
	
	return p
end

# ╔═╡ faaa8dc7-c5c4-47cc-823a-1a116ba2a5e5
let
	if show_std
		opt_pcts, average_rewards = experiment(ε=eps1)
		plot_average_rewards(average_rewards)
	end
end

# ╔═╡ Cell order:
# ╠═48d4dd64-a47d-11eb-14d1-6161cbc1413a
# ╠═3f4a1865-6b69-4485-a285-b4ebee56c015
# ╠═a26cf864-d5a4-4d9e-91ad-fbfe4e385d87
# ╠═351277e6-6777-417b-96b8-023b4ef5707d
# ╠═c08545e2-5218-4489-8975-5c3f86c66345
# ╟─8cee4777-f9b7-4584-a498-3e08903ca9a3
# ╠═86e725ba-ff06-4aa4-bd85-0c1b6f30065b
# ╠═15f1567e-c69a-4ff6-8276-114628be3018
# ╟─cf6cb7ee-84d8-48fa-9573-07dfacee5962
# ╟─baa33a22-49e9-4469-b707-585a786d8124
# ╟─0b68c4f7-ba82-4e86-b4ad-0d52e8d510a4
# ╟─c311c9d2-cfd1-41b9-a47c-b5d7620e5937
# ╠═e084c765-f5e4-4bbb-b6dc-db72207bb6a1
# ╟─033f1958-f54a-4398-a1fd-a7d1c2dd6da5
# ╟─42797c0e-6785-4868-a78a-5f492351476d
# ╠═faaa8dc7-c5c4-47cc-823a-1a116ba2a5e5
# ╟─49f80719-524e-426c-b0f5-b8c8e2463692
# ╟─4910f9fb-3a75-40ce-bb1f-e5fcfe3b3da4
# ╟─7fb0c7d1-cd78-4996-aca9-c16dd7989d2a
# ╟─09e5bc8f-ed94-4d16-b722-27c3d6aaf56f
# ╠═770c1609-10aa-4720-b2d0-b89e64d18a9c
# ╟─029983e9-40fc-45a6-b4ef-4af0e5d96327
# ╠═018c6afe-6a60-4aee-a4da-8612955b64d2
# ╠═0bbdea00-060b-4ce0-a3e2-d3dc5d141649
# ╠═731acaab-e910-40bf-915e-7868d742abd1
# ╟─eb826b48-6fc2-468e-8bc7-d96861307b71
# ╟─81afbb4c-84e2-4c40-9983-78915da37df1
# ╟─34b6a4d6-b292-4dda-9eb6-1c6371773f43
# ╟─b01c2fe5-a4a8-43ea-9c36-4c5b785ea3a0
# ╠═3aabf6b6-f22a-47ee-8729-4a7840c0bebc
# ╟─e9831e70-5052-494e-8682-a88a37fbbd61
# ╟─b2df3274-7981-4fb6-82fa-d58354e02b60
# ╟─6a2bcf9e-f4d4-4e7e-9686-16fe8055c03d
# ╟─8a35ab08-a496-4599-bdd9-4f0a03b8bd01
# ╠═2889dd86-c0fc-4cee-9a2d-6b829cb746e1
# ╟─4b672fe5-fb95-4d00-b8d0-1621b8d62ec9
# ╠═14791977-1866-46f1-8ff8-6a107559f822
# ╟─871dfe49-59b5-40d5-9067-b41374bb949a
# ╟─462bb53b-a215-4cc7-ab97-a5b8260c423e
# ╟─638c891b-b223-4813-a997-318ae75355a1
# ╟─d8899170-07d9-49fd-9d03-3c9ceaca3a9d
# ╟─f4bbe00c-b033-415c-9c8e-e877270adc8f
# ╟─6eaf46db-dd07-4a47-96b1-6a07c632a79a
# ╟─e4012f10-c56a-48e1-adfb-07cfc07c4d87
# ╟─01d8ac29-c38e-4641-a6c8-8a05add94890
# ╠═d0e247a0-2c40-474c-b359-4d2fc045e882
# ╟─939cb982-427e-4c4f-91fa-dc60acb97a01
# ╟─8b4765e8-e2dd-4d0f-b86e-9d5afb6e77d2
# ╟─6cc2ca49-8d5d-411a-82c3-01f11c6743bd
# ╠═286c4c5d-bac8-4a46-9bbb-f1cbb75c9605
# ╟─f35e791f-9abf-4c70-ba2b-67139332b98f
# ╠═aa4d7178-5058-4caa-a5a6-1b6c96ede324
# ╠═609c08da-2250-4788-ac7d-fe381a5d34b6
# ╟─09b803c6-f06d-4813-bade-c0f93321daa3
# ╟─9bae2290-32c3-4b75-91b1-f589e421bcdb
# ╠═adffebe6-685a-46ab-93dd-5699f754ce99
# ╟─bf7d348b-c059-4cc2-9fc6-560f64751104
# ╟─b88c8d7e-20b2-4ab6-9a4c-492486cac649
# ╟─f720ba6c-c199-48e7-bebb-aaeae7654536
# ╠═2faf1f5e-9534-4913-ba7f-72ee9a8150ff
# ╟─8bd817d4-9cce-44ee-8595-567119a4cf43
# ╟─655ee62d-cd27-47c8-b3df-622aa539b98f
# ╟─92a924e6-8189-40e6-9cdf-3fad210267c8
# ╟─93b4b77a-8055-4d09-9024-a81efcb7d7f9
# ╟─d4023ff2-cfa3-49a8-9a1c-5eaa3f9b7a0e
# ╟─c2417fcc-016d-4be4-a1f0-b32a85b6a5a1
# ╟─ef167d48-dd50-4bb3-9ada-d53ec0b9e3a6
# ╟─5f88d99c-1ef3-4223-82bd-3a29a033db54
# ╠═00623f53-5a44-4c42-8e16-cafe490f642a
