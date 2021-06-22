### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ b90d0a00-cfa4-11eb-091e-7f136ef1ed78
begin
    import Pkg
    Pkg.activate("/media/arto/work/dev/git/rlbook.jl/Project.toml")
	
    using Statistics, Plots, PlutoUI, LinearAlgebra, StatsBase, Distributions
end

# ╔═╡ e357b25f-7f0e-434e-a5ce-9be2c44f2ec3
begin
	abstract type AbstractEnv end
	abstract type AbstractAgent end
	abstract type AbstractPolicy end
end

# ╔═╡ cfc6b3a8-048f-447c-86ff-28053b4ce3be
md"# Racetrack"

# ╔═╡ 3ef9347b-6426-4ce8-8468-2dd0620e086b
const racetrack_actions = reshape([(i,j) for i=-1:1, j=-1:1], 9)

# ╔═╡ b5fe1032-8246-4372-a66c-0e7c624f0214
function orientation(a, b, c)
	xa, ya = a
	xb, yb = b
	xc, yc = c
	s = (yb - ya)*(xc - xb) - (yc - yb)*(xb - xa)
	if s < 0
		return -1
	elseif s > 0
		return 1
	else
		return 0
	end
end

# ╔═╡ 55d2030a-3f89-498c-a970-4d1ef1a4fdc0
function intersects(seg1, seg2)
	# (xa1, ya1), (xb1, xb1) = seg1
	# (xa2, ya2), (xb2, xb2) = seg2
	(xa1, xb1), (ya1, yb1) = seg1
	(xa2, xb2), (ya2, yb2) = seg2
	if ((min(xa1,xb1) > max(xa2, xb2)) ||
		(max(xa1,xb1) < min(xa2, xb2)) ||
		(min(ya1,yb1) > max(ya2, yb2)) ||
		(max(ya1,yb1) < min(ya2, yb2)))
		return false
	else
		if orientation((xa1, ya1), (xb1, yb1), (xa2, ya2)) != orientation((xa1, ya1), (xb1, yb1), (xb2, yb2))
			return true
		else
			return false
		end
	end
end

# ╔═╡ 97b340af-2b70-45a2-9f14-f4198c8316e3
begin
	struct Racetrack <: AbstractEnv
		m::Int
		n::Int
		track::Matrix
		start::AbstractRange
		finish::AbstractRange
	end
	
	function (env::Racetrack)(s::Tuple, a::Tuple)
		
		x, y, vx, vy = s
		ax, ay = a
		vx = clamp(vx+ax, 0,4)
		vy = clamp(vy+ay, 0,4)
		new_x = x+vx
		new_y = y+vy
		finishline = ((env.n, env.n), (env.finish.start-0.5, env.finish.stop+0.5))
		# check if crossed finish line
		if intersects(((x, new_x),(y,new_y)), finishline)
			return 0, (new_x,new_y,0,0), true
		end
		
		if (new_y > env.m) || (new_x > env.n) || (env.track[new_y,new_x] == 0)
			new_x, new_y = rand(env.start), 1
			vx, vy = 0,0
		end
		finished = false
		return -1, (new_x,new_y,vx,vy), finished
	end
	
	function show_track(track::Racetrack)
		p = heatmap(track.track, legend=:none)
		plot!(p, [track.start.start-0.5, track.start.stop+0.5], [1,1], linewidth=5, colour=:red)
		plot!(p, [track.n, track.n] , [track.finish.start-0.5, track.finish.stop+0.5], linewidth=5, colour=:green)
		plot!(size=(track.n*20, track.m*20))
	end
	
	mutable struct Car <: AbstractAgent
		state::Tuple # (x, y, vx, vy)
		actions::Vector
	end
end

# ╔═╡ d7c473df-0c00-4999-8a14-6a7333c9e040
function make_track()
	m, n = 30, 20
	track = zeros(m,n)
	for i = 1:m, j = 1:n
		track[i,j] = Int(
			(i≤20 && j < 10 && !(0.5*i + 2*j < 10)) ||
			i>20
		)
	end
	start = track[1,:] .== 1
	s1, s2 = findfirst(start), findlast(start)
	finish = track[:, end] .== 1
	f1, f2 = findfirst(finish), findlast(finish)
	Racetrack(m,n,track, s1:s2, f1:f2)
end

# ╔═╡ b5310945-b90c-4eea-b4c3-78add711b16b
show_track(make_track())

# ╔═╡ 4d0b01ad-ebd6-449f-8ee0-0f585e8d0263
let
	env = make_track()
	car = Car((18,25, 2,0), racetrack_actions)
	
	env(car.state, racetrack_actions[9])
	finishline = ([env.n, env.n], [env.finish.start-0.5, env.finish.stop+0.5])
	fin = intersects(((18,21), (25,25)), finishline)
	# p = plot([18,21], [25,25])
	# plot!(p, finishline)
	# title!(fin ? "True" : "False")
end

# ╔═╡ ce19b163-73b9-4bbd-9054-d82a710b6852
let
	track = make_track()
	car = Car((5,1, 0,0), racetrack_actions)
	
	
	p = show_track(track)
	
	trajectory = []
	push!(trajectory, car.state[1:2])
	actions = []
	s = car.state
	for i = 1:10
		a = (i%2 == 0) ? 5 : 9
		push!(actions, racetrack_actions[a])
		r, s, f = track(s, racetrack_actions[a])
		push!(trajectory, s[1:2])
	end
	actions
	
	trajectory
	xs, ys = [x[1] for x in trajectory], [x[2] for x in trajectory]
	plot!(p, xs, ys, linewidth=3, colour=:blue)
	
end

# ╔═╡ faa01632-a71c-435f-a8da-c2c26049ac99
racetrack_actions

# ╔═╡ f89a04bc-6601-4450-a93b-5fe0993967ca
function state2idx(state)
	x, y, vx ,vy = state
	return y, x, vx+1, vy+1
end

# ╔═╡ 16072174-2ae3-4b55-8bb6-820d88be2832
function episode(agent::Car, env::Racetrack, π::Array; first_action=nothing)
	history = []
	finished = false
	state = agent.state
	i = 0
	while !(finished) && (i < 10_000)
		i += 1
		if i > 1 || first_action == nothing
			action = sample(1:9, ProbabilityWeights(π[state2idx(state)..., :]))
		else
			action = first_action
		end
		reward, new_state, finished = env(state, racetrack_actions[action])
		push!(history, (s=state, a=action, r=reward))
		state = new_state
	end
	push!(history, (s=state, a=5, r=0))
	history
end

# ╔═╡ be8dedf0-f1c6-4777-a97b-7676728659d7
function plot_trajectory!(p, history)
	xs = zeros(Int64, length(history))
	ys = similar(xs)
	for (i, s) in enumerate(history)
		@inbounds xs[i] = s.s[1]
		@inbounds ys[i] = s.s[2]
	end
	
	plot!(p, xs, ys, colour=:blue, linewidth=:2)
end

# ╔═╡ 18489b80-1f75-43f9-8862-970d7c95c591
begin
	track = make_track()
	π = zeros(track.m,track.n,5,5,9) .+0.5
	car = Car((rand(track.start), 1, 0, 0), racetrack_actions)
	
	history = episode(car, track, π)
	
	p = show_track(track)
	plot_trajectory!(p, history)
	
end

# ╔═╡ 6b894555-90d0-422b-8b1b-eaca6d7789f7
function off_policy_control!(π::Array, b::Array, max_iter::Int=100)
	# evaluete naive policy
	d = length(size(π))
	@assert all(abs.(sum(π, dims=d) .-1) .< 1e-4) "Policy π should be given as valid action probability distribution"
	@assert all(abs.(sum(b, dims=d) .-1) .< 1e-4) "Policy b should be given as valid action probability distribution"
	env = make_track()
	γ = 1
	Q  = zeros(size(π))
	C  = zeros(size(Q))
	history = []
	for i = 1:max_iter
		agent = Car((rand(track.start), 1, 0, 0), racetrack_actions)
		# behaviour policy
		
		
		history = episode(agent, env, b)
		visited_states = map(x -> x.s, history)
		G = 0
		W = 1
		T = length(history) - 1
		for i = T:-1:1
			s, a, r = history[i]
			G = γ*G + r
			
			idx = (state2idx(s)..., a)
			# Q-values update
			C[idx...] += W
			Q[idx...] += (G - Q[idx...])*W/C[idx...]
			# policy update
			pidx = idx[1:end-1]
			a_star = argmax(Q[pidx..., :])
			π[pidx..., :] .= 0.
			π[pidx..., a_star] += 1.
			@assert all(abs.(sum(π, dims=d) .-1) .< 1e-4)
			if π[idx...] == 0.
				break
			end
			W *= 1 / b[idx...]
		end
	end
	Q, π, C
end

# ╔═╡ 6aacf5d9-c822-4fc3-9ec5-7d5ef86ce0a6
let
	π = zeros(track.m,track.n,5,5,9) .+1/9
	b = zeros(track.m,track.n,5,5,9) .+1/9
	Q, π, C = off_policy_control!(π, b, 1000)
	track = make_track()
	car = Car((rand(track.start), 1, 0, 0), racetrack_actions)
	history = episode(car, track, π)
	p = show_track(track)
	plot_trajectory!(p, history)
end

# ╔═╡ Cell order:
# ╟─b90d0a00-cfa4-11eb-091e-7f136ef1ed78
# ╟─e357b25f-7f0e-434e-a5ce-9be2c44f2ec3
# ╟─cfc6b3a8-048f-447c-86ff-28053b4ce3be
# ╠═3ef9347b-6426-4ce8-8468-2dd0620e086b
# ╟─b5fe1032-8246-4372-a66c-0e7c624f0214
# ╟─55d2030a-3f89-498c-a970-4d1ef1a4fdc0
# ╠═97b340af-2b70-45a2-9f14-f4198c8316e3
# ╠═d7c473df-0c00-4999-8a14-6a7333c9e040
# ╠═b5310945-b90c-4eea-b4c3-78add711b16b
# ╟─4d0b01ad-ebd6-449f-8ee0-0f585e8d0263
# ╠═ce19b163-73b9-4bbd-9054-d82a710b6852
# ╠═faa01632-a71c-435f-a8da-c2c26049ac99
# ╠═f89a04bc-6601-4450-a93b-5fe0993967ca
# ╠═16072174-2ae3-4b55-8bb6-820d88be2832
# ╠═18489b80-1f75-43f9-8862-970d7c95c591
# ╠═be8dedf0-f1c6-4777-a97b-7676728659d7
# ╠═6b894555-90d0-422b-8b1b-eaca6d7789f7
# ╠═6aacf5d9-c822-4fc3-9ec5-7d5ef86ce0a6
