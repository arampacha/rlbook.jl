### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

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

# ╔═╡ Cell order:
# ╠═c3b6bce8-ab7e-11eb-292e-3bb1464bfad2
# ╠═786fdaa3-db0e-408c-b1cb-d839735f7e38
# ╠═78c1efb9-823d-4103-b3df-cba410804543
# ╠═d00cf809-6f0c-4057-b2ac-ad2c81eba6ae
# ╠═7da62397-7f49-4109-82a9-470c37fb31ae
# ╠═52eaab79-825d-487c-beb3-a647710037ee
# ╟─326c9837-5770-4da0-afda-9f9950b56563
