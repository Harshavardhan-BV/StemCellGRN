include("./Boolean.jl/bmodel.jl")
using Base.Threads

minWt = 0.0
maxWt = 1.0
nPert = 100 # Number of samples of edge weights

fileList = readdir()
topoFiles = String[]
for i in fileList
	if endswith(i, "topo")
		push!(topoFiles, i)
	end
end

println(Threads.nthreads())

for topoFile in topoFiles
 	y1 = @elapsed x = edgeWeightPert(topoFile; nPerts = nPert, nIter=1000, nInit = 100000,
			minWeight = minWt, maxWeight = maxWt)
	println(topoFile, " - ", y1, " seconds.")
end
