include("/Volumes/A/MTech Project/Sig_ActA_LS/bmodel.jl")
using Base.Threads

fileList = readdir()
topoFiles = String[]
for i in fileList
	if endswith(i, "topo")
		push!(topoFiles, i)
	end
end

for CytStrength in 0:10
	for Asymmetry in 1:10
println(Threads.nthreads())
println("Strength of signalling = ", CytStrength)
println("Egde weight = ", Asymmetry)
Threads.@threads for topoFile in topoFiles
 	y1 = @elapsed x = bmodel_reps(topoFile; nInit = 100000, nIter = 1000, mode = "Async", stateRep = -1, randSim=false, shubham = false, SigStrength = CytStrength, EdgeWt = Asymmetry)
 	println(topoFile, " - ", y1, " seconds.")
end
end
end
#for topoFile in topoFiles
#	y3 = @elapsed x = edgeWeightPert(topoFile; nPerts = 100, nInit = 10000, types = [0])
#	println(topoFile, " - ", y3, " seconds.")
#end

