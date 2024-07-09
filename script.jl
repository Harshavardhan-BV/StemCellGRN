include("./Boolean.jl/bmodel.jl")
using Base.Threads

fileList = readdir()
topoFiles = String[]
for i in fileList
    if endswith(i, "topo")
        push!(topoFiles, i)
    end
end

println(Threads.nthreads())

Threads.@threads for topoFile in topoFiles
    y1 = @elapsed x = bmodel_reps(topoFile; nInit = 100000, nIter = 1000, mode = "Async", stateRep = -1, randSim=false, shubham = false)
    println(topoFile, " - ", y1, " seconds.")
end
