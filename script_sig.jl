include("bmodel_signalling.jl")
using Base.Threads

fileList = readdir()
topoFiles = String[]
for i in fileList
    if endswith(i, "topo")
        push!(topoFiles, i)
    end
end

cyts = ["IFNG","IL12","IL2","IL4","TGFB","IL6","IL21"]
println(Threads.nthreads())

Threads.@threads for topoFile in topoFiles
    y1 = @elapsed x = bmodel_sigreps(topoFile, cyts; nInit = 100000, nIter = 1000)
    println(topoFile, " - ", y1, " seconds.")
end