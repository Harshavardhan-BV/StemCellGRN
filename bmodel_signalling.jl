include("Boolean.jl/dependencies.jl")
include("Boolean.jl/utils.jl")

function async_sig_Update(update_matrix::Array{Int,2},
    nInit::Int, nIter::Int, stateRep::Int; cytNodes::Vector{Int})
    n_nodes = size(update_matrix,1)
    stateVec = ifelse(stateRep == 0, [0,1], [-1,1])
    initVec = []
    finVec = []
    flagVec = []
    frustVec = []
    timeVec = []
    # states_df = DataFrame(init = String[], fin = String[], flag = Int[])
    update_matrix2 = 2*update_matrix + Matrix(I, n_nodes, n_nodes)
    update_matrix2 = sparse(update_matrix2')
    updFunc = ifelse(stateRep == 0, zeroConv, signVec)
    @showprogress for i in 1:nInit
        state = rand(stateVec, n_nodes) #pick random state
        state[cytNodes] .= 1
        init = join(zeroConv(state), "_")
        flag = 0
        time = 1
        uList = rand(1:n_nodes, nIter)
        j = 1
        while j <= nIter
            s1 = updFunc(update_matrix2*state)
            u = uList[j]
            if s1[u] != state[u]
                j = j + 1
                time = time + 1
                state[u] = s1[u]
                continue
            end
            while s1[u] == state[u]
                if iszero(j%10) # check after every ten steps,hopefully reduce the time
                    if s1 == state
                        flag = 1
                        break
                    end
                end
                j = j + 1
                time = time + 1
                if j > nIter
                    break
                end
                u = uList[j]
            end
            if flag == 1
                break
            end
            state[u] = s1[u]
        end
        if stateRep == 0
            fr = frustration(state, findnz(sparse(update_matrix)); negConv = true)
        else
            fr = frustration(state, findnz(sparse(update_matrix)))
        end
        fin = join(zeroConv(state), "_")
        push!(frustVec, fr)
        push!(initVec, init)
        push!(finVec, fin)
        push!(flagVec, flag)
        push!(timeVec, time)       
        # push!(states_df, (init, fin, flag))
    end
    states_df = DataFrame(init=initVec, 
            fin = finVec, flag = flagVec)
    frust_df = DataFrame(fin = finVec, 
        frust = frustVec)
    frust_df = unique(frust_df, :fin)
    timeData = DataFrame(fin = finVec, time = timeVec)
    timeData = groupby(timeData, :fin)
    timeData = combine(timeData, :time => avg, renamecols = false)
    frust_df = innerjoin(frust_df, timeData, on = :fin)
    # print(frust_df)
    return states_df, frust_df
end

function bmodel_sig(topoFile::String, cyts::Vector; nInit::Int64=10000, nIter::Int64=1000,
    stateRep::Int64=-1, type::Int=0)
    update_matrix,Nodes = topo2interaction(topoFile, type)
    cytNodes = findall(x -> x in cyts, Nodes)
    println(Nodes[cytNodes])
    if length(Nodes) > 50
        print("Too many nodes. Exiting.")
        return 0,0,0
    end
    state_df, frust_df = async_sig_Update(update_matrix, nInit, nIter, stateRep, cytNodes=cytNodes)

    # file_name = join([replace(topoFile, ".topo" => "_"), repl])
    # CSV.write(join(name,"_bmRes.csv"]), state_df)
    return state_df,Nodes, frust_df
end

function bmodel_sigreps(topoFile::String, cyts::Vector{String}; nInit::Int64=10000, nIter::Int64=1000,stateRep::Int64=-1, reps::Int = 3,
    types::Array{Int, 1} = [0],init::Bool=false, randSim::Bool=false, root::String="", nonMatrix::Bool = false,
    turnOffNodes::Union{Int64, Array{Int,1}} = Int[], 
        turnOffKey = "",
    oddLevel::Bool = false, negativeOdd::Bool = false,
    write::Bool = true, getData::Bool = false)
    update_matrix,Nodes = topo2interaction(topoFile)
    nNodes = length(Nodes)
    finFlagFreqFinal_df_list_list = []
    initFinFlagFreqFinal_df_list_list = []
    frust_df_list = []

    if (typeof(turnOffNodes) == Int64)
        turnOffNodes = [turnOffNodes]
    end

    for type in types

        finFlagFreqFinal_df_list = []
        initFinFlagFreqFinal_df_list = []
        frust_df_list = []

        for rep in 1:reps
            states_df, Nodes, frust_df = bmodel_sig(topoFile,cyts, nInit = nInit,nIter = nIter, stateRep = stateRep, type = type)
            if states_df == 0
                print("Too many nodes. Exiting.")
                return
            end
            # state_df = dropmissing(state_df, disallowmissing = true)
            push!(frust_df_list, frust_df)
            # Frequnecy table 
            #final states with flag
            finFlagFreq_df = dfFreq(states_df, [:fin, :flag])

            # all counts
            if init
                initFinFlagFreq_df = dfFreq(states_df, [:fin, :flag, :init])
                push!(initFinFlagFreqFinal_df_list, initFinFlagFreq_df)
            end
            push!(finFlagFreqFinal_df_list, finFlagFreq_df)
        end

        # println(typeof(finFreqFinal_df))
        finFlagFreqFinal_df = finFlagFreqFinal_df_list[1]
        if init
            initFinFlagFreqFinal_df = initFinFlagFreqFinal_df_list[1]
        end
        for i in 2:reps
            finFlagFreqFinal_df = outerjoin(finFlagFreqFinal_df, 
                finFlagFreqFinal_df_list[i], 
                on = [:states, :flag], makeunique=true)
            if init
                initFinFlagFreqFinal_df = outerjoin(initFinFlagFreqFinal_df, 
                    initFinFlagFreqFinal_df_list[i],
                    on = [:init, :states, :flag], makeunique = true)
            end
        end

        frust_df = reduce(vcat, frust_df_list)
        # for i in frust_df_list
        #     frust_df = vcat(frust_df, i)
        # end
        frust_df = unique(frust_df, [:fin, :time])
        frust_df = dfAvgGen(frust_df, [:fin, :frust], [:time])

        finFlagFreqFinal_df = meanSD(finFlagFreqFinal_df, "frequency")
        finFlagFreqFinal_df = outerjoin(finFlagFreqFinal_df, frust_df, 
            on = :states => :fin, makeunique =true)
        finFlagFreqFinal_df = rename(finFlagFreqFinal_df, 
            Dict(:Avg => Symbol(join(["Avg", type])), 
                :SD => Symbol(join(["SD", type])),
                :frust => Symbol(join(["frust", type]))))
        push!(finFlagFreqFinal_df_list_list, finFlagFreqFinal_df)


        if init
            initFinFlagFreqFinal_df = meanSD(initFinFlagFreqFinal_df, "frequency")
            initFinFlagFreqFinal_df = outerjoin(initFinFlagFreqFinal_df, frust_df, 
                on = :states => :fin, makeunique =true)
            initFinFlagFreqFinal_df = rename(initFinFlagFreqFinal_df, 
                Dict(:Avg => Symbol(join(["Avg", type])), 
                :SD => Symbol(join(["SD", type])),
                :frust => Symbol(join(["frust", type]))))
            push!(initFinFlagFreqFinal_df_list_list, initFinFlagFreqFinal_df)

        end
    end
        # println(typeof(finFreqFinal_df))
    finFlagFreqFinal_df = finFlagFreqFinal_df_list_list[1]
    if init
        initFinFlagFreqFinal_df = initFinFlagFreqFinal_df_list_list[1]
    end
    n = length(types)
    if n > 1
        for i in 2:n
            finFlagFreqFinal_df = outerjoin(finFlagFreqFinal_df, 
                finFlagFreqFinal_df_list_list[i], 
                on = [:states, :flag], makeunique=true)
            if init
                initFinFlagFreqFinal_df = outerjoin(initFinFlagFreqFinal_df, 
                    initFinFlagFreqFinal_df_list_list[i],
                    on = [:init, :states, :flag], makeunique = true)
            end
        end
    end

    if !randSim
        nodesName = replace(topoFile, ".topo" => "_nodes.txt")
        update_matrix,Nodes = topo2interaction(topoFile)
        io = open(nodesName, "w")
        for i in Nodes
            println(io, i)
        end
        close(io);
    end

    if write
        rootName = replace(topoFile, ".topo" => "")
        if root !=""
            rootName = join([rootName, "_",root])
        end
        if oddLevel
            rootName = join([rootName, "_oddLevel"])
            if negativeOdd
                rootName = join([rootName, "_negative"])
            else
                rootName = join([rootName, "_positive"])
            end
        end
        # println(rootName)
        if stateRep == 0
            rootName = join([rootName, "_nIsing"])
        end
        if nonMatrix
            rootName = join([rootName, "_nonMatrix"])
        end
        finFlagFreqName = join([rootName, "_finFlagFreq.csv"])

        CSV.write(finFlagFreqName, finFlagFreqFinal_df)


        if init
            initFinFlagFreqName = join([rootName, "_initFinFlagFreq.csv"])
            CSV.write(initFinFlagFreqName, initFinFlagFreqFinal_df)
        end
    end

    if getData
        if init
            return(finFlagFreqFinal_df, 
                initFinFlagFreqFinal_df)
        else
            return(finFlagFreqFinal_df)
        end
    end
end





