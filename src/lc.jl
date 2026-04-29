function lc(;link=link, constraint=constraint)
    N=1
    function constLC(ax, bx, kt, constraint)
        bx=bx[:,1]
        kt=kt[1,:]
        if constraint=="sum"
            c1=mean(kt)
        elseif constraint=="last"
            c1=kt[end]
        elseif constraint=="first"
            c1=kt[1]
        end
        ax=ax+c1*bx
        kt=kt.-c1
        c2=sum(bx)
        bx=bx/c2
        kt=kt.*c2
        return (ax=ax,bx=bx,kt=kt)    
    end

    function model(α, β, κ, i, j)
        return α[i]+β[i,1]*κ[1,j]
    end

    function staringvalues(Dxt,Ext,Ages_fit)
        log_m=log.(Dxt./Ext)
        log_m[isinf.(log_m)] .= 0
        log_m[isnan.(log_m)] .= 0
        ax0=mean(log_m, dims=2)[minimum(Ages_fit):maximum(Ages_fit)]
        Z=log_m .- mean(log_m, dims=2)
        U, Σ, V=svd(Z)
        bx0=U[:,1][minimum(Ages_fit):maximum(Ages_fit)]
        kt0=Σ[1]*V[:,1] 
        c1=mean(kt0)
        c2=sum(bx0)
        ax0=ax0+c1*bx0
        kt0=kt0.-c1
        bx0=bx0/c2
        kt0=kt0.*c2
        return vcat(ax0,bx0,kt0)
    end

    return (link=link, constFun=constLC, modelFun=model, startingValues=staringvalues, N=N, constraint=constraint)
end