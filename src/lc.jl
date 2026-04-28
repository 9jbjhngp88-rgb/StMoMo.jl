function lc(;link=["log", "logit"], constraint=["sum", "last", "first"])
    function constLC(ax, bx, kt,b0x, gc, wxt, ages)
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
        return (ax=ax,bx=bx,kt=kt,b0x=b0x,gc=gc)    
    end
    function dxt_hat(Ext, α, βκ, i, j)
        return Ext[i+min_Age,j]*exp(α[i]+βκ[i,j])
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
        return vcat(ax0,bx0,kt0)
    end
    return (link=link,  staticAgeFun=true, periodAgeFun="NP", cohortAgeFun= nothing, constFun=constLC(), N=N,optim_func=dxt_hat,starting_values=staringvalues())
end