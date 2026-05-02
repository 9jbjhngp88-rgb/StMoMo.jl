function neg_loglikelihood(params,N,Dxt,Ext,Wxt,Years, modelFun, nalpha, nbeta, nYears)
    α=params[1:nalpha]
    β=reshape(params[(1+nalpha):(nalpha+nbeta)],nalpha,N)
    κ=reshape(params[(nalpha+nbeta+1):end],N,length(Years))
    likelihood=0
    @inbounds for i in 1:nalpha
        α_i=α[i]
        β_i=β[i,1]
        for j in 1:nYears
            model=modelFun(α_i, β_i, κ[1,j])
            likelihood-=Wxt[i,j]*(Dxt[i,j]*model-Ext[i,j]*exp(model))
        end
    end
    return likelihood
end

function fit_StMoMo(; model=nothing,Dxt=nothing,Ext=nothing,Ages_fit=nothing,Wxt=nothing,Years=nothing)
    link=model.link
    constFun=model.constFun
    modelFun=model.modelFun
    startingvalues=model.startingValues
    N=model.N
    constraint=model.constraint

    X0=startingvalues(Dxt,Ext,Ages_fit)

    min_Age=minimum(Ages_fit)
    max_ages=maximum(Ages_fit)
    nalpha=length(Ages_fit)
    nbeta=nalpha*N
    nYears=length(Years)
    Ext=Ext[min_Age+1:max_ages+1,:]
    Dxt=Dxt[min_Age+1:max_ages+1,:]

   
    x=optimize(p -> neg_loglikelihood(p,N,Dxt,Ext,Wxt,Years, modelFun, nalpha, nbeta, nYears),X0,BFGS(); autodiff=AutoForwardDiff())

    α=x.minimizer[1:nalpha]
    β=reshape(x.minimizer[(1+nalpha):(nalpha+nbeta)],nalpha,N)
    κ=reshape(x.minimizer[(nalpha+nbeta+1):end],N,length(Years))
    res=constFun(α, β, κ, constraint)
    return (loglikelihood=-x.minimum, α=res.ax, β=res.bx, κ=res.kt)
end
