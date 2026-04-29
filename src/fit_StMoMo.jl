function neg_loglikelihood(params,N,Dxt,Ext,Ages_fit,Wxt,Years, modelFun)
    min_Age=minimum(Ages_fit)
    nalpha=length(Ages_fit)
    nbeta=length(Ages_fit)*N
    α=params[1:nalpha]
    β=reshape(params[(1+nalpha):(nalpha+nbeta)],nalpha,N)
    κ=reshape(params[(nalpha+nbeta+1):end],N,length(Years))
    # βκ=β*κ
    likelihood=0

    for i in 1:nalpha
        for j in 1:length(Years)
            d_hat_xt=Ext[i+min_Age,j]*exp(modelFun(α, β, κ, i, j))
            dxt=Dxt[i+min_Age,j]
            w=Wxt[i,j]
            likelihood-=w*(dxt*log(d_hat_xt)-d_hat_xt-logfactorial(dxt))
        end
    end
    return likelihood
end

function fit_StMoMo(; model=model,Dxt=nothing,Ext=nothing,Ages_fit=nothing,Wxt=nothing,Years=nothing)
    link=model.link
    constFun=model.constFun
    modelFun=model.modelFun
    startingvalues=model.startingValues
    N=model.N
    constraint=model.constraint
    

    X0=startingvalues(Dxt,Ext,Ages_fit)
    #x1=optimize(p->neg_loglikelihood(p,N,Dxt,Ext,Ages_fit,Wxt,Years, modelFun=modelFun),X0,NelderMead())  
    t1= @elapsed x=optimize(p -> neg_loglikelihood(p,N,Dxt,Ext,Ages_fit,Wxt,Years, modelFun),X0,BFGS(); autodiff=AutoForwardDiff())

        
    nalpha=length(Ages_fit)
    nbeta=nalpha*N
    α=x.minimizer[1:nalpha]
    β=reshape(x.minimizer[(1+nalpha):(nalpha+nbeta)],nalpha,N)
    κ=reshape(x.minimizer[(nalpha+nbeta+1):end],N,length(Years))
    res=constFun(α, β, κ, constraint)
    return (loglikelihood=-x.minimum, α=res.ax, β=res.bx, κ=res.kt, type=typeof(constFun))
end
