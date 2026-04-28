function fit_StMoMo(N,Dxt,Ext,Ages_fit,Wxt,Years)
    neg_loglikelihood=function(params,N,Dxt,Ext,Ages_fit,Wxt,Years)
        min_Age=minimum(Ages_fit)
        nalpha=length(Ages_fit)
        nbeta=length(Ages_fit)*N
        α=params[1:nalpha]
        β=reshape(params[1+nalpha:(nalpha+nbeta)],nalpha,N)
        κ=reshape(params[(nalpha+nbeta+1):end],N,length(Years))
        βκ=β*κ
        likelihood=0
        
        for i in 1:nalpha
            for j in 1:length(Years)
                d_hat_xt=Ext[i+min_Age,j]*exp(α[i]+βκ[i,j])
                dxt=Dxt[i+min_Age,j]
                w=Wxt[i,j]
                likelihood-=w*(dxt*log(d_hat_xt)-d_hat_xt-logfactorial(dxt))
            end
        end
        return likelihood
    end
    startingvalues_LC=function(Dxt,Ext,Ages_fit)
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

    X0=startingvalues_LC(Dxt,Ext,Ages_fit)
    x=optimize(p->neg_loglikelihood(p,N,Dxt,Ext,Ages_fit,Wxt,Years),X0,BFGS(); autodiff=AutoForwardDiff())  
    
    return(-x.minimum)
end
