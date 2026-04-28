function genWeightMat(ages, years; clip=0, zeroCohorts=nothing)
    nAges=length(ages)
    nYears=length(years)
    cohorts=(years[1]-ages[end]):(years[end]-ages[1])
    Wxt=ones(nAges, nYears)
    
    if clip>0
        cl=ones(clip,clip)
        triu!(cl,1)
        Wxt[nAges-clip+1:nAges,1:clip]=cl
        Wxt[1:clip,nYears-clip+1:nYears]=transpose(cl)
    end
    if zeroCohorts!=nothing
        for i in zeroCohorts
            h= i-cohorts[1]+1-nAges
            if h<=0
                col=1
                row=1-h
            else
                col=h+1
                row=1
            end

            while col<=nYears && row<=nAges
                Wxt[row,col]=0
                col+=1
                row+=1 
            end
        end
    end
    return Wxt
end