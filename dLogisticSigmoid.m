function Result = dLogisticSigmoid(x)
    Result = LogisticSigmoid(x).*(1 - LogisticSigmoid(x));
end