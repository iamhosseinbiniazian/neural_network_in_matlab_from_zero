function Result = dActivation_Bipolar( x )
    Result = 0.5 .* (1 + Activation_Bipolar(x)).* (1 - Activation_Bipolar(x));
end
