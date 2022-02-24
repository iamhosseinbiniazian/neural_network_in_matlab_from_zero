function [Error] = TestFunction( hObject, eventdata, handles,inputValues, targetValues,BTSize)
global NNetwork;
IS= NNetwork.gactivetyflag;
   if IS== 1
        ActivationFunction = @LogisticSigmoid;
        dActivationFunction = @dLogisticSigmoid;        
    else
        ActivationFunction = @Activation_Bipolar;
        dActivationFunction = @dActivation_Bipolar;        
   end
        error=0;
        LayerNeruonNumber=NNetwork.gLNN;
        Heddinlayer=NNetwork.ghl;
        W=NNetwork.gW;
        X = [];
        Y = [];
        OV = zeros(max(LayerNeruonNumber),Heddinlayer+2);
        IV = zeros(max(LayerNeruonNumber),Heddinlayer+2);
          
         TSSize = size(inputValues, 2);    
    IDim = size(inputValues, 1);
    ODim = size(targetValues, 1); 
        for i = 1:BTSize
    
          inputVector = inputValues(:, i);
            targetVector = targetValues(:,i);
            OV(1:IDim,1) = inputVector;
            
            for k = 1:Heddinlayer + 1 % for training
                
                IV(1:LayerNeruonNumber(k+1),k+1) = W(1:LayerNeruonNumber(k),1:LayerNeruonNumber(k+1),k)' * OV(1:LayerNeruonNumber(k),k);
                OV(1:LayerNeruonNumber(k+1),k+1) = ActivationFunction(IV(1:LayerNeruonNumber(k+1), k+1));
                
            end
             error = error + norm(targetVector - OV(1:LayerNeruonNumber(Heddinlayer+2),Heddinlayer+2));
              maxOutput = max(OV(1:LayerNeruonNumber(Heddinlayer+2),Heddinlayer+2));
            tmpOutput = OV(1:LayerNeruonNumber(Heddinlayer+2),Heddinlayer+2);
            if LayerNeruonNumber(1) == 2
                tmpOutput(find(tmpOutput <= 0.5)) = 0;
                tmpOutput(find(tmpOutput > 0.5)) = 1;            
            else
                tmpOutput(find(tmpOutput < maxOutput)) = 0;
                tmpOutput(find(tmpOutput == maxOutput)) = 1;
            end
        
        end
     error = error/BTSize;
     Error=error;
end