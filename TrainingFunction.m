function Result = TrainingFunction(hObject, eventdata, handles,Lrate,BSize,BESize,Momentum,LayerNeruonNumber,Heddinlayer,activetyflag,stopcon,scvalue,isWidrow)
    global NNetwork;
    global PW;
    global ERROR;
    global ERRORE;
    global LabelValues;
    global InputValues;
    global IFT;
    global WWW;
    NNetwork.gactivetyflag=activetyflag;
      if activetyflag== 1
        ActivationFunction = @LogisticSigmoid;
        dActivationFunction = @dLogisticSigmoid;        
    else
        ActivationFunction = @Activation_Bipolar;
        dActivationFunction = @dActivation_Bipolar;        
      end
        inputValues = InputValues; 
        labels = LabelValues;
        TargetValues = zeros(10, size(labels, 1));
        for n = 1: size(labels, 1)
            TargetValues(labels(n) + 1, n) = 1;
        end 
         TargetValuesE=[];
          inputValuesE=[];
        isFirst_Train = IFT;
    TSSize = size(inputValues, 2);    
    IDim = size(inputValues, 1);
    ODim = size(TargetValues, 1);  
        if isFirst_Train == 1
        n = floor(rand(BSize, 1)*TSSize + 1);% testcase haE ke mibayest amozesh dade shavand moshakhas mishavand.
        W = rand(max(LayerNeruonNumber),max(LayerNeruonNumber),Heddinlayer+2)-.5; % weigth matrix
        if isWidrow == 1
            for i = 1:Heddinlayer+1
                beta = .7 * (sum(LayerNeruonNumber(2):LayerNeruonNumber(Heddinlayer+1)).^(1/LayerNeruonNumber(1)));
                normW = norm(W(1:LayerNeruonNumber(i),1:LayerNeruonNumber(i+1),i));
                W(1:LayerNeruonNumber(i),1:LayerNeruonNumber(i+1),i) = beta * (W(1:LayerNeruonNumber(i),1:LayerNeruonNumber(i+1),i)/normW);
            end
        end
        NNetwork.gW = W;
        PW = W;
        ERROR = [];
        ERRORE=[];
    else
        W = NNetwork.gW;
        Heddinlayer=NNetwork.ghl;
        LayerNeruonNumber=NNetwork.gLNN;
        end
       OV = zeros(max(LayerNeruonNumber),Heddinlayer+2);
    IV = zeros(max(LayerNeruonNumber),Heddinlayer+2);
    WeightD=[];
    Delta = zeros(max(LayerNeruonNumber),Heddinlayer+2);
    X = [];
    Y = [];
    inputValues=inputValues(:,n);
    TargetValues=TargetValues(:,n);
    n1 = floor(rand(BESize, 1)*BSize+1);
    TargetValuesE=TargetValues(:,n1);
    inputValuesE= inputValues(:,n1);
    TargetValues(:,n1)=[];
    inputValues(:,n1)=[];
    if stopcon==1;
      
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   for i = 1:scvalue        
        error = 0;
        for j = 1:(BSize-BESize-1)
            inputVector = inputValues(:, j);
            targetVector = TargetValues(:,j);
            OV(1:IDim,1) = inputVector;
            
            for k = 1:Heddinlayer + 1 % for training
                
                IV(1:LayerNeruonNumber(k+1),k+1) = W(1:LayerNeruonNumber(k),1:LayerNeruonNumber(k+1),k)' * OV(1:LayerNeruonNumber(k),k);
                OV(1:LayerNeruonNumber(k+1),k+1) = ActivationFunction(IV(1:LayerNeruonNumber(k+1), k+1));
                
            end
            
            
            Delta(1:LayerNeruonNumber(Heddinlayer+2),Heddinlayer+2) = dActivationFunction(IV(1:LayerNeruonNumber(Heddinlayer+2),Heddinlayer+2)) .* (targetVector - OV(1:LayerNeruonNumber(Heddinlayer+2),Heddinlayer+2));
         
            
            for k = Heddinlayer + 1:-1:2
                Delta(1:LayerNeruonNumber(k),k) = dActivationFunction(IV(1:LayerNeruonNumber(k),k)) .* (W(1:LayerNeruonNumber(k), 1:LayerNeruonNumber(k+1), k) * Delta(1:LayerNeruonNumber(k+1),k+1)); 
            end
            
            tmpW = W;
            for k = 1:Heddinlayer + 1
                W(1:LayerNeruonNumber(k), 1:LayerNeruonNumber(k+1) ,k) = W(1:LayerNeruonNumber(k), 1:LayerNeruonNumber(k+1) ,k) + Lrate .* (OV(1:LayerNeruonNumber(k),k) * Delta(1:LayerNeruonNumber(k+1),k+1)') + Momentum .* (W(1:LayerNeruonNumber(k), 1:LayerNeruonNumber(k+1) ,k) - PW(1:LayerNeruonNumber(k), 1:LayerNeruonNumber(k+1) ,k));
            end
            NNetwork.gW = W;
            PW = tmpW;
            
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
        WeightD=[];
        for hhh=1:Heddinlayer+1
            
               WeightD=[WeightD W(:,:,hhh)];
        
        end
    WWW=WeightD;
        X = [X i];
        Y = [Y error];
        Show(hObject, eventdata, handles , WeightD,[X;Y],2,'Epoch', 'Error');
        error = error/BSize;
        X = [X i];
        Y = [Y error];
     
 
        %%%%%%%%%%%%%%
        Erroree = TestFunction( hObject, eventdata, handles,inputValuesE, TargetValuesE,BESize);
        ERRORE=[ERRORE Erroree];
            ERROR = [ERROR error];
            ttt = 1:size(ERROR,2);
            Show(hObject, eventdata, handles , [ttt;ERROR],[ttt;ERRORE],1,'Epoch', 'Error');
            NNetwork.gW = W;
    %%%%%%%%%%%%%%%%%%
        
    end
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
        error=0;
        Erroree=1;
        i=1;
        while Erroree>scvalue
            error = 0;
            Erroree=0;
        for j = 1:(BSize-BESize-1)
            inputVector = inputValues(:, j);
            targetVector = TargetValues(:,j);
            OV(1:IDim,1) = inputVector;
            
            for k = 1:Heddinlayer + 1 % for training
                
                IV(1:LayerNeruonNumber(k+1),k+1) = W(1:LayerNeruonNumber(k),1:LayerNeruonNumber(k+1),k)' * OV(1:LayerNeruonNumber(k),k);
                OV(1:LayerNeruonNumber(k+1),k+1) = ActivationFunction(IV(1:LayerNeruonNumber(k+1), k+1));
                
            end
            
            
            Delta(1:LayerNeruonNumber(Heddinlayer+2),Heddinlayer+2) = dActivationFunction(IV(1:LayerNeruonNumber(Heddinlayer+2),Heddinlayer+2)) .* (targetVector - OV(1:LayerNeruonNumber(Heddinlayer+2),Heddinlayer+2));
      
            
            for k = Heddinlayer + 1:-1:2
                Delta(1:LayerNeruonNumber(k),k) = dActivationFunction(IV(1:LayerNeruonNumber(k),k)) .* (W(1:LayerNeruonNumber(k), 1:LayerNeruonNumber(k+1), k) * Delta(1:LayerNeruonNumber(k+1),k+1)); 
            end
            
            tmpW = W;
            for k = 1:Heddinlayer + 1
                W(1:LayerNeruonNumber(k), 1:LayerNeruonNumber(k+1) ,k) = W(1:LayerNeruonNumber(k), 1:LayerNeruonNumber(k+1) ,k) + Lrate .* (OV(1:LayerNeruonNumber(k),k) * Delta(1:LayerNeruonNumber(k+1),k+1)') + Momentum .* (W(1:LayerNeruonNumber(k), 1:LayerNeruonNumber(k+1) ,k) - PW(1:LayerNeruonNumber(k), 1:LayerNeruonNumber(k+1) ,k));
            end
            NNetwork.gW = W;
            PW = tmpW;
            
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
        
       WeightD=[];
        for hhh=1:Heddinlayer+1
            
               WeightD=[WeightD W(:,:,hhh)];
        
        end
WWW=WeightD;
        X = [X i];
        Y = [Y error];
        Show(hObject, eventdata, handles , WeightD,[X;Y],2,'Epoch', 'Error');
        error = error/BSize;
        %%%%%%%%%%%%%
        Erroree = TestFunction( hObject, eventdata, handles,inputValuesE, TargetValuesE,BESize);
        ERRORE=[ERRORE Erroree];
            ERROR = [ERROR error];
            ttt = 1:size(ERROR,2);
            Show(hObject, eventdata, handles , [ttt;ERROR],[ttt;ERRORE],1,'Epoch', 'Error');
            NNetwork.gW = W;
    %%%%%%%%%%%%%%%%%%
    i=i+1;
        end
    end
 
end