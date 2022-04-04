SetDirectory["/data/home/scy0446/run/Data"];
months=DateRange[{1979,1,1},{1989,12,1},"Month"];
p=Block[{tempt=Table[Block[{},Print[months[[i]]];Import["N_"<>DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["data"]],{i,Length[months]}]},
        tempt=tempt[[;;,1]];
        NumericArray[Flatten[Normal[tempt],1],"Real32"]];
ele=NumericArray[{Log[Normal[Import["/data/home/scy0446/run/Data/CONUS/Elevation.mx"]["data"]]+1.]},"Real32"];
dmeanvar=Import["/data/home/scy0446/run/Data/CONUS/meanvar_dynamics.mx"]["data"];
SetDirectory["/data/home/scy0446/run/Data/CONUS"];
var={"pressfc","prmsl","pwat","q500","q850","q925","tmpsfc","u200","u850","v200","v850","z1000","z200","z500","z850"};
{dlat,dlon}=Block[{tempt=Import[DateString[{1979,1,1},{"D_CONUS_","Year","_","Month",".mx"}]]},{tempt["lat"],tempt["lon"]}];
dynamics=Block[{tempt=Table[Block[{},Print[months[[i]]];Import[DateString[months[[i]],{"D_CONUS_","Year","_","Month",".mx"}]]["data"]],{i,Length[months]}]},
 Flatten[Map[Table[#[[;;,((i-1)*3+1);;(i*3)]],{i,Dimensions[#][[2]]/3}]&,tempt]]];

cStatic=4;
cDynamic=32;
cEncoder=16;

dStatic={1,301,621};
dDynamic={15,3,59,123};
dLatent=4;
dOutput={1,40,80};

StaticNet=Block[{c=cStatic},
        NetGraph[
 <|"resize"->{ResizeLayer[{320,640}],ConvolutionLayer[c,{1,1}], ParametricRampLayer[]},
   "down1"->{Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],ParametricRampLayer[]},3],PoolingLayer[2,2]},
   "down2"->{Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],ParametricRampLayer[]},3],PoolingLayer[2,2]},
   "down3"->{Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],ParametricRampLayer[]},3],PoolingLayer[2,2]}|>,
        {NetPort["Static"]->"resize",
        "resize"->"down1",
        "down1"->"down2",
        "down2"->"down3"->NetPort["Down3"]},
"Static"->{1,301,621}]]

res[c_]:=NetGraph[<|"l"->Flatten[Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],BatchNormalizationLayer[],Ramp},3]][[1;;-2]],
          "plus"->TotalLayer[],
          "post"->Ramp|>,
   {NetPort["Input"]->"l"->"plus",NetPort["Input"]->"plus"->"post"}]

DynamicNet1=Block[{c=cDynamic,c2=cDynamic+cStatic},
        NetGraph[
 <|"resize0"->{Table[{ConvolutionLayer[c,{2,3,3}],ParametricRampLayer[]},2],FlattenLayer[1],ResizeLayer[{40,80}]},
   "cate0"->CatenateLayer[],
   "up0"->{res[c2],res[c2],res[c2]}|>,
   {NetPort["Dynamics"]->"resize0"->"cate0",
    NetPort["Static0"]->"cate0"->"up0"},
"Dynamics"->{15,3,59,123},
"Static0"->{cStatic,40,80}]]

contract[channel_,crop_:{{1,1},{1,1}}]:=NetGraph[{"conv"->{ConvolutionLayer[channel,{3,3},"Stride"->1,"PaddingSize"->1],Ramp,
                                                           ConvolutionLayer[channel,{3,3},"Stride"->1,"PaddingSize"->1],Ramp},
                                   "pooling"->PoolingLayer[2,2],
                                   "cropping"->PartLayer[{;;,crop[[1,1]];;-crop[[1,-1]],crop[[2,1]];;-crop[[2,-1]]}]},
                         {NetPort["Input"]->"conv"->"pooling"->NetPort["Pooling"],"conv"->"cropping"->NetPort["Shortcut"]}];

expand[channel_]:=NetGraph[{"deconv"->{DeconvolutionLayer[channel,{2,2},"Stride"->{2,2}],
                                                                          ConvolutionLayer[channel,{1,1}(*,"Stride"->1,"PaddingSize"->1*)]},
                            "join"->CatenateLayer[],
                            "conv"->{ConvolutionLayer[channel,{3,3},"Stride"->1,"PaddingSize"->1],Ramp,
                                    ConvolutionLayer[channel,{3,3},"Stride"->1,"PaddingSize"->1],Ramp}},
                         {NetPort["Input"]->"deconv"->"join",
                          NetPort["Shortcut"]->"join"->"conv"}];

DynamicNet=NetInitialize[Block[{c=16},
        NetGraph[<|"contract_1"->contract[c],"contract_2"->contract[2c],"contract_3"->contract[3c],
               "ubase"->{ConvolutionLayer[3c,{3,3},PaddingSize->{1,1}],Ramp,
                        ConvolutionLayer[3c,{3,3},PaddingSize->{1,1}],Ramp,DropoutLayer[0.5]},
               "expand_3"->expand[3c],"expand_2"->expand[2c],"expand_1"->expand[c],
               "latent"->{ReshapeLayer[{dLatent,5,10}],DeconvolutionLayer[dLatent,{2,2},"Stride"->{2,2}],Ramp,
                         DeconvolutionLayer[dLatent,{2,2},"Stride"->{2,2}],Ramp,
                         DeconvolutionLayer[dLatent,{2,2},"Stride"->{2,2}]},
               "cate"->CatenateLayer[],
               "output"->{ConvolutionLayer[1,{1,1}],Ramp},
               "dynamic"->DynamicNet1,
               "staticnet"->StaticNet|>,
       {NetPort["Input"]->"dynamic"->"cate",
        NetPort["Ele"]->"staticnet"->NetPort["dynamic","Static0"],
       NetPort["Latent"]->"latent"->"cate"->"contract_1",
        NetPort["contract_1","Pooling"]->"contract_2",
        NetPort["contract_2","Pooling"]->"contract_3",
        NetPort["contract_3","Pooling"]->"ubase"->NetPort["expand_3","Input"],
        NetPort["contract_3","Shortcut"]->NetPort["expand_3","Shortcut"],
        NetPort["expand_3","Output"]->NetPort["expand_2","Input"],
        NetPort["contract_2","Shortcut"]->NetPort["expand_2","Shortcut"],
        NetPort["expand_2","Output"]->NetPort["expand_1","Input"],
        NetPort["contract_1","Shortcut"]->NetPort["expand_1","Shortcut"],
        "expand_1"->"output"},
      "Input"->{15,3,59,123},
      "Latent"->dLatent*5*10]]]


EncoderNet=NetInitialize[NetGraph[<|"seq"->{ConvolutionLayer[cEncoder,{3,3}],ParametricRampLayer[],
                                          ConvolutionLayer[cEncoder,{3,3}],ParametricRampLayer[],PoolingLayer[2,2],
                                          ConvolutionLayer[2cEncoder,{3,3}],ParametricRampLayer[],
                                          ConvolutionLayer[2cEncoder,{3,3}],ParametricRampLayer[],PoolingLayer[2,2],
                                          ConvolutionLayer[4cEncoder,{3,3}],ParametricRampLayer[],
                                          ConvolutionLayer[4cEncoder,{3,3}],ParametricRampLayer[],PoolingLayer[2,2],
                                          FlattenLayer[],ReplicateLayer[2]},
                                   "MeanPart"->{PartLayer[1],LinearLayer[200],Ramp,dLatent*5*10},
                                   "StdPart"->{PartLayer[2],LinearLayer[200],Ramp,dLatent*5*10}|>,
        {NetPort["Input"]->"seq","seq"->"MeanPart"->NetPort["mn"],"seq"->"StdPart"->NetPort["sd"]},
        "Input"->{1,40,80}]]

DiscriminatorNet = NetChain[{
    ReshapeLayer[{1,40,80}],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    FlattenLayer[],BatchNormalizationLayer[],100,Ramp,BatchNormalizationLayer[], 1, ElementwiseLayer["HardSigmoid"]},
  "Input" -> dOutput[[2;;]]];

LatentLoss = NetGraph[{ElementwiseLayer[-0.5*# &], TotalLayer[], ElementwiseLayer[-(#^2) &],
   ElementwiseLayer[-Exp[2*#] &], ElementwiseLayer[1 + 2*# &], SummationLayer[]},
  {NetPort["mn"] -> 3,NetPort["sd"] -> 4,NetPort["sd"] -> 5, {3, 4, 5} -> 2 -> 6 -> 1}];

Reparameterization=NetGraph[<|"LatentLoss"->LatentLoss,
                                   "expSd"->ElementwiseLayer[Exp[#]&],
                                   "time"->ThreadingLayer[Times],
                                   "add"->ThreadingLayer[Plus]|>,
         {NetPort["mn"]->NetPort["LatentLoss","mn"],
          NetPort["sd"]->NetPort["LatentLoss","sd"],
          "LatentLoss"->NetPort["KL_Loss"],
          NetPort["sd"]->"expSd"->"time",
          NetPort["Random"]->"time"->"add",
          NetPort["mn"]->"add"}];

d = NetInsertSharedArrays[DynamicNet, "DynamicNet1/"];
e = NetInsertSharedArrays[EncoderNet,"Encoder1/"];

BicycleNet=NetInitialize[NetGraph[<|"VAE_DynamicNet"->d,
                "LR_DynamicNet"->d,
                                "VAE_Cate"->CatenateLayer[],
                                "VAE_Discriminator"->NetMapOperator[DiscriminatorNet],
                                "VAE_Reshape"->ReshapeLayer[{2}],
                                "VAE_Scale" -> ConstantTimesLayer["Scaling" -> {-1, 1}],
                                "VAE_True"->PartLayer[1],
                                "VAE_False"->PartLayer[2],
                                "VAE_MSE"->MeanSquaredLossLayer[],
                                "VAE_Encoder"->e,
                                "Reparameterization"->Reparameterization,
                                "LR_Cate"->CatenateLayer[],
                                "LR_Discriminator"->NetMapOperator[DiscriminatorNet],
                                "LR_Reshape"->ReshapeLayer[{2}],
                                "LR_Scale" -> ConstantTimesLayer["Scaling" -> {-1, 1}],
                                "LR_True"->PartLayer[1],
                                "LR_False"->PartLayer[2],
                                "LR_Encoder"->e ,
                                "LR_MSE"->MeanSquaredLossLayer[]|>,
        {NetPort["Dynamic"]->NetPort["VAE_DynamicNet","Input"],
         NetPort["VAE_DynamicNet","Output"]->"VAE_Cate",
         NetPort["Po_0"]->"VAE_Cate"->"VAE_Discriminator"->"VAE_Reshape"->"VAE_Scale",
         "VAE_Scale"->"VAE_True"->NetPort["VAE_True_Loss"],
         "VAE_Scale"->"VAE_False"->NetPort["VAE_Fake_Loss"],
         NetPort["Po_0"]->"VAE_MSE",
         NetPort["VAE_DynamicNet","Output"]->"VAE_MSE"->NetPort["VAE_MSE_Loss"],
         NetPort["Po_0"]->NetPort["VAE_Encoder","Input"],
         NetPort["VAE_Encoder","mn"]->NetPort["Reparameterization","mn"],
         NetPort["VAE_Encoder","sd"]->NetPort["Reparameterization","sd"],
         NetPort["VAE_Random"]->NetPort["Reparameterization","Random"],
         NetPort["Reparameterization","Output"]->NetPort["VAE_DynamicNet","Latent"],

         NetPort["Dynamic"]->NetPort["LR_DynamicNet","Input"],
         NetPort["LR_Random"]->NetPort["LR_DynamicNet","Latent"],
         NetPort["LR_DynamicNet","Output"]->"LR_Cate",
         NetPort["Po_0"]->"LR_Cate"->"LR_Discriminator"->"LR_Reshape"->"LR_Scale",
         "LR_Scale"->"LR_True"->NetPort["LR_True_Loss"],
         "LR_Scale"->"LR_False"->NetPort["LR_Fake_Loss"],

         NetPort["LR_DynamicNet","Output"]->"LR_Encoder",
         NetPort["LR_Encoder","mn"]->"LR_MSE",
         NetPort["LR_Random"]->"LR_MSE"->NetPort["LR_MSE_Loss"]},
        "Dynamic"->{15,3,59,123},
        "Po_0"->{1,40,80},
        "VAE_Random"->dLatent*5*10,
        "LR_Random"->dLatent*5*10]]

seq=Length[p];
batch=32;

trained=NetTrain[BicycleNet,
   {Function[Block[{select},
        select=RandomSample[Range[seq],batch];
        <|"Dynamic"->dynamics[[select]],
          "Po_0"->p[[select]],
          "Ele"->Table[ele,batch],
          "LR_Random"->Table[RandomReal[NormalDistribution[0,1],dLatent*5*10],batch],
          "VAE_Random"->Table[RandomReal[NormalDistribution[0,1],dLatent*5*10],batch]|>]],
       "RoundLength" -> seq},
   LossFunction ->{"VAE_Fake_Loss"->Scaled[1],"VAE_True_Loss"->Scaled[1],"VAE_MSE_Loss"->Scaled[-100],"KL_Loss"->Scaled[-1],
                    "LR_Fake_Loss"->Scaled[1],"LR_True_Loss"->Scaled[1],"LR_MSE_Loss"->Scaled[-1]},
    TrainingUpdateSchedule -> {"VAE_Discriminator","LR_Discriminator","VAE_Encoder","VAE_DynamicNet"|"LR_DynamicNet"},
    LearningRateMultipliers -> {"VAE_Discriminator" -> 1, "LR_Discriminator"->1,
                                "VAE_Encoder"->-10,
                                "LR_Encoder"->0,
                                "VAE_DynamicNet"|"LR_DynamicNet"->-1,
                                "VAE_Scale"->0,"LR_Scale"->0},
    BatchSize -> batch,
    TargetDevice->"GPU",
    MaxTrainingRounds->500,
    Method -> {"ADAM", "Beta1" -> 0.5, "LearningRate" -> 10^-4, "WeightClipping" -> {"VAE_Discriminator"->.5*10^-1,"LR_Discriminator"->.5*10^-1}}];
~
~
~
