cStatic=4;
cDynamic=32;
cEncoder=16;

dStatic={1,301,621};
dDynamic={15,3,59,123};
dLatent=4;
dOutput={1,40,80};

contract[channel_,crop_:{{1,1},{1,1}}]:=NetGraph[{"conv"->{ConvolutionLayer[channel,{3,3},"Stride"->1,"PaddingSize"->1],BatchNormalizationLayer[],Ramp,
                                                           ConvolutionLayer[channel,{3,3},"Stride"->1,"PaddingSize"->1],BatchNormalizationLayer[],Ramp},
                                   "pooling"->PoolingLayer[2,2],
                                   "cropping"->PartLayer[{;;,crop[[1,1]];;-crop[[1,-1]],crop[[2,1]];;-crop[[2,-1]]}]},
                         {NetPort["Input"]->"conv"->"pooling"->NetPort["Pooling"],"conv"->"cropping"->NetPort["Shortcut"]}];

expand[channel_]:=NetGraph[{"deconv"->{DeconvolutionLayer[channel,{2,2},"Stride"->{2,2}],Ramp},
                            "join"->CatenateLayer[],
                            "conv"->{ConvolutionLayer[channel,{3,3},"Stride"->1,"PaddingSize"->1],BatchNormalizationLayer[],Ramp,
                                    ConvolutionLayer[channel,{3,3},"Stride"->1,"PaddingSize"->1],BatchNormalizationLayer[],Ramp}},
                         {NetPort["Input"]->"deconv"->"join",
                          NetPort["Shortcut"]->"join"->"conv"}];

DynamicNet=Block[{c=16},
	NetGraph[<|"contract_1"->contract[c],"contract_2"->contract[2c],"contract_3"->contract[3c],
               "ubase"->{ConvolutionLayer[3c,{3,3},PaddingSize->{1,1}],Ramp,
                        ConvolutionLayer[3c,{3,3},PaddingSize->{1,1}],Ramp,DropoutLayer[0.5]},
               "expand_3"->expand[3c],"expand_2"->expand[2c],"expand_1"->expand[c],
               "latent1"->{ReplicateLayer[{40,80}],TransposeLayer[{2,3,1}]},
               "latent2"->{ReplicateLayer[{20,40}],TransposeLayer[{2,3,1}]},
               "latent3"->{ReplicateLayer[{10,20}],TransposeLayer[{2,3,1}]},
               "latent4"->{ReplicateLayer[{5,10}],TransposeLayer[{2,3,1}]},
               "cate1"->CatenateLayer[],
               "cate2"->CatenateLayer[],
               "cate3"->CatenateLayer[],
               "cate4"->CatenateLayer[],
               "output"->{ConvolutionLayer[1,{1,1}],BatchNormalizationLayer[]},
               "thread"->ThreadingLayer[Plus],
               "final"->Ramp|>,
       {NetPort["Latent1"]->"latent1"->"cate1",
        NetPort["Input"]->"cate1"->"contract_1",
        NetPort["contract_1","Pooling"]->"cate2",
        NetPort["Latent2"]->"latent2"->"cate2"->"contract_2",
        NetPort["contract_2","Pooling"]->"cate3",
        NetPort["Latent3"]->"latent3"->"cate3"->"contract_3",
        NetPort["contract_3","Pooling"]->"cate4",
        NetPort["Latent4"]->"latent4"->"cate4"->"ubase",
        "ubase"->NetPort["expand_3","Input"],
        NetPort["contract_3","Shortcut"]->NetPort["expand_3","Shortcut"],
        NetPort["expand_3","Output"]->NetPort["expand_2","Input"],
        NetPort["contract_2","Shortcut"]->NetPort["expand_2","Shortcut"],
        NetPort["expand_2","Output"]->NetPort["expand_1","Input"],
        NetPort["contract_1","Shortcut"]->NetPort["expand_1","Shortcut"],
        "expand_1"->"output"->"thread",
        NetPort["Input"]->"thread"->"final"->NetPort["Output"]},
      "Input"->{1,40,80},
      "Latent1"->dLatent,
      "Latent2"->2dLatent,
      "Latent3"->3dLatent,
      "Latent4"->4dLatent]]
      
EncoderNet=NetGraph[<|"seq"->{ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],
				          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],PoolingLayer[2,2],
				          ConvolutionLayer[2cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],
				          ConvolutionLayer[2cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],PoolingLayer[2,2],
				          ConvolutionLayer[4cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],
				          ConvolutionLayer[4cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],PoolingLayer[2,2],
				          FlattenLayer[],ReplicateLayer[2]},
				   "MeanPart"->{PartLayer[1],BatchNormalizationLayer[],LinearLayer[200],Ramp,dLatent*10},
				   "StdPart"->{PartLayer[2],BatchNormalizationLayer[],LinearLayer[200],Ramp,dLatent*10}|>,
	{NetPort["Input"]->"seq","seq"->"MeanPart"->NetPort["mn"],"seq"->"StdPart"->NetPort["sd"]},
	"Input"->{1,40,80}];

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

VAEDynamicNet = NetInsertSharedArrays[DynamicNet, "DynamicNet/"];
LRDynamicNet =  NetInsertSharedArrays[DynamicNet, "DynamicNet/"];

VAEEncoder= NetInsertSharedArrays[EncoderNet,"Encoder/"];
LREncoder = NetInsertSharedArrays[EncoderNet,"Encoder/"];

BicycleNet=NetInitialize[NetGraph[<|"VAE_DynamicNet"->VAEDynamicNet,
                "LR_DynamicNet"->LRDynamicNet,
				"VAE_Cate"->CatenateLayer[],
				"VAE_Discriminator"->NetMapOperator[DiscriminatorNet],
				"VAE_Reshape"->ReshapeLayer[{2}],
				"VAE_Scale" -> ConstantTimesLayer["Scaling" -> {-1, 1}],
				"VAE_True"->PartLayer[1],
				"VAE_False"->PartLayer[2],
				"VAE_MSE"->MeanSquaredLossLayer[],
				"VAE_Encoder"->VAEEncoder,
				"Reparameterization"->Reparameterization,
				"LR_Cate"->CatenateLayer[],
				"LR_Discriminator"->NetMapOperator[DiscriminatorNet],
				"LR_Reshape"->ReshapeLayer[{2}],
				"LR_Scale" -> ConstantTimesLayer["Scaling" -> {-1, 1}],
				"LR_True"->PartLayer[1],
				"LR_False"->PartLayer[2],
				"LR_Encoder"->LREncoder,
				"LR_MSE"->MeanSquaredLossLayer[],
				
				"PartVAE1"->PartLayer[1;;dLatent],
				"PartVAE2"->PartLayer[(dLatent+1);;3dLatent],
				"PartVAE3"->PartLayer[(3dLatent+1);;6dLatent],
				"PartVAE4"->PartLayer[(6dLatent+1);;10dLatent],
				
				"PartLR1"->PartLayer[1;;dLatent],
				"PartLR2"->PartLayer[(dLatent+1);;3dLatent],
				"PartLR3"->PartLayer[(3dLatent+1);;6dLatent],
				"PartLR4"->PartLayer[(6dLatent+1);;10dLatent]|>,
	{NetPort["Ps_0"]->NetPort["VAE_DynamicNet","Input"],
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
	 NetPort["Reparameterization","Output"]->"PartVAE1"->NetPort["VAE_DynamicNet","Latent1"],
	 NetPort["Reparameterization","Output"]->"PartVAE2"->NetPort["VAE_DynamicNet","Latent2"],
	 NetPort["Reparameterization","Output"]->"PartVAE3"->NetPort["VAE_DynamicNet","Latent3"],
	 NetPort["Reparameterization","Output"]->"PartVAE4"->NetPort["VAE_DynamicNet","Latent4"],
	 
	 NetPort["Ps_0"]->NetPort["LR_DynamicNet","Input"],
	 NetPort["LR_Random"]->"PartLR1"->NetPort["LR_DynamicNet","Latent1"],
	 NetPort["LR_Random"]->"PartLR2"->NetPort["LR_DynamicNet","Latent2"],
	 NetPort["LR_Random"]->"PartLR3"->NetPort["LR_DynamicNet","Latent3"],
	 NetPort["LR_Random"]->"PartLR4"->NetPort["LR_DynamicNet","Latent4"],
	 NetPort["LR_DynamicNet","Output"]->"LR_Cate",
	 NetPort["Po_0"]->"LR_Cate"->"LR_Discriminator"->"LR_Reshape"->"LR_Scale",
	 "LR_Scale"->"LR_True"->NetPort["LR_True_Loss"],
	 "LR_Scale"->"LR_False"->NetPort["LR_Fake_Loss"],
	 
	 NetPort["LR_DynamicNet","Output"]->"LR_Encoder",
	 NetPort["LR_Encoder","mn"]->"LR_MSE",
	 NetPort["LR_Random"]->"LR_MSE"->NetPort["LR_MSE_Loss"]},
	"Ps_0"->{1,40,80},
	"Po_0"->{1,40,80},
	"VAE_Random"->dLatent*10,
	"LR_Random"->dLatent*10]]
  
  
months=months[[1;;-25]];
validation=Block[{select=90,day},
  day=Table[RandomSample[Range[Length[p[[i]]]],select],{i,-24,-1}];
  Flatten[Table[<|"Ps_0"->ps[[i,day[[i,j]]]],"Po_0"->p[[i,day[[i,j]]]]|>,{j,select},{i,-24,-1}]]];
corrR=Block[{obser,simuR},
        obser=validation[[;;,"Po_0"]][[;;,1]];
        simuR=validation[[;;,"Ps_0"]][[;;,1]];
        corr=Quiet[Table[Correlation[simuR[[;;,i,j]],obser[[;;,i,j]]],{i,40},{j,80}]];
        Mean[Select[Flatten[corr],NumberQ]]];


Report[net_]:=Block[{obser,simu,snet,corr,snet2,simu2,corr2},
        obser=validation[[;;,"Po_0"]][[;;,1]];
        snet=NetTake[net,"LR_DynamicNet"];
        snet2=NetTake[net,"VAE_DynamicNet"];
        simu=Table[snet[<|"Ps_0"->validation[[i,"Ps_0"]],"LR_Random"->RandomReal[NormalDistribution[0,1],dLatent*10]|>, TargetDevice->"GPU"],{i,Length[validation]}][[;;,1]];
        simu2=Table[snet2[<|"Ps_0"->validation[[i,"Ps_0"]],"VAE_Random"->RandomReal[NormalDistribution[0,1],dLatent*10],"Po_0"->validation[[i,"Po_0"]]|>, TargetDevice->"GPU"]["Output"],{i,Length[validation]}][[;;,1]];
        corr=Quiet[Table[Correlation[simu[[;;,i,j]],obser[[;;,i,j]]],{i,40},{j,80}]];
        corr2=Quiet[Table[Correlation[simu2[[;;,i,j]],obser[[;;,i,j]]],{i,40},{j,80}]];
        Print[{Mean[Select[Flatten[corr],NumberQ]],Mean[Select[Flatten[corr2],NumberQ]],corrR}];];
  
  
batch=128;
NetTrain[BicycleNet,
   {Function[Block[{Nmonth=8,m,d},
        m=RandomSample[Range[Length[months]],Nmonth];
        d=Map[RandomSample[Range[Length[p[[#]]]],batch/Nmonth]&,m];
        <|"Ps_0"->Flatten[Table[Table[ps[[m[[i]]]][[d[[i,j]]]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "Po_0"->Flatten[Table[Table[p[[m[[i]]]][[d[[i,j]]]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "LR_Random"->Table[RandomReal[NormalDistribution[0,1],dLatent*10],batch],
          "VAE_Random"->Table[RandomReal[NormalDistribution[0,1],dLatent*10],batch]|>]],
       "RoundLength" -> Total[Map[Length[#]&,p]]},
    LossFunction ->{"VAE_Fake_Loss"->Scaled[1],"VAE_True_Loss"->Scaled[1],"VAE_MSE_Loss"->Scaled[-10],"KL_Loss"->Scaled[-1],
                    "LR_Fake_Loss"->Scaled[1],"LR_True_Loss"->Scaled[1],"LR_MSE_Loss"->Scaled[-1]},
    TrainingUpdateSchedule -> {"VAE_Discriminator","LR_Discriminator","VAE_Encoder","LR_Encoder","VAE_DynamicNet","LR_DynamicNet"},
    LearningRateMultipliers -> {"VAE_Discriminator" -> 1, "LR_Discriminator"->1,
                                "VAE_Encoder"->-5,"LR_Encoder"->-5,
                                "VAE_DynamicNet"->-1,"LR_DynamicNet"->-1,
                                "VAE_Scale"->0,"LR_Scale"->0},
    BatchSize -> batch,
    TargetDevice->"GPU",
    MaxTrainingRounds->500,
    Method -> {"ADAM", "Beta1" -> 0.5, "LearningRate" -> 10^-4, "WeightClipping" -> {"VAE_Discriminator"->10^-1,"LR_Discriminator"->10^-1}},
    TrainingProgressReporting->{{Function@Report[#Net], "Interval" -> Quantity[50, "Batches"]},"Print"}];
