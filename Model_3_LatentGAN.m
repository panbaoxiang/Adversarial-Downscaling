cStatic=4;
cDynamic=32;
cLatent=4;

dStatic={1,301,621};
dDynamic={15,3,59,123};
dLatent=64;

StaticNet=Block[{c=cStatic},
	NetGraph[
 <|"resize"->{ResizeLayer[{320,640}],ConvolutionLayer[c,{1,1}],BatchNormalizationLayer[], ParametricRampLayer[]},
   "down1"->{Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],BatchNormalizationLayer[],ParametricRampLayer[]},3],PoolingLayer[2,2]},
   "down2"->{Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],BatchNormalizationLayer[],ParametricRampLayer[]},3],PoolingLayer[2,2]},
   "down3"->{Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],BatchNormalizationLayer[],ParametricRampLayer[]},3],PoolingLayer[2,2]}|>,
	{NetPort["Static"]->"resize"(*->NetPort["Down0"]*),
	"resize"->"down1"(*->NetPort["Down1"]*),
	"down1"->"down2"(*->NetPort["Down2"]*),
	"down2"->"down3"->NetPort["Down3"]},
"Static"->dStatic]];

res[c_]:=NetGraph[<|"l"->Flatten[Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],BatchNormalizationLayer[],Ramp},3]][[1;;-2]],
          "plus"->TotalLayer[],
          "post"->Ramp|>,
   {NetPort["Input"]->"l"->"plus",NetPort["Input"]->"plus"->"post"}]
   
DynamicNet=Block[{c=cDynamic,c2=cDynamic+cStatic+cLatent},
	NetGraph[
 <|"latent"->{LinearLayer[10*20],BatchNormalizationLayer[],ParametricRampLayer[],ReshapeLayer[{1,10,20}],
          DeconvolutionLayer[cLatent,2,2],BatchNormalizationLayer[],ParametricRampLayer[],
		  DeconvolutionLayer[cLatent,2,2],BatchNormalizationLayer[],ParametricRampLayer[]},
   "resize0"->{Table[{ConvolutionLayer[c,{2,3,3}],BatchNormalizationLayer[],ParametricRampLayer[]},2],FlattenLayer[1],ResizeLayer[{40,80}]},
   "cate0"->CatenateLayer[],
   "up0"->{res[c2],res[c2]},
   "output0"->{ConvolutionLayer[c,{1,1}],BatchNormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],BatchNormalizationLayer[],Ramp}|>,	
   {NetPort["Dynamics"]->"resize0"->"cate0",
    NetPort["Latent"]->"latent"->"cate0",
	NetPort["Static0"]->"cate0"->"up0"->"output0"->NetPort["P_0"]},
"Dynamics"->dD,
"Static0"->NetExtract[StaticNet,"Down3"],
"Latent"->dLatent]]

discriminator = NetChain[{
    ReshapeLayer[{1,40,80}],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    FlattenLayer[],BatchNormalizationLayer[],100,Ramp,BatchNormalizationLayer[], 1, ElementwiseLayer["HardSigmoid"]},
  "Input" -> {40,80}]
  
SRNet=NetInitialize[NetGraph[<|"DynamicNet"->DynamicNet,
				"StaticNet"->StaticNet,
				"Cate"->CatenateLayer[],
				"Discriminator"->NetMapOperator[discriminator],
				"Reshape"->ReshapeLayer[{2}],
				"Scale_GCM->Obser" -> ConstantTimesLayer["Scaling" -> {-1, 1},LearningRateMultipliers->0],
				"True"->PartLayer[1],
				"False"->PartLayer[2],
				"MSE"->MeanSquaredLossLayer[]|>,
	{NetPort["Dynamics"]->NetPort["DynamicNet","Dynamics"],
	 NetPort["Latent"]->NetPort["DynamicNet","Latent"],
	 NetPort["Static"]->NetPort["StaticNet","Static"],
	 NetPort["StaticNet","Down3"]->NetPort["DynamicNet","Static0"],
	 NetPort["DynamicNet","P_0"]->"Cate",
	 NetPort["Po_0"]->"Cate"->"Discriminator"->"Reshape"->"Scale_GCM->Obser",
	 "Scale_GCM->Obser"->"True"->NetPort["True_Loss"],
	 "Scale_GCM->Obser"->"False"->NetPort["Fake_Loss"],
	 NetPort["Po_0"]->"MSE",
	 NetPort["DynamicNet","P_0"]->"MSE"->NetPort["MSE_Loss"]},
	"Dynamics"->{15,3,59,123},
	"Static"->{1,301,621},
	"Po_0"->{1,40,80},
	"Latent"->dLatent]]
  
NetTrain[SRNet,
   {Function[Block[{Nmonth=8,m,d},
        m=RandomSample[Range[Length[months]],Nmonth];
        d=Map[RandomSample[Range[Length[p[[#]]]],batch/Nmonth]&,m];
        <|"Static"->Table[ele,batch],
          "Dynamics"->Flatten[Table[Table[dynamics[[m[[i]]]][[;;,(d[[i,j]]-1)*3+1;;d[[i,j]]*3]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "Po_0"->Flatten[Table[Table[p[[m[[i]]]][[d[[i,j]]]],{j,Length[d[[i]]]}],{i,Length[m]}]]|>]],
       "RoundLength" -> Total[Map[Length[#[[1]]]&,p]]},
    LossFunction ->{"Fake_Loss"->Scaled[1],"True_Loss"->Scaled[1],"MSE_Loss"->Scaled[-1]},
    TrainingUpdateSchedule -> {"Discriminator","StaticNet","DynamicNet"},
    LearningRateMultipliers -> {"Discriminator" -> 1, "StaticNet" -> -1,"DynamicNet"->-1,"Scale_GCM->Obser"->0},
    BatchSize -> 32,
    TargetDevice->"GPU",
    MaxTrainingRounds->400,
    Method -> {"ADAM", "Beta1" -> 0.5, "LearningRate" -> 10^-4, "WeightClipping" -> {"Discriminator"->10^-2}}]
