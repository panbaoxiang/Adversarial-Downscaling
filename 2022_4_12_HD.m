~
~
~
~
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         171,8         Bot
SetDirectory["/data/home/scy0446/run/Data"];
months=DateRange[{1979,1,1},{1995,10,1},"Month"];
p=Block[{tempt=Table[Block[{},Print[months[[i]]];Import["N_"<>DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["data"]],{i,Length[months]}]},
        tempt=tempt[[;;,-1]];tempt/.List->Join];
ele=NumericArray[{Log[Normal[Import["/data/home/scy0446/run/Data/CONUS/Elevation.mx"]["data"]]+1.]},"Real32"];
dmeanvar=Import["/data/home/scy0446/run/Data/CONUS/meanvar_dynamics.mx"]["data"];
SetDirectory["/data/home/scy0446/run/Data/CONUS"];
var={"pressfc","prmsl","pwat","q500","q850","q925","tmpsfc","u200","u850","v200","v850","z1000","z200","z500","z850"};
{dlat,dlon}=Block[{tempt=Import[DateString[{1979,1,1},{"D_CONUS_","Year","_","Month",".mx"}]]},{tempt["lat"],tempt["lon"]}];
dynamics=Block[{tempt=Table[Block[{},Print[months[[i]]];Import[DateString[months[[i]],{"D_CONUS_","Year","_","Month",".mx"}]]["data"]],{i,Length[months]}]},
 Flatten[Map[Table[#[[;;,((i-1)*3+1);;(i*3)]],{i,Dimensions[#][[2]]/3}]&,tempt]]];
Print["Loaded"];

cStatic=4;
cDynamic=32;
cEncoder=16;

dP={1,301,621};
dDynamic={15,3,59,123};
dLatent=8;

StaticNet=Block[{c=cStatic},
        NetGraph[
 <|"resize"->{ResizeLayer[{320,640}],ConvolutionLayer[c,{1,1}],NormalizationLayer[], ParametricRampLayer[]},
   "down1"->{Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],NormalizationLayer[],ParametricRampLayer[]},3],PoolingLayer[2,2]},
   "down2"->{Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],NormalizationLayer[],ParametricRampLayer[]},3],PoolingLayer[2,2]},
   "down3"->{Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],NormalizationLayer[],ParametricRampLayer[]},3],PoolingLayer[2,2]}|>,
        {NetPort["Static"]->"resize"->NetPort["Down0"],
        "resize"->"down1"->NetPort["Down1"],
        "down1"->"down2"->NetPort["Down2"],
        "down2"->"down3"->NetPort["Down3"]},
"Static"->dP]];

res[c_]:=NetGraph[<|"long"->Flatten[Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],NormalizationLayer[],Ramp},2]][[1;;-2]],
          "plus"->TotalLayer[],
          "short"->ConvolutionLayer[c,{1,1}]|>,
   {NetPort["Input"]->"long"->"plus",NetPort["Input"]->"short"->"plus"}]

upres[c_,size_]:=NetGraph[<|"long"->{NormalizationLayer[],Ramp,ResizeLayer[size],ConvolutionLayer[c,{3,3},"PaddingSize"->1],
                                        NormalizationLayer[],Ramp,ConvolutionLayer[c,{3,3},"PaddingSize"->1]},
          "plus"->TotalLayer[],
          "short"->{ResizeLayer[size],ConvolutionLayer[c,{1,1}]}|>,
   {NetPort["Input"]->"long"->"plus",NetPort["Input"]->"short"->"plus"}]

merge=Block[{c=cDynamic,c2=cDynamic},
  NetGraph[<|"resize"->{Table[{ConvolutionLayer[c,{2,5,5}],NormalizationLayer[],Ramp},2],FlattenLayer[1],res[c],res[c],ResizeLayer[{40,80}]},
   "cate"->CatenateLayer[],
   "up"->res[c2]|>,
   {NetPort["Dynamics"]->"resize"->"cate",
    NetPort["Static"]->"cate"->"up"},
"Dynamics"->dDynamic]];

contract[channel_,crop_:{{1,1},{1,1}}]:=NetGraph[{"conv"->res[channel],"pooling"->PoolingLayer[2,2,"Function"->Mean],
                                   "cropping"->PartLayer[{;;,crop[[1,1]];;-crop[[1,-1]],crop[[2,1]];;-crop[[2,-1]]}]},
                         {NetPort["Input"]->"conv"->"pooling"->NetPort["Pooling"],"conv"->"cropping"->NetPort["Shortcut"]}];

expand[channel_,size_]:=NetGraph[{"deconv"->upres[channel,size],
                            "join"->CatenateLayer[],
                            "conv"->res[channel/2]},
                         {NetPort["Input"]->"deconv"->"join",
                          NetPort["Shortcut"]->"join"->"conv"}];

DynamicNet=Block[{c=64},
         NetGraph[<|"contract_1"->contract[c],"contract_2"->contract[c*2],"contract_3"->contract[c*4],
               "ubase"->res[c*4], "merge"->merge,
                           "expand_3"->expand[c*4,{10,20}],"expand_2"->expand[c*2,{20,40}],"expand_1"->expand[c,{40,80}],"expand_0"->expand[c,{40,80}],
                           "expand_-1"->expand[c,{80,160}],"expand_-2"->expand[c,{160,320}],"expand_-3"->expand[c,{320,640}],
                           "output"->{ResizeLayer[dP[[2;;]]],ConvolutionLayer[1,{1,1}],NormalizationLayer[],Ramp},
                           "thread"->ThreadingLayer[Plus],
                           "latent"->{res[16],res[32],res[64],res[128],res[256]},
                           "static"->StaticNet,
                           "thread2"->ThreadingLayer[Times]|>,
   {"merge"->NetPort["contract_1","Input"],
   NetPort["contract_1","Pooling"]->"contract_2",
   NetPort["contract_2","Pooling"]->"contract_3",
   NetPort["contract_3","Pooling"]->"thread",
   NetPort["Latent"]->"latent"->"thread2"->"thread"->"ubase"->NetPort["expand_3","Input"],
   NetPort["Switch"]->"thread2",
   NetPort["contract_3","Shortcut"]->NetPort["expand_3","Shortcut"],
   NetPort["expand_3","Output"]->NetPort["expand_2","Input"],
   NetPort["contract_2","Shortcut"]->NetPort["expand_2","Shortcut"],
   NetPort["expand_2","Output"]->NetPort["expand_1","Input"],
   NetPort["contract_1","Shortcut"]->NetPort["expand_1","Shortcut"],
   NetPort["expand_1","Output"]->NetPort["expand_0","Input"],
   "merge"->NetPort["expand_0","Shortcut"],
   NetPort["expand_0","Output"]->NetPort["expand_-1","Input"],
   NetPort["expand_-1","Output"]->NetPort["expand_-2","Input"],
   NetPort["expand_-2","Output"]->NetPort["expand_-3","Input"],
   NetPort["Static"]->"static",
   NetPort["static","Down3"]->NetPort["merge","Static"],
   NetPort["static","Down2"]->NetPort["expand_-1","Shortcut"],
   NetPort["static","Down1"]->NetPort["expand_-2","Shortcut"],
   NetPort["static","Down0"]->NetPort["expand_-3","Shortcut"],
   "expand_-3"->"output"}]];


EncoderNet=NetGraph[<|"long"->{ResizeLayer[{320,640}],res[64],PoolingLayer[2,2],
                                                     res[64],PoolingLayer[2,2],res[64],PoolingLayer[2,2],
                                                     res[64],PoolingLayer[2,2],res[64],PoolingLayer[2,2],res[64],PoolingLayer[2,2]},
                     "part1"->{ConvolutionLayer[64,{1,1}],NormalizationLayer[],Ramp,ConvolutionLayer[dLatent,{1,1}]},
                     "part2"->{ConvolutionLayer[64,{1,1}],NormalizationLayer[],Ramp,ConvolutionLayer[dLatent,{1,1}]}|>,
 {NetPort["Input"]->"long"->"part1"->NetPort["mn"],"long"->"part2"->NetPort["sd"]},"Input"->dP]

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

DiscriminatorNet = NetChain[{
    ReshapeLayer[dP], PoolingLayer[2,2],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    FlattenLayer[],BatchNormalizationLayer[],200,Ramp, 1, ElementwiseLayer["HardSigmoid"]},
  "Input" -> dP[[2;;]]];

dynamic=NetInsertSharedArrays[DynamicNet,"dshare"];
encoder=NetInsertSharedArrays[EncoderNet,"eshare"];
sigma=6.0*10^-2;

CVAEGAN=NetInitialize[NetGraph[<|"dynamic"->dynamic,
                                                         "encoder"->encoder,
                                                         "reparameterization"->Reparameterization,
                                                         "mse"->MeanSquaredLossLayer[],
                                                         "dynamic2"->dynamic,
                                                         "encoder2"->encoder,
                                                         "mseLatent"->MeanSquaredLossLayer[],
                                                         "mse2"->MeanSquaredLossLayer[],
                                                         "cate"->CatenateLayer[],
                                                         "discriminate"->NetMapOperator[DiscriminatorNet],
                                                         "scale" ->{FlattenLayer[], ConstantTimesLayer["Scaling" -> {-1, 1}]},
                                                         "true"->PartLayer[1],
                                             "false"->PartLayer[2],
                                             "cate2"->CatenateLayer[],
                                                         "discriminate2"->NetMapOperator[DiscriminatorNet],
                                                         "scale2" ->{FlattenLayer[], ConstantTimesLayer["Scaling" -> {-1, 1}]},
                                                         "true2"->PartLayer[1],
                                             "false2"->PartLayer[2],
                                             "hinge"->ElementwiseLayer[(Max[#,sigma]-sigma)&]|>,
 {NetPort["P"]->"encoder",
  NetPort["encoder","mn"]->NetPort["reparameterization","mn"],
  NetPort["encoder","sd"]->NetPort["reparameterization","sd"],
  NetPort["reparameterization","Output"]->NetPort["dynamic","Latent"],
  NetPort["P"]->"mse","dynamic"->"mse"->NetPort["Reconstruction_Loss"],
  NetPort["Random2"]->NetPort["dynamic2","Latent"],
  "dynamic2"->"mse2",NetPort["P"]->"mse2"->"hinge"->NetPort["MSE_Loss"],
  NetPort["P"]->"cate",
  "dynamic"->"cate"->"discriminate"->"scale"->"true"->NetPort["True_1"],"scale"->"false"->NetPort["False_1"],
  NetPort["P"]->"cate2",
  "dynamic2"->"cate2"->"discriminate2"->"scale2"->"true2"->NetPort["True_2"],"scale2"->"false2"->NetPort["False_2"],
  "dynamic2"->"encoder2",
  NetPort["encoder2","mn"]->"mseLatent",
  NetPort["Random2"]->"mseLatent"->NetPort["MSE_Latent"]},
  "Random2"->{dLatent,5,10}]]


length=Length[p];
sep=Round[length*0.8];
training=Table[<|"P"->p[[i]],"Dynamics"->dynamics[[i]]|>,{i,sep}];
validation=Table[<|"P"->p[[i]],"Dynamics"->dynamics[[i]]|>,{i,sep,length}];

GlobeLoss=Infinity;
Report[net_]:=Block[{nnet,select=RandomSample[Range[Length[validation]],10],m=10,obser,simu,msimu,corr},
  nnet=NetExtract[net,"dynamic2"];
  obser=Normal[validation[[select,"P"]]];
  simu=Table[nnet[<|"Latent"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],m],
                   "Dynamics"->Table[validation[[i,"Dynamics"]],m],
                   "Switch"->Table[1.,{m},{256},{5},{10}],
                   "Static"->Table[ele,m]|>,TargetDevice->"GPU"],{i,select}];
  msimu=Map[Mean,Normal[simu]];
  mse=Mean[Flatten[(msimu-obser)^2]];
  Print[{GlobeLoss,mse}];
  corr=Quiet[Table[Correlation[msimu[[;;,1,i,j]],obser[[;;,1,i,j]]],{i,Dimensions[obser][[3]]},{j,Dimensions[obser][[4]]}]];
  Print[Mean[Select[Flatten[corr],NumberQ]]];
  If[mse<GlobeLoss,
   Block[{},Set[GlobeLoss,mse];Export["/data/home/scy0446/run/Code/BICYCLEGAN_HD_"<>ToString[dLatent]<>".mx",net]]]];

(*
batch=8;
trained=NetTrain[CVAEGAN,
       {Function[Block[{select},
        select=RandomSample[Range[sep],batch];
        <|"Dynamics"->training[[select,"Dynamics"]],
          "Static"->Table[ele,batch],
          "P"->training[[select,"P"]],
          "Switch"->Table[1.,{batch},{256},{5},{10}],
          "Random"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],batch],
          "Random2"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],batch]|>]],
       "RoundLength" -> sep},
        MaxTrainingRounds->100,
        LossFunction ->{"MSE_Loss"->Scaled[-1],"KL_Loss"->Scaled[-10^-4],"Reconstruction_Loss"->Scaled[-1],"True_1"->Scaled[.1],"False_1"->Scaled[.1],"True_2"->Scaled[.1],"False_2"->Scaled[.1],"MSE_Latent"->Scaled[-10^-2]},
        LearningRateMultipliers -> {"discriminate" -> 1, "discriminate2"->1, "encoder"->-1,"encoder2"->0,"dynamic"->-1,"dynamic2"->-1,"scale"->0,"scale2"->0},
        TargetDevice->"GPU",
        Method -> {"ADAM", "Beta1" -> 0.5, "LearningRate" -> 10^-4, "WeightClipping" -> {"discriminate"->.3*10^-1,"discriminate2"->.3*10^-1}},
        BatchSize->batch,
        TrainingProgressReporting->{{Function@Report[#Net], "Interval" -> Quantity[1, "Rounds"]},"Print"}];

(*
batch=32;
trained=NetTrain[CVAEGAN,
        {Function[Block[{select},
        select=RandomSample[Range[sep],batch];
        <|"Dynamics"->training[[select,"Dynamics"]],
          "Static"->Table[ele,batch],
          "P"->training[[select,"P"]],
          "Switch"->Table[0,{batch},{256},{5},{10}],
          "Random"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],batch],
          "Random2"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],batch]|>]],
       "RoundLength" -> sep},
        MaxTrainingRounds->100,
        LossFunction ->{"MSE_Loss"->Scaled[-1],"KL_Loss"->Scaled[0],"Reconstruction_Loss"->Scaled[-1],"True_1"->Scaled[0],"False_1"->Scaled[0],"True_2"->Scaled[0],"False_2"->Scaled[0],"MSE_Latent"->Scaled[0]},
        LearningRateMultipliers -> {"discriminate" -> 0, "discriminate2"->0, "encoder"->-0,"encoder2"->0,"dynamic"->-1,"dynamic2"->-1,"scale"->0,"scale2"->0},
        TargetDevice->{"GPU",All},
        Method -> {"ADAM", "Beta1" -> 0.5, "LearningRate" -> 10^-4},
        BatchSize->batch];

batch=64;
trained=NetTrain[CVAEGAN,
       {Function[Block[{select},
        select=RandomSample[Range[sep],batch];
        <|"Dynamics"->training[[select,"Dynamics"]],
          "Static"->Table[ele,batch],
          "P"->training[[select,"P"]],
          "Switch"->Table[1.,{batch},{256},{5},{10}],
          "Random"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],batch],
          "Random2"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],batch]|>]],
       "RoundLength" -> sep},
        MaxTrainingRounds->100,
        LossFunction ->{"MSE_Loss"->Scaled[-1],"KL_Loss"->Scaled[-10^-4],"Reconstruction_Loss"->Scaled[-1],"True_1"->Scaled[.1],"False_1"->Scaled[.1],"True_2"->Scaled[.1],"False_2"->Scaled[.1],"MSE_Latent"->Scaled[-10^-2]},
        LearningRateMultipliers -> {"discriminate" -> 1, "discriminate2"->1, "encoder"->-1,"encoder2"->0,"dynamic"->-1,"dynamic2"->-1,"scale"->0,"scale2"->0},
        TargetDevice->{"GPU",All},
        Method -> {"ADAM", "Beta1" -> 0.5, "LearningRate" -> 10^-4, "GradientClipping" -> {"discriminate"->.3*10^-1,"discriminate2"->.3*10^-1}},
        BatchSize->batch,
        TrainingProgressReporting->{{Function@Report[#Net], "Interval" -> Quantity[1, "Rounds"]},"Print"}];
*)

CVAEGAN=trained;
net=NetExtract[CVAEGAN,"dynamic2"];
m=10;
l=100;
obser=validation[[;;l,"P"]];
simu=Table[net[<|"Latent"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],m],
                 "Dynamics"->Table[validation[[i,"Dynamics"]],m],
                 "Switch"->Table[1.,{m},{256},{5},{10}],
                 "Static"->Table[ele,m]|>,TargetDevice->"GPU"],{i,(*Length[validation]*)l}];
Table[Correlation[Flatten[Normal[simu[[i,o]]]],Flatten[Normal[obser[[i]]]]],{i,l},{o,m}]
Table[Correlation[Flatten[Mean[Normal[simu[[i,;;]]]]],Flatten[Normal[obser[[i]]]]],{i,l}]
Export["/data/home/scy0446/run/tempt.mx",{simu,obser}]
*)
~
~
~
