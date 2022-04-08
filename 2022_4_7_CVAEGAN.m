SetDirectory["/data/home/scy0446/run/Data"];
months=DateRange[{1979,1,1},{1992,10,1},"Month"];
p=Block[{tempt=Table[Block[{},Print[months[[i]]];Import["N_"<>DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["data"]],{i,Length[months]}]},
        tempt=tempt[[;;,1]];
        NumericArray[Flatten[Normal[tempt],1],"Real32"]];
ele=NumericArray[{Log[Normal[Import["/data/home/scy0446/run/Data/CONUS/Elevation.mx"]["data"]]+1.]},"Real32"];
ResNet=Import["/data/home/scy0446/run/Code/trained_Res.mx"];
StaticNet=NetTake[ResNet,"StaticNet"];
static=StaticNet[ele]["Down3"];
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

length=Length[p];
sep=Round[length*0.8];
training=Table[<|"P"->p[[i]],"Dynamics"->dynamics[[i]],"Static"->static|>,{i,sep}];
validation=Table[<|"P"->p[[i]],"Dynamics"->dynamics[[i]],"Static"->static,"Random"->RandomReal[NormalDistribution[0,1],{dLatent,5,10}],"Random2"->RandomReal[NormalDistribution[0,1],{dLatent,5,10}]|>,{i,sep,length}];


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
"Dynamics"->{15,3,59,123},
"Static"->{cStatic,40,80}]];

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
                           "output"->{ConvolutionLayer[1,{1,1}],NormalizationLayer[],Ramp},
                           "thread"->ThreadingLayer[Plus],
                           "latent"->{res[16],res[32],res[64],res[128],res[256]}|>,
   {"merge"->NetPort["contract_1","Input"],
   NetPort["contract_1","Pooling"]->"contract_2",
   NetPort["contract_2","Pooling"]->"contract_3",
   NetPort["contract_3","Pooling"]->"thread",
   NetPort["Latent"]->"latent"->"thread"->"ubase"->NetPort["expand_3","Input"],
   NetPort["contract_3","Shortcut"]->NetPort["expand_3","Shortcut"],
   NetPort["expand_3","Output"]->NetPort["expand_2","Input"],
   NetPort["contract_2","Shortcut"]->NetPort["expand_2","Shortcut"],
   NetPort["expand_2","Output"]->NetPort["expand_1","Input"],
   NetPort["contract_1","Shortcut"]->NetPort["expand_1","Shortcut"],
   NetPort["expand_1","Output"]->NetPort["expand_0","Input"],
   "merge"->NetPort["expand_0","Shortcut"],
   NetPort["expand_0","Output"]->"output"}]]

EncoderNet=NetGraph[<|"long"->{res[64],PoolingLayer[2,2],res[64],PoolingLayer[2,2],res[64],PoolingLayer[2,2]},
                     "part1"->{ConvolutionLayer[64,{1,1}],NormalizationLayer[],Ramp,ConvolutionLayer[dLatent,{1,1}]},
                     "part2"->{ConvolutionLayer[64,{1,1}],NormalizationLayer[],Ramp,ConvolutionLayer[dLatent,{1,1}]}|>,
 {NetPort["Input"]->"long"->"part1"->NetPort["mn"],"long"->"part2"->NetPort["sd"]},
 "Input"->{1,40,80}]

LatentLoss = NetGraph[{ElementwiseLayer[-0.5*# &], TotalLayer[], ElementwiseLayer[-(#^2) &],
   ElementwiseLayer[-Exp[2*#] &], ElementwiseLayer[1 + 2*# &], SummationLayer[]},
  {NetPort["mn"] -> 3,NetPort["sd"] -> 4,NetPort["sd"] -> 5, {3, 4, 5} -> 2 -> 6 -> 1}]

Reparameterization=NetGraph[<|"LatentLoss"->LatentLoss,
                                   "expSd"->ElementwiseLayer[Exp[#]&],
                                   "time"->ThreadingLayer[Times],
                                   "add"->ThreadingLayer[Plus]|>,
         {NetPort["mn"]->NetPort["LatentLoss","mn"],
          NetPort["sd"]->NetPort["LatentLoss","sd"],
          "LatentLoss"->NetPort["KL_Loss"],
          NetPort["sd"]->"expSd"->"time",
          NetPort["Random"]->"time"->"add",
          NetPort["mn"]->"add"}]

DiscriminatorNet = NetChain[{
    ReshapeLayer[{1,40,80}],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    FlattenLayer[],BatchNormalizationLayer[],100,Ramp, 1, ElementwiseLayer["HardSigmoid"]},
  "Input" -> dOutput[[2;;]]]

dynamic=NetInsertSharedArrays[DynamicNet,"dshare"];
CVAEGAN=NetInitialize[NetGraph[<|"dynamic"->dynamic,
                                                         "encoder"->EncoderNet,
                                                         "reparameterization"->Reparameterization,
                                                         "mse"->MeanSquaredLossLayer[],
                                                         "dynamic2"->dynamic,
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
                                             "false2"->PartLayer[2]|>,
 {NetPort["P"]->"encoder",
  NetPort["encoder","mn"]->NetPort["reparameterization","mn"],
  NetPort["encoder","sd"]->NetPort["reparameterization","sd"],
  NetPort["reparameterization","Output"]->NetPort["dynamic","Latent"],
  NetPort["P"]->"mse","dynamic"->"mse"->NetPort["Reconstruction_Loss"],
  NetPort["Random2"]->NetPort["dynamic2","Latent"],
  "dynamic2"->"mse2",NetPort["P"]->"mse2"->NetPort["MSE_Loss"],
  NetPort["P"]->"cate",
  "dynamic"->"cate"->"discriminate"->"scale"->"true"->NetPort["True_1"],"scale"->"false"->NetPort["False_1"],
  NetPort["P"]->"cate2",
  "dynamic2"->"cate2"->"discriminate2"->"scale2"->"true2"->NetPort["True_2"],"scale2"->"false2"->NetPort["False_2"]},
  "Random2"->{dLatent,5,10}]];

GlobeLoss=Infinity;
Report[net_]:=Block[{nnet,select=RandomSample[Range[Length[validation]],1000],m=10,obser,simu,msimu,corr},
  nnet=NetExtract[net,"dynamic2"];
  obser=Normal[validation[[select,"P"]]];
  simu=Table[nnet[<|"Latent"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],m],
                   "Dynamics"->Table[validation[[i,"Dynamics"]],m],
                   "Static"->Table[static,m]|>,TargetDevice->"GPU"],{i,select}];
  msimu=Map[Mean,Normal[simu]];
  mse=Mean[Flatten[(msimu-obser)^2]];
  Print[{GlobeLoss,mse}];
  If[mse<GlobeLoss,
   Block[{corr},Set[GlobeLoss,mse];
            corr=Quiet[Table[Correlation[msimu[[;;,1,i,j]],obser[[;;,1,i,j]]],{i,40},{j,80}]];
            Print[Mean[Select[Flatten[corr],NumberQ]]];
           Export["/data/home/scy0446/run/Code/CVAEGAN.mx",net]]]];

(*
batch=128;
trained=NetTrain[CVAEGAN,
        {Function[Block[{select},
        select=RandomSample[Range[sep],batch];
        <|"Dynamics"->training[[select,"Dynamics"]],
          "Static"->Table[static,batch],
          "P"->training[[select,"P"]],
          "Random"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],batch],
          "Random2"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],batch]|>]],
       "RoundLength" -> sep},
       (*ValidationSet->validation,*)
        MaxTrainingRounds->100,
        LossFunction ->{"MSE_Loss"->Scaled[-1],"KL_Loss"->Scaled[-10^-4],"Reconstruction_Loss"->Scaled[-1],"True_1"->Scaled[.1],"False_1"->Scaled[.1],"True_2"->Scaled[.1],"False_2"->Scaled[.1]},
        (*TrainingUpdateSchedule -> {"discriminate","discriminate2","encoder","dynamic","dynamic2"},*)
        LearningRateMultipliers -> {"discriminate" -> 1, "discriminate2"->1,"encoder"->-1,"dynamic"->-1,"dynamic2"->-1,"scale"->0,"scale2"->0},
        TargetDevice->"GPU",
        BatchSize->batch,
        Method -> {"ADAM", "Beta1" -> 0.5, "LearningRate" -> 10^-5, "WeightClipping" -> {"discriminate"->7*10^-2,"discriminate2"->7*10^-2}},
        TrainingProgressReporting->{{Function@Report[#Net], "Interval" -> Quantity[1, "Rounds"]},"Print"}];

CVAE=trained;
net=NetExtract[CVAE,"dynamic2"];
m=50;
l=80;
obser=validation[[;;l,"P"]];
simu=Table[net[<|"Latent"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],m],
                 "Dynamics"->Table[validation[[i,"Dynamics"]],m],
                 "Static"->Table[static,m]|>,TargetDevice->"GPU"],{i,(*Length[validation]*)l}];
Table[Correlation[Flatten[Normal[simu[[i,o]]]],Flatten[Normal[obser[[i]]]]],{i,l},{o,m}]
Table[Correlation[Flatten[Mean[Normal[simu[[i,;;]]]]],Flatten[Normal[obser[[i]]]]],{i,l}]
Export["/data/home/scy0446/run/tempt.mx",{simu,obser}]
*)
~
~
~
