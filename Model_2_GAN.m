                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                72,1-8        All
(*import data*)
SetDirectory["/data/home/scy0446/run/Data"];
months=DateRange[{1979,1,1},{2006,12,1},"Month"];
(*
{plat,plon}=Block[{tempt=Import[DateString[months[[1]],{"CONUS_","Year","Month",".mx"}]]},{tempt["lat"],tempt["lon"]}];
p=Table[Block[{tempt,np2,np1,np0},
  Print[months[[i]]];
  tempt=Normal[Import[DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["data"]];
  If[Min[tempt]<0,Block[{position},
    Print["negative"];
    Print[months[[i]]];
    position=Position[tempt,_?(#<0&)];
    Map[Set[tempt[[#[[1]],#[[2]],#[[3]]]],0.]&,position];
    Export[DateString[months[[i]],{"CONUS_","Year","Month",".mx"}],
      <|"data"->NumericArray[tempt,"Real32"],
        "lat"->Import[DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["lat"],
        "lon"->Import[DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["lon"],
        "date"->Import[DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["date"],
        "description"->Import[DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["description"]|>];]];
  tempt=Log[tempt+1.];
  {np2,np1,np0}=Block[{tempt2=ArrayResample[tempt,{Length[tempt],320,640}]},Table[Mean[Mean[Table[tempt2[[;;,i;;-1;;q,j;;-1;;q]],{i,q},{j,q}]]],{q,{2,4,8}}]];
  Map[NumericArray[Map[List,#],"Real32"]&,{np0,np1,np2,tempt}]],{i,Length[months]}];

Table[Block[{},
  Print[months[[i]]];
  Export["N_"<>DateString[months[[i]],{"CONUS_","Year","Month",".mx"}],<|"data"->p[[i]],"lat"->plat,"lon"->plon,"description"->"log(p in mm/3h +1)"|>];
DeleteFile[DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]],{i,Length[months]}]
*)
p=Table[Block[{},Print[months[[i]]];Import["N_"<>DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["data"][[1]]],{i,Length[months]}];

ele=NumericArray[{Log[Normal[Import["/data/home/scy0446/run/Data/CONUS/Elevation.mx"]["data"]]+1.]},"Real32"];
dmeanvar=Import["/data/home/scy0446/run/Data/CONUS/meanvar_dynamics.mx"]["data"];
SetDirectory["/data/home/scy0446/run/Data/CONUS"];
var={"pressfc","prmsl","pwat","q500","q850","q925","tmpsfc","u200","u850","v200","v850","z1000","z200","z500","z850"};

{dlat,dlon}=Block[{tempt=Import[DateString[{1979,1,1},{"D_CONUS_","Year","_","Month",".mx"}]]},{tempt["lat"],tempt["lon"]}];

(*
dynamics=Table[Block[{tempt,l},
  Print[months[[i]]];
  tempt=Table[Import[DateString[months[[i]],{"CONUS_","Year","_","Month","_",var[[j]],".mx"}]]["data"],{j,Length[var]}];
  l=Min[Map[Length,tempt]];
  tempt=Normal[Map[#[[;;l]]&,tempt]];
  tempt=Table[(tempt[[i]]-dmeanvar[[i,1]])/dmeanvar[[i,2]],{i,Length[var]}];
  NumericArray[tempt,"Real32"]],{i,Length[months]}];

Table[Block[{},
 Print[months[[i]]];
 Export[DateString[months[[i]],{"D_CONUS_","Year","_","Month",".mx"}],
  <|"data"->dynamics[[i]],
    "var"->var,
    "dmeanvar"->dmeanvar,
    "lat"->dlat,
    "lon"->dlon|>];
 Table[DeleteFile[DateString[months[[i]],{"CONUS_","Year","_","Month","_",var[[j]],".mx"}]],{j,Length[var]}]],{i,Length[months]}];
*)
dynamics=Table[Block[{},Print[months[[i]]];Import[DateString[months[[i]],{"D_CONUS_","Year","_","Month",".mx"}]]["data"]],{i,Length[months]}];

(*data loaded*)

months=months[[1;;-25]];

validation=Block[{select=30,day},
  day=Table[RandomSample[Range[Length[p[[i]]]],select],{i,-24,-1}];
  Flatten[Table[<|"Dynamics"->dynamics[[i,;;,(day[[i,j]]-1)*3+1;;day[[i,j]]*3]],
    "Static"->ele,
    "Po_0"->p[[i,day[[i,j]]]]|>,{j,select},{i,-24,-1}]]];

cStatic=4;
cDynamic=32;
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
"Static"->{1,301,621}]]

res[c_]:=NetGraph[<|"l"->Flatten[Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],BatchNormalizationLayer[],Ramp},3]][[1;;-2]],
          "plus"->TotalLayer[],
          "post"->Ramp|>,
   {NetPort["Input"]->"l"->"plus",NetPort["Input"]->"plus"->"post"}]
DynamicNet=Block[{c=cDynamic,c2=cDynamic+cStatic},
        NetGraph[
 <|"resize0"->{Table[{ConvolutionLayer[c,{2,3,3}],BatchNormalizationLayer[],ParametricRampLayer[]},2],FlattenLayer[1],ResizeLayer[{40,80}]},
   "cate0"->CatenateLayer[],
   "up0"->{res[c2],res[c2]},
   "output0"->{ConvolutionLayer[c,{1,1}],BatchNormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],BatchNormalizationLayer[],Ramp}|>,
   {NetPort["Dynamics"]->"resize0"->"cate0",
         NetPort["Static0"]->"cate0"->"up0"->"output0"->NetPort["P_0"]},
  "Dynamics"->{15,3,59,123}]]
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
        "Po_0"->{1,40,80}]]

batch=32;
NetTrain[SRNet,
   {Function[Block[{Nmonth=8,m,d},
        m=RandomSample[Range[Length[months]],Nmonth];
        d=Map[RandomSample[Range[Length[p[[#]]]],batch/Nmonth]&,m];
        <|"Static"->Table[ele,batch],
          "Dynamics"->Flatten[Table[Table[dynamics[[m[[i]]]][[;;,(d[[i,j]]-1)*3+1;;d[[i,j]]*3]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "Po_0"->Flatten[Table[Table[p[[m[[i]]]][[d[[i,j]]]],{j,Length[d[[i]]]}],{i,Length[m]}]]|>]],
       "RoundLength" -> Total[Map[Length[#[[1]]]&,p]]},
    LossFunction ->{"Fake_Loss"->Scaled[1],"True_Loss"->Scaled[1],"MSE_Loss"->Scaled[-5]},
    TrainingUpdateSchedule -> {"Discriminator","StaticNet","DynamicNet"},
    LearningRateMultipliers -> {"Discriminator" -> 1, "StaticNet" -> -1,"DynamicNet"->-1,"Scale_GCM->Obser"->0},
    BatchSize -> 32,
    TargetDevice->"GPU",
    MaxTrainingRounds->1000,
    Method -> {"ADAM", "Beta1" -> 0.5, "LearningRate" -> 10^-4, "WeightClipping" -> {"Discriminator"->10^-1}}]

net=NetTake[SRNet,"DynamicNet"];
simu=Table[net[<|"Dynamics"->validation[[i]]["Dynamics"],"Static"->validation[[i]]["Static"]|>,TargetDevice->"GPU"],{i,Length[validation]}];
obser=validation[[;;,"Po_0"]];
~
~
~
~
~
~
