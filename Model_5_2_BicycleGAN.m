(*import data*)
SetDirectory["/data/home/scy0446/run/Data"];
months=DateRange[{1979,1,1},{2006,12,1},"Month"];
p=Table[Block[{},Print[months[[i]]];Import["N_"<>DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["data"][[1]]],{i,Length[months]}];
ele=NumericArray[{Log[Normal[Import["/data/home/scy0446/run/Data/CONUS/Elevation.mx"]["data"]]+1.]},"Real32"];
dmeanvar=Import["/data/home/scy0446/run/Data/CONUS/meanvar_dynamics.mx"]["data"];
SetDirectory["/data/home/scy0446/run/Data/CONUS"];
var={"pressfc","prmsl","pwat","q500","q850","q925","tmpsfc","u200","u850","v200","v850","z1000","z200","z500","z850"};
{dlat,dlon}=Block[{tempt=Import[DateString[{1979,1,1},{"D_CONUS_","Year","_","Month",".mx"}]]},{tempt["lat"],tempt["lon"]}];
dynamics=Table[Block[{},Print[months[[i]]];Import[DateString[months[[i]],{"D_CONUS_","Year","_","Month",".mx"}]]["data"]],{i,Length[months]}];

(*model*)
cStatic=4;
cDynamic=32;
cEncoder=16;
dStatic={1,301,621};
dDynamic={15,3,59,123};
dLatent=8;
dOutput={1,40,80};

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

DynamicNet=Block[{c=cDynamic,c2=cDynamic+cStatic+dLatent},
        NetGraph[
 <|"latent"->{ReplicateLayer[3200],TransposeLayer[],ReshapeLayer[{dLatent,40,80}]},
   "resize0"->{Table[{ConvolutionLayer[c,{2,3,3}],BatchNormalizationLayer[],ParametricRampLayer[]},2],FlattenLayer[1],ResizeLayer[dOutput[[2;;]]]},
   "cate0"->CatenateLayer[],
   "up0"->{res[c2],res[c2]},
   "output0"->{ConvolutionLayer[c,{1,1}],BatchNormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],BatchNormalizationLayer[],Ramp}|>,
   {NetPort["Dynamics"]->"resize0"->"cate0",
    NetPort["Latent"]->"latent"->"cate0",
        NetPort["Static0"]->"cate0"->"up0"->"output0"->NetPort["P_0"]},
"Dynamics"->dDynamic,
"Static0"->NetExtract[StaticNet,"Down3"],
"Latent"->dLatent]];

discriminator = NetChain[{
    ReshapeLayer[{1,40,80}],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    FlattenLayer[],BatchNormalizationLayer[],100,Ramp,BatchNormalizationLayer[], 1, ElementwiseLayer["HardSigmoid"]},
  "Input" -> dOutput[[2;;]]]

latentLoss = NetGraph[{ElementwiseLayer[-0.5*# &], TotalLayer[], ElementwiseLayer[-(#^2) &],
   ElementwiseLayer[-Exp[2*#] &], ElementwiseLayer[1 + 2*# &], SummationLayer[]},
  {NetPort["mn"] -> 3,NetPort["sd"] -> 4,NetPort["sd"] -> 5, {3, 4, 5} -> 2 -> 6 -> 1}]

Encoder=NetGraph[<|"seq"->{ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],
                                          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],PoolingLayer[2,2],
                                          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],
                                          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],PoolingLayer[2,2],
                                          FlattenLayer[],ReplicateLayer[2]},
                                   "MeanPart"->{PartLayer[1],BatchNormalizationLayer[],LinearLayer[200],Ramp,dLatent},
                                   "StdPart"->{PartLayer[2],BatchNormalizationLayer[],LinearLayer[200],Ramp,dLatent}|>,
        {NetPort["Input"]->"seq","seq"->"MeanPart"->NetPort["mn"],"seq"->"StdPart"->NetPort["sd"]
         },
        "Input"->dOutput]

Reparameterization=NetGraph[<|"LatentLoss"->latentLoss,
                                   "expSd"->ElementwiseLayer[Exp[#]&],
                                   "time"->ThreadingLayer[Times],
                                   "add"->ThreadingLayer[Plus]|>,
         {NetPort["mn"]->NetPort["LatentLoss","mn"],
          NetPort["sd"]->NetPort["LatentLoss","sd"],
          "LatentLoss"->NetPort["KL_Loss"],
          NetPort["sd"]->"expSd"->"time",
          NetPort["Random"]->"time"->"add",
          NetPort["mn"]->"add"},
        "mn"->dLatent,
        "sd"->dLatent,
        "Random"->dLatent]

VAEDynamicNet = NetInsertSharedArrays[DynamicNet, "DynamicNet/"];
LRDynamicNet = NetInsertSharedArrays[DynamicNet, "DynamicNet/"];

VAEEncoder = NetInsertSharedArrays[Encoder,"Encoder/"];
LREncoder = NetInsertSharedArrays[Encoder,"Encoder/"];

SRNet=NetInitialize[NetGraph[<|"VAE_DynamicNet"->VAEDynamicNet,
                "LR_DynamicNet"->LRDynamicNet,
                                "StaticNet"->StaticNet,
                                "VAE_Cate"->CatenateLayer[],
                                "VAE_Discriminator"->NetMapOperator[discriminator],
                                "VAE_Reshape"->ReshapeLayer[{2}],
                                "VAE_Scale" -> ConstantTimesLayer["Scaling" -> {-1, 1}],
                                "VAE_True"->PartLayer[1],
                                "VAE_False"->PartLayer[2],
                                "VAE_MSE"->MeanSquaredLossLayer[],
                                "VAE_Encoder"->VAEEncoder,
                                "Reparameterization"->Reparameterization,

                                "LR_Cate"->CatenateLayer[],
                                "LR_Discriminator"->NetMapOperator[discriminator],
                                "LR_Reshape"->ReshapeLayer[{2}],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         19,1          Top
(*import data*)
SetDirectory["/data/home/scy0446/run/Data"];
months=DateRange[{1979,1,1},{2006,12,1},"Month"];
p=Table[Block[{},Print[months[[i]]];Import["N_"<>DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["data"][[1]]],{i,Length[months]}];
ele=NumericArray[{Log[Normal[Import["/data/home/scy0446/run/Data/CONUS/Elevation.mx"]["data"]]+1.]},"Real32"];
dmeanvar=Import["/data/home/scy0446/run/Data/CONUS/meanvar_dynamics.mx"]["data"];
SetDirectory["/data/home/scy0446/run/Data/CONUS"];
var={"pressfc","prmsl","pwat","q500","q850","q925","tmpsfc","u200","u850","v200","v850","z1000","z200","z500","z850"};
{dlat,dlon}=Block[{tempt=Import[DateString[{1979,1,1},{"D_CONUS_","Year","_","Month",".mx"}]]},{tempt["lat"],tempt["lon"]}];
dynamics=Table[Block[{},Print[months[[i]]];Import[DateString[months[[i]],{"D_CONUS_","Year","_","Month",".mx"}]]["data"]],{i,Length[months]}];

(*model*)
cStatic=4;
cDynamic=32;
cEncoder=16;
dStatic={1,301,621};
dDynamic={15,3,59,123};
dLatent=8;
dOutput={1,40,80};

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

DynamicNet=Block[{c=cDynamic,c2=cDynamic+cStatic+dLatent},
        NetGraph[
 <|"latent"->{ReplicateLayer[3200],TransposeLayer[],ReshapeLayer[{dLatent,40,80}]},
   "resize0"->{Table[{ConvolutionLayer[c,{2,3,3}],BatchNormalizationLayer[],ParametricRampLayer[]},2],FlattenLayer[1],ResizeLayer[dOutput[[2;;]]]},
   "cate0"->CatenateLayer[],
   "up0"->{res[c2],res[c2]},
   "output0"->{ConvolutionLayer[c,{1,1}],BatchNormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],BatchNormalizationLayer[],Ramp}|>,
   {NetPort["Dynamics"]->"resize0"->"cate0",
    NetPort["Latent"]->"latent"->"cate0",
        NetPort["Static0"]->"cate0"->"up0"->"output0"->NetPort["P_0"]},
"Dynamics"->dD,
"Static0"->NetExtract[StaticNet,"Down3"],
"Latent"->dLatent]];

discriminator = NetChain[{
    ReshapeLayer[{1,40,80}],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    FlattenLayer[],BatchNormalizationLayer[],100,Ramp,BatchNormalizationLayer[], 1, ElementwiseLayer["HardSigmoid"]},
  "Input" -> dOutput[[2;;]]]

latentLoss = NetGraph[{ElementwiseLayer[-0.5*# &], TotalLayer[], ElementwiseLayer[-(#^2) &],
   ElementwiseLayer[-Exp[2*#] &], ElementwiseLayer[1 + 2*# &], SummationLayer[]},
  {NetPort["mn"] -> 3,NetPort["sd"] -> 4,NetPort["sd"] -> 5, {3, 4, 5} -> 2 -> 6 -> 1}]

Encoder=NetGraph[<|"seq"->{ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],
                                          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],PoolingLayer[2,2],
                                          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],
                                          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],PoolingLayer[2,2],
                                          FlattenLayer[],ReplicateLayer[2]},
                                   "MeanPart"->{PartLayer[1],BatchNormalizationLayer[],LinearLayer[200],Ramp,dLatent},
                                   "StdPart"->{PartLayer[2],BatchNormalizationLayer[],LinearLayer[200],Ramp,dLatent}|>,
        {NetPort["Input"]->"seq","seq"->"MeanPart"->NetPort["mn"],"seq"->"StdPart"->NetPort["sd"]
         },
        "Input"->dOutput]

Reparameterization=NetGraph[<|"LatentLoss"->latentLoss,
                                   "expSd"->ElementwiseLayer[Exp[#]&],
                                   "time"->ThreadingLayer[Times],
                                   "add"->ThreadingLayer[Plus]|>,
         {NetPort["mn"]->NetPort["LatentLoss","mn"],
          NetPort["sd"]->NetPort["LatentLoss","sd"],
          "LatentLoss"->NetPort["KL_Loss"],
          NetPort["sd"]->"expSd"->"time",
          NetPort["Random"]->"time"->"add",
          NetPort["mn"]->"add"},
        "mn"->dLatent,
        "sd"->dLatent,
        "Random"->dLatent]

VAEDynamicNet = NetInsertSharedArrays[DynamicNet, "DynamicNet/"];
LRDynamicNet = NetInsertSharedArrays[DynamicNet, "DynamicNet/"];

VAEEncoder = NetInsertSharedArrays[Encoder,"Encoder/"];
LREncoder = NetInsertSharedArrays[Encoder,"Encoder/"];

SRNet=NetInitialize[NetGraph[<|"VAE_DynamicNet"->VAEDynamicNet,
                "LR_DynamicNet"->LRDynamicNet,
                                "StaticNet"->StaticNet,
                                "VAE_Cate"->CatenateLayer[],
                                "VAE_Discriminator"->NetMapOperator[discriminator],
                                "VAE_Reshape"->ReshapeLayer[{2}],
                                "VAE_Scale" -> ConstantTimesLayer["Scaling" -> {-1, 1}],
                                "VAE_True"->PartLayer[1],
                                "VAE_False"->PartLayer[2],
                                "VAE_MSE"->MeanSquaredLossLayer[],
                                "VAE_Encoder"->VAEEncoder,
                                "Reparameterization"->Reparameterization,

                                "LR_Cate"->CatenateLayer[],
                                "LR_Discriminator"->NetMapOperator[discriminator],
                                "LR_Reshape"->ReshapeLayer[{2}],
                                "LR_Scale" -> ConstantTimesLayer["Scaling" -> {-1, 1}],
                                "LR_True"->PartLayer[1],
                                "LR_False"->PartLayer[2],
                                "LR_Encoder"->LREncoder,
                                "LR_MSE"->MeanSquaredLossLayer[]|>,
        {NetPort["Dynamics"]->NetPort["VAE_DynamicNet","Dynamics"],
         NetPort["Static"]->NetPort["StaticNet","Static"],
         NetPort["StaticNet","Down3"]->NetPort["VAE_DynamicNet","Static0"],
         NetPort["VAE_DynamicNet","P_0"]->"VAE_Cate",
         NetPort["Po_0"]->"VAE_Cate"->"VAE_Discriminator"->"VAE_Reshape"->"VAE_Scale",
         "VAE_Scale"->"VAE_True"->NetPort["VAE_True_Loss"],
         "VAE_Scale"->"VAE_False"->NetPort["VAE_Fake_Loss"],
         NetPort["Po_0"]->"VAE_MSE",
         NetPort["VAE_DynamicNet","P_0"]->"VAE_MSE"->NetPort["VAE_MSE_Loss"],
         NetPort["Po_0"]->NetPort["VAE_Encoder","Input"],
         NetPort["VAE_Encoder","mn"]->NetPort["Reparameterization","mn"],
         NetPort["VAE_Encoder","sd"]->NetPort["Reparameterization","sd"],
         NetPort["VAE_Random"]->NetPort["Reparameterization","Random"],
         NetPort["Reparameterization","Output"]->NetPort["VAE_DynamicNet","Latent"],

         NetPort["Dynamics"]->NetPort["LR_DynamicNet","Dynamics"],
         NetPort["StaticNet","Down3"]->NetPort["LR_DynamicNet","Static0"],
         NetPort["LR_Random"]->NetPort["LR_DynamicNet","Latent"],
         NetPort["LR_DynamicNet","P_0"]->"LR_Cate",
         NetPort["Po_0"]->"LR_Cate"->"LR_Discriminator"->"LR_Reshape"->"LR_Scale",
         "LR_Scale"->"LR_True"->NetPort["LR_True_Loss"],
         "LR_Scale"->"LR_False"->NetPort["LR_Fake_Loss"],

         NetPort["LR_DynamicNet","P_0"]->"LR_Encoder",
         NetPort["LR_Encoder","mn"]->"LR_MSE",
         NetPort["LR_Random"]->"LR_MSE"->NetPort["LR_MSE_Loss"]},
        "Dynamics"->dDynamic,
        "Static"->dStatic,
        "Po_0"->dOutput,
        "VAE_Random"->dLatent,
        "LR_Random"->dLatent]]


months=months[[1;;-25]];

validation=Block[{select=30,day},
  day=Table[RandomSample[Range[Length[p[[i]]]],select],{i,-24,-1}];
  Flatten[Table[<|"Dynamics"->dynamics[[i,;;,(day[[i,j]]-1)*3+1;;day[[i,j]]*3]],
    "Static"->ele,
    "Po_0"->p[[i,day[[i,j]]]]|>,{j,select},{i,-24,-1}]]];

EnergyDistance[x_,y_]:=Block[{a,b,c},
 a=2*ParallelTable[Total[Flatten[(x[[i]]-y[[j]])^2]],{i,Length[x]},{j,Length[y]}];
 Print[Mean[Flatten[a]]];
 b=ParallelTable[Total[Flatten[(x[[i]]-x[[j]])^2]],{i,Length[x]},{j,Length[x]}];
 Print[Mean[Flatten[b]]];
 c=ParallelTable[Total[Flatten[(y[[i]]-y[[j]])^2]],{i,Length[y]},{j,Length[y]}];
 Print[Mean[Flatten[c]]];
 Mean[Flatten[a]]-Mean[Flatten[b]]-Mean[Flatten[c]]]

index=StringSplit[CreateUUID[],"-"][[1]];
globe=Infinity;
gcorr=-Infinity;

(*
ReportSRnet1[net_] :=
  Block[{obser,simu,snet,result,corr},
        obser=validation[[;;,"Po_0"]][[;;,1]];
        snet=NetTake[net,"LR_DynamicNet"];
        simu=Table[snet[<|"Dynamics"->validation[[i,"Dynamics"]],"Static"->validation[[i,"Static"]],"LR_Random"->Table[0,dLatent]|>, TargetDevice->"GPU"],{i,Length[validation]}][[;;,1]];
        corr=Table[Correlation[simu[[;;,i,j]],obser[[;;,i,j]]],{i,40},{j,80}];
        Print[Mean[Select[Flatten[corr],NumberQ]]];
        If[Mean[Select[Flatten[corr],NumberQ]]>gcorr,
          Block[{},
                Export["/data/home/scy0446/run/Code/trained_Bicycle_"<>index<>"_d.mx",net];
                Print["update"];
                Set[gcorr,Mean[Select[Flatten[corr],NumberQ]]];]]];

(*training*)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    19,1          Top
(*import data*)
SetDirectory["/data/home/scy0446/run/Data"];
months=DateRange[{1979,1,1},{2006,12,1},"Month"];
p=Table[Block[{},Print[months[[i]]];Import["N_"<>DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["data"][[1]]],{i,Length[months]}];
ele=NumericArray[{Log[Normal[Import["/data/home/scy0446/run/Data/CONUS/Elevation.mx"]["data"]]+1.]},"Real32"];
dmeanvar=Import["/data/home/scy0446/run/Data/CONUS/meanvar_dynamics.mx"]["data"];
SetDirectory["/data/home/scy0446/run/Data/CONUS"];
var={"pressfc","prmsl","pwat","q500","q850","q925","tmpsfc","u200","u850","v200","v850","z1000","z200","z500","z850"};
{dlat,dlon}=Block[{tempt=Import[DateString[{1979,1,1},{"D_CONUS_","Year","_","Month",".mx"}]]},{tempt["lat"],tempt["lon"]}];
dynamics=Table[Block[{},Print[months[[i]]];Import[DateString[months[[i]],{"D_CONUS_","Year","_","Month",".mx"}]]["data"]],{i,Length[months]}];

(*model*)
cStatic=4;
cDynamic=32;
cEncoder=16;
dStatic={1,301,621};
dDynamic={15,3,59,123};
dLatent=8;
dOutput={1,40,80};

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

DynamicNet=Block[{c=cDynamic,c2=cDynamic+cStatic+dLatent},
        NetGraph[
 <|"latent"->{ReplicateLayer[3200],TransposeLayer[],ReshapeLayer[{dLatent,40,80}]},
   "resize0"->{Table[{ConvolutionLayer[c,{2,3,3}],BatchNormalizationLayer[],ParametricRampLayer[]},2],FlattenLayer[1],ResizeLayer[dOutput[[2;;]]]},
   "cate0"->CatenateLayer[],
   "up0"->{res[c2],res[c2]},
   "output0"->{ConvolutionLayer[c,{1,1}],BatchNormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],BatchNormalizationLayer[],Ramp}|>,
   {NetPort["Dynamics"]->"resize0"->"cate0",
    NetPort["Latent"]->"latent"->"cate0",
        NetPort["Static0"]->"cate0"->"up0"->"output0"->NetPort["P_0"]},
"Dynamics"->dD,
"Static0"->NetExtract[StaticNet,"Down3"],
"Latent"->dLatent]];

discriminator = NetChain[{
    ReshapeLayer[{1,40,80}],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    FlattenLayer[],BatchNormalizationLayer[],100,Ramp,BatchNormalizationLayer[], 1, ElementwiseLayer["HardSigmoid"]},
  "Input" -> dOutput[[2;;]]]

latentLoss = NetGraph[{ElementwiseLayer[-0.5*# &], TotalLayer[], ElementwiseLayer[-(#^2) &],
   ElementwiseLayer[-Exp[2*#] &], ElementwiseLayer[1 + 2*# &], SummationLayer[]},
  {NetPort["mn"] -> 3,NetPort["sd"] -> 4,NetPort["sd"] -> 5, {3, 4, 5} -> 2 -> 6 -> 1}]

Encoder=NetGraph[<|"seq"->{ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],
                                          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],PoolingLayer[2,2],
                                          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],
                                          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],PoolingLayer[2,2],
                                          FlattenLayer[],ReplicateLayer[2]},
                                   "MeanPart"->{PartLayer[1],BatchNormalizationLayer[],LinearLayer[200],Ramp,dLatent},
                                   "StdPart"->{PartLayer[2],BatchNormalizationLayer[],LinearLayer[200],Ramp,dLatent}|>,
        {NetPort["Input"]->"seq","seq"->"MeanPart"->NetPort["mn"],"seq"->"StdPart"->NetPort["sd"]
         },
        "Input"->dOutput]

Reparameterization=NetGraph[<|"LatentLoss"->latentLoss,
                                   "expSd"->ElementwiseLayer[Exp[#]&],
                                   "time"->ThreadingLayer[Times],
                                   "add"->ThreadingLayer[Plus]|>,
         {NetPort["mn"]->NetPort["LatentLoss","mn"],
          NetPort["sd"]->NetPort["LatentLoss","sd"],
          "LatentLoss"->NetPort["KL_Loss"],
          NetPort["sd"]->"expSd"->"time",
          NetPort["Random"]->"time"->"add",
          NetPort["mn"]->"add"},
        "mn"->dLatent,
        "sd"->dLatent,
        "Random"->dLatent]

VAEDynamicNet = NetInsertSharedArrays[DynamicNet, "DynamicNet/"];
LRDynamicNet = NetInsertSharedArrays[DynamicNet, "DynamicNet/"];

VAEEncoder = NetInsertSharedArrays[Encoder,"Encoder/"];
LREncoder = NetInsertSharedArrays[Encoder,"Encoder/"];

SRNet=NetInitialize[NetGraph[<|"VAE_DynamicNet"->VAEDynamicNet,
                "LR_DynamicNet"->LRDynamicNet,
                                "StaticNet"->StaticNet,
                                "VAE_Cate"->CatenateLayer[],
                                "VAE_Discriminator"->NetMapOperator[discriminator],
                                "VAE_Reshape"->ReshapeLayer[{2}],
                                "VAE_Scale" -> ConstantTimesLayer["Scaling" -> {-1, 1}],
                                "VAE_True"->PartLayer[1],
                                "VAE_False"->PartLayer[2],
                                "VAE_MSE"->MeanSquaredLossLayer[],
                                "VAE_Encoder"->VAEEncoder,
                                "Reparameterization"->Reparameterization,

                                "LR_Cate"->CatenateLayer[],
                                "LR_Discriminator"->NetMapOperator[discriminator],
                                "LR_Reshape"->ReshapeLayer[{2}],
                                "LR_Scale" -> ConstantTimesLayer["Scaling" -> {-1, 1}],
                                "LR_True"->PartLayer[1],
                                "LR_False"->PartLayer[2],
                                "LR_Encoder"->LREncoder,
                                "LR_MSE"->MeanSquaredLossLayer[]|>,
        {NetPort["Dynamics"]->NetPort["VAE_DynamicNet","Dynamics"],
         NetPort["Static"]->NetPort["StaticNet","Static"],
         NetPort["StaticNet","Down3"]->NetPort["VAE_DynamicNet","Static0"],
         NetPort["VAE_DynamicNet","P_0"]->"VAE_Cate",
         NetPort["Po_0"]->"VAE_Cate"->"VAE_Discriminator"->"VAE_Reshape"->"VAE_Scale",
         "VAE_Scale"->"VAE_True"->NetPort["VAE_True_Loss"],
         "VAE_Scale"->"VAE_False"->NetPort["VAE_Fake_Loss"],
         NetPort["Po_0"]->"VAE_MSE",
         NetPort["VAE_DynamicNet","P_0"]->"VAE_MSE"->NetPort["VAE_MSE_Loss"],
         NetPort["Po_0"]->NetPort["VAE_Encoder","Input"],
         NetPort["VAE_Encoder","mn"]->NetPort["Reparameterization","mn"],
         NetPort["VAE_Encoder","sd"]->NetPort["Reparameterization","sd"],
         NetPort["VAE_Random"]->NetPort["Reparameterization","Random"],
         NetPort["Reparameterization","Output"]->NetPort["VAE_DynamicNet","Latent"],

         NetPort["Dynamics"]->NetPort["LR_DynamicNet","Dynamics"],
         NetPort["StaticNet","Down3"]->NetPort["LR_DynamicNet","Static0"],
         NetPort["LR_Random"]->NetPort["LR_DynamicNet","Latent"],
         NetPort["LR_DynamicNet","P_0"]->"LR_Cate",
         NetPort["Po_0"]->"LR_Cate"->"LR_Discriminator"->"LR_Reshape"->"LR_Scale",
         "LR_Scale"->"LR_True"->NetPort["LR_True_Loss"],
         "LR_Scale"->"LR_False"->NetPort["LR_Fake_Loss"],

         NetPort["LR_DynamicNet","P_0"]->"LR_Encoder",
         NetPort["LR_Encoder","mn"]->"LR_MSE",
         NetPort["LR_Random"]->"LR_MSE"->NetPort["LR_MSE_Loss"]},
        "Dynamics"->dDynamic,
        "Static"->dStatic,
        "Po_0"->dOutput,
        "VAE_Random"->dLatent,
        "LR_Random"->dLatent]]


months=months[[1;;-25]];

validation=Block[{select=30,day},
  day=Table[RandomSample[Range[Length[p[[i]]]],select],{i,-24,-1}];
  Flatten[Table[<|"Dynamics"->dynamics[[i,;;,(day[[i,j]]-1)*3+1;;day[[i,j]]*3]],
    "Static"->ele,
    "Po_0"->p[[i,day[[i,j]]]]|>,{j,select},{i,-24,-1}]]];

EnergyDistance[x_,y_]:=Block[{a,b,c},
 a=2*ParallelTable[Total[Flatten[(x[[i]]-y[[j]])^2]],{i,Length[x]},{j,Length[y]}];
 Print[Mean[Flatten[a]]];
 b=ParallelTable[Total[Flatten[(x[[i]]-x[[j]])^2]],{i,Length[x]},{j,Length[x]}];
 Print[Mean[Flatten[b]]];
 c=ParallelTable[Total[Flatten[(y[[i]]-y[[j]])^2]],{i,Length[y]},{j,Length[y]}];
 Print[Mean[Flatten[c]]];
 Mean[Flatten[a]]-Mean[Flatten[b]]-Mean[Flatten[c]]]

index=StringSplit[CreateUUID[],"-"][[1]];
globe=Infinity;
gcorr=-Infinity;

(*
ReportSRnet1[net_] :=
  Block[{obser,simu,snet,result,corr},
        obser=validation[[;;,"Po_0"]][[;;,1]];
        snet=NetTake[net,"LR_DynamicNet"];
        simu=Table[snet[<|"Dynamics"->validation[[i,"Dynamics"]],"Static"->validation[[i,"Static"]],"LR_Random"->Table[0,dLatent]|>, TargetDevice->"GPU"],{i,Length[validation]}][[;;,1]];
        corr=Table[Correlation[simu[[;;,i,j]],obser[[;;,i,j]]],{i,40},{j,80}];
        Print[Mean[Select[Flatten[corr],NumberQ]]];
        If[Mean[Select[Flatten[corr],NumberQ]]>gcorr,
          Block[{},
                Export["/data/home/scy0446/run/Code/trained_Bicycle_"<>index<>"_d.mx",net];
                Print["update"];
                Set[gcorr,Mean[Select[Flatten[corr],NumberQ]]];]]];

(*training*)
batch=64;
NetTrain[SRNet,
   {Function[Block[{Nmonth=8,m,d},
        m=RandomSample[Range[Length[months]],Nmonth];
        d=Map[RandomSample[Range[Length[p[[#]]]],batch/Nmonth]&,m];
        <|"Static"->Table[ele,batch],
          "Dynamics"->Flatten[Table[Table[dynamics[[m[[i]]]][[;;,(d[[i,j]]-1)*3+1;;d[[i,j]]*3]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "Po_0"->Flatten[Table[Table[p[[m[[i]]]][[d[[i,j]]]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "LR_Random"->Table[Table[0,dLatent],batch],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    19,1          Top
(*import data*)
SetDirectory["/data/home/scy0446/run/Data"];
months=DateRange[{1979,1,1},{2006,12,1},"Month"];
p=Table[Block[{},Print[months[[i]]];Import["N_"<>DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["data"][[1]]],{i,Length[months]}];
ele=NumericArray[{Log[Normal[Import["/data/home/scy0446/run/Data/CONUS/Elevation.mx"]["data"]]+1.]},"Real32"];
dmeanvar=Import["/data/home/scy0446/run/Data/CONUS/meanvar_dynamics.mx"]["data"];
SetDirectory["/data/home/scy0446/run/Data/CONUS"];
var={"pressfc","prmsl","pwat","q500","q850","q925","tmpsfc","u200","u850","v200","v850","z1000","z200","z500","z850"};
{dlat,dlon}=Block[{tempt=Import[DateString[{1979,1,1},{"D_CONUS_","Year","_","Month",".mx"}]]},{tempt["lat"],tempt["lon"]}];
dynamics=Table[Block[{},Print[months[[i]]];Import[DateString[months[[i]],{"D_CONUS_","Year","_","Month",".mx"}]]["data"]],{i,Length[months]}];

(*model*)
cStatic=4;
cDynamic=32;
cEncoder=16;
dStatic={1,301,621};
dDynamic={15,3,59,123};
dLatent=8;
dOutput={1,40,80};

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

DynamicNet=Block[{c=cDynamic,c2=cDynamic+cStatic+dLatent},
        NetGraph[
 <|"latent"->{ReplicateLayer[3200],TransposeLayer[],ReshapeLayer[{dLatent,40,80}]},
   "resize0"->{Table[{ConvolutionLayer[c,{2,3,3}],BatchNormalizationLayer[],ParametricRampLayer[]},2],FlattenLayer[1],ResizeLayer[dOutput[[2;;]]]},
   "cate0"->CatenateLayer[],
   "up0"->{res[c2],res[c2]},
   "output0"->{ConvolutionLayer[c,{1,1}],BatchNormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],BatchNormalizationLayer[],Ramp}|>,
   {NetPort["Dynamics"]->"resize0"->"cate0",
    NetPort["Latent"]->"latent"->"cate0",
        NetPort["Static0"]->"cate0"->"up0"->"output0"->NetPort["P_0"]},
"Dynamics"->dD,
"Static0"->NetExtract[StaticNet,"Down3"],
"Latent"->dLatent]];

discriminator = NetChain[{
    ReshapeLayer[{1,40,80}],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    FlattenLayer[],BatchNormalizationLayer[],100,Ramp,BatchNormalizationLayer[], 1, ElementwiseLayer["HardSigmoid"]},
  "Input" -> dOutput[[2;;]]]

latentLoss = NetGraph[{ElementwiseLayer[-0.5*# &], TotalLayer[], ElementwiseLayer[-(#^2) &],
   ElementwiseLayer[-Exp[2*#] &], ElementwiseLayer[1 + 2*# &], SummationLayer[]},
  {NetPort["mn"] -> 3,NetPort["sd"] -> 4,NetPort["sd"] -> 5, {3, 4, 5} -> 2 -> 6 -> 1}]

Encoder=NetGraph[<|"seq"->{ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],
                                          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],PoolingLayer[2,2],
                                          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],
                                          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],PoolingLayer[2,2],
                                          FlattenLayer[],ReplicateLayer[2]},
                                   "MeanPart"->{PartLayer[1],BatchNormalizationLayer[],LinearLayer[200],Ramp,dLatent},
                                   "StdPart"->{PartLayer[2],BatchNormalizationLayer[],LinearLayer[200],Ramp,dLatent}|>,
        {NetPort["Input"]->"seq","seq"->"MeanPart"->NetPort["mn"],"seq"->"StdPart"->NetPort["sd"]
         },
        "Input"->dOutput]

Reparameterization=NetGraph[<|"LatentLoss"->latentLoss,
                                   "expSd"->ElementwiseLayer[Exp[#]&],
                                   "time"->ThreadingLayer[Times],
                                   "add"->ThreadingLayer[Plus]|>,
         {NetPort["mn"]->NetPort["LatentLoss","mn"],
          NetPort["sd"]->NetPort["LatentLoss","sd"],
          "LatentLoss"->NetPort["KL_Loss"],
          NetPort["sd"]->"expSd"->"time",
          NetPort["Random"]->"time"->"add",
          NetPort["mn"]->"add"},
        "mn"->dLatent,
        "sd"->dLatent,
        "Random"->dLatent]

VAEDynamicNet = NetInsertSharedArrays[DynamicNet, "DynamicNet/"];
LRDynamicNet = NetInsertSharedArrays[DynamicNet, "DynamicNet/"];

VAEEncoder = NetInsertSharedArrays[Encoder,"Encoder/"];
LREncoder = NetInsertSharedArrays[Encoder,"Encoder/"];

SRNet=NetInitialize[NetGraph[<|"VAE_DynamicNet"->VAEDynamicNet,
                "LR_DynamicNet"->LRDynamicNet,
                                "StaticNet"->StaticNet,
                                "VAE_Cate"->CatenateLayer[],
                                "VAE_Discriminator"->NetMapOperator[discriminator],
                                "VAE_Reshape"->ReshapeLayer[{2}],
                                "VAE_Scale" -> ConstantTimesLayer["Scaling" -> {-1, 1}],
                                "VAE_True"->PartLayer[1],
                                "VAE_False"->PartLayer[2],
                                "VAE_MSE"->MeanSquaredLossLayer[],
                                "VAE_Encoder"->VAEEncoder,
                                "Reparameterization"->Reparameterization,

                                "LR_Cate"->CatenateLayer[],
                                "LR_Discriminator"->NetMapOperator[discriminator],
                                "LR_Reshape"->ReshapeLayer[{2}],
                                "LR_Scale" -> ConstantTimesLayer["Scaling" -> {-1, 1}],
                                "LR_True"->PartLayer[1],
                                "LR_False"->PartLayer[2],
                                "LR_Encoder"->LREncoder,
                                "LR_MSE"->MeanSquaredLossLayer[]|>,
        {NetPort["Dynamics"]->NetPort["VAE_DynamicNet","Dynamics"],
         NetPort["Static"]->NetPort["StaticNet","Static"],
         NetPort["StaticNet","Down3"]->NetPort["VAE_DynamicNet","Static0"],
         NetPort["VAE_DynamicNet","P_0"]->"VAE_Cate",
         NetPort["Po_0"]->"VAE_Cate"->"VAE_Discriminator"->"VAE_Reshape"->"VAE_Scale",
         "VAE_Scale"->"VAE_True"->NetPort["VAE_True_Loss"],
         "VAE_Scale"->"VAE_False"->NetPort["VAE_Fake_Loss"],
         NetPort["Po_0"]->"VAE_MSE",
         NetPort["VAE_DynamicNet","P_0"]->"VAE_MSE"->NetPort["VAE_MSE_Loss"],
         NetPort["Po_0"]->NetPort["VAE_Encoder","Input"],
         NetPort["VAE_Encoder","mn"]->NetPort["Reparameterization","mn"],
         NetPort["VAE_Encoder","sd"]->NetPort["Reparameterization","sd"],
         NetPort["VAE_Random"]->NetPort["Reparameterization","Random"],
         NetPort["Reparameterization","Output"]->NetPort["VAE_DynamicNet","Latent"],

         NetPort["Dynamics"]->NetPort["LR_DynamicNet","Dynamics"],
         NetPort["StaticNet","Down3"]->NetPort["LR_DynamicNet","Static0"],
         NetPort["LR_Random"]->NetPort["LR_DynamicNet","Latent"],
         NetPort["LR_DynamicNet","P_0"]->"LR_Cate",
         NetPort["Po_0"]->"LR_Cate"->"LR_Discriminator"->"LR_Reshape"->"LR_Scale",
         "LR_Scale"->"LR_True"->NetPort["LR_True_Loss"],
         "LR_Scale"->"LR_False"->NetPort["LR_Fake_Loss"],

         NetPort["LR_DynamicNet","P_0"]->"LR_Encoder",
         NetPort["LR_Encoder","mn"]->"LR_MSE",
         NetPort["LR_Random"]->"LR_MSE"->NetPort["LR_MSE_Loss"]},
        "Dynamics"->dDynamic,
        "Static"->dStatic,
        "Po_0"->dOutput,
        "VAE_Random"->dLatent,
        "LR_Random"->dLatent]]


months=months[[1;;-25]];

validation=Block[{select=30,day},
  day=Table[RandomSample[Range[Length[p[[i]]]],select],{i,-24,-1}];
  Flatten[Table[<|"Dynamics"->dynamics[[i,;;,(day[[i,j]]-1)*3+1;;day[[i,j]]*3]],
    "Static"->ele,
    "Po_0"->p[[i,day[[i,j]]]]|>,{j,select},{i,-24,-1}]]];

EnergyDistance[x_,y_]:=Block[{a,b,c},
 a=2*ParallelTable[Total[Flatten[(x[[i]]-y[[j]])^2]],{i,Length[x]},{j,Length[y]}];
 Print[Mean[Flatten[a]]];
 b=ParallelTable[Total[Flatten[(x[[i]]-x[[j]])^2]],{i,Length[x]},{j,Length[x]}];
 Print[Mean[Flatten[b]]];
 c=ParallelTable[Total[Flatten[(y[[i]]-y[[j]])^2]],{i,Length[y]},{j,Length[y]}];
 Print[Mean[Flatten[c]]];
 Mean[Flatten[a]]-Mean[Flatten[b]]-Mean[Flatten[c]]]

index=StringSplit[CreateUUID[],"-"][[1]];
globe=Infinity;
gcorr=-Infinity;

(*
ReportSRnet1[net_] :=
  Block[{obser,simu,snet,result,corr},
        obser=validation[[;;,"Po_0"]][[;;,1]];
        snet=NetTake[net,"LR_DynamicNet"];
        simu=Table[snet[<|"Dynamics"->validation[[i,"Dynamics"]],"Static"->validation[[i,"Static"]],"LR_Random"->Table[0,dLatent]|>, TargetDevice->"GPU"],{i,Length[validation]}][[;;,1]];
        corr=Table[Correlation[simu[[;;,i,j]],obser[[;;,i,j]]],{i,40},{j,80}];
        Print[Mean[Select[Flatten[corr],NumberQ]]];
        If[Mean[Select[Flatten[corr],NumberQ]]>gcorr,
          Block[{},
                Export["/data/home/scy0446/run/Code/trained_Bicycle_"<>index<>"_d.mx",net];
                Print["update"];
                Set[gcorr,Mean[Select[Flatten[corr],NumberQ]]];]]];

(*training*)
batch=64;
NetTrain[SRNet,
   {Function[Block[{Nmonth=8,m,d},
        m=RandomSample[Range[Length[months]],Nmonth];
        d=Map[RandomSample[Range[Length[p[[#]]]],batch/Nmonth]&,m];
        <|"Static"->Table[ele,batch],
          "Dynamics"->Flatten[Table[Table[dynamics[[m[[i]]]][[;;,(d[[i,j]]-1)*3+1;;d[[i,j]]*3]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "Po_0"->Flatten[Table[Table[p[[m[[i]]]][[d[[i,j]]]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "LR_Random"->Table[Table[0,dLatent],batch],
          "VAE_Random"->Table[Table[0,dLatent],batch]|>]],
       "RoundLength" -> Total[Map[Length[#]&,p]]},
    LossFunction ->{"VAE_Fake_Loss"->Scaled[0],"VAE_True_Loss"->Scaled[0],"VAE_MSE_Loss"->Scaled[-10],"KL_Loss"->Scaled[0],
                    "LR_Fake_Loss"->Scaled[0],"LR_True_Loss"->Scaled[0],"LR_MSE_Loss"->Scaled[0]},
    TrainingUpdateSchedule -> {"VAE_Encoder","LR_Encoder","StaticNet","VAE_DynamicNet","LR_DynamicNet"},
    LearningRateMultipliers -> {"VAE_Discriminator" -> 0, "LR_Discriminator"->0,
                                "VAE_Encoder"->-1,"LR_Encoder"->-1,
                                "StaticNet"->-1,
                                "VAE_DynamicNet"->-1,"LR_DynamicNet"->-1,
                                "VAE_Scale"->0,"LR_Scale"->0},
    BatchSize -> 64,
    TargetDevice->{"GPU",All},
    MaxTrainingRounds->100,
    Method -> {"ADAM", "Beta1" -> 0.5, "LearningRate" -> 10^-4},
    TrainingProgressReporting->{{Function@ReportSRnet1[#Net], "Interval" -> Quantity[100, "Batches"]},"Print"}];

SRNet=Import["/data/home/scy0446/run/Code/trained_Bicycle_"<>index<>"_d.mx"];
*)

gcorr=-Infinity;
ReportSRnet2[net_] :=
  Block[{obser,simu,snet,result,corr},
        obser=validation[[;;,"Po_0"]][[;;,1]];
        snet=NetTake[net,"LR_DynamicNet"];
        simu=Table[snet[<|"Dynamics"->validation[[i,"Dynamics"]],"Static"->validation[[i,"Static"]],"LR_Random"->RandomReal[NormalDistribution[0,1],dLatent]|>, TargetDevice->"GPU"],{i,Length[validation]}][[;;,1]];
        corr=Table[Correlation[simu[[;;,i,j]],obser[[;;,i,j]]],{i,40},{j,80}];
        Print[Mean[Select[Flatten[corr],NumberQ]]];
        result=EnergyDistance[Normal[obser],Normal[simu]];
        Print[{globe,result}];
        If[result<globe,
          Block[{},
                Export["/data/home/scy0446/run/Code/trained_Bicycle_"<>index<>".mx",net];
                Print["update"];
                Set[globe,result];]]];


ReportSRnet2[net_] :=
  Block[{obser,simu,snet,result,corr},
        obser=validation[[;;,"Po_0"]][[;;,1]];
        snet=NetTake[net,"LR_DynamicNet"];
        simu=Table[snet[<|"Dynamics"->validation[[i,"Dynamics"]],"Static"->validation[[i,"Static"]],"LR_Random"->RandomReal[NormalDistribution[0,1],dLatent]|>, TargetDevice->"GPU"],{i,Length[validation]}][[;;,1]];
        corr=Quiet[Table[Correlation[simu[[;;,i,j]],obser[[;;,i,j]]],{i,40},{j,80}]];
        Print[Mean[Select[Flatten[corr],NumberQ]]];
        Print[{gcorr,Mean[Select[Flatten[corr],NumberQ]]}];
        If[gcorr<Mean[Select[Flatten[corr],NumberQ]],
          Block[{},
                Export["/data/home/scy0446/run/Code/trained_Bicycle_"<>index<>".mx",net];
                Print["update"];
                Set[gcorr,Mean[Select[Flatten[corr],NumberQ]]];]]];

(*training*)
batch=128;
NetTrain[SRNet,
   {Function[Block[{Nmonth=8,m,d},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    19,1          Top

































































































































































































































































"Bicycle_GAN_3.m" 279L, 14172C
(*import data*)
SetDirectory["/data/home/scy0446/run/Data"];
months=DateRange[{1979,1,1},{2006,12,1},"Month"];
p=Table[Block[{},Print[months[[i]]];Import["N_"<>DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["data"][[1]]],{i,Length[months]}];
ele=NumericArray[{Log[Normal[Import["/data/home/scy0446/run/Data/CONUS/Elevation.mx"]["data"]]+1.]},"Real32"];
dmeanvar=Import["/data/home/scy0446/run/Data/CONUS/meanvar_dynamics.mx"]["data"];
SetDirectory["/data/home/scy0446/run/Data/CONUS"];
var={"pressfc","prmsl","pwat","q500","q850","q925","tmpsfc","u200","u850","v200","v850","z1000","z200","z500","z850"};
{dlat,dlon}=Block[{tempt=Import[DateString[{1979,1,1},{"D_CONUS_","Year","_","Month",".mx"}]]},{tempt["lat"],tempt["lon"]}];
dynamics=Table[Block[{},Print[months[[i]]];Import[DateString[months[[i]],{"D_CONUS_","Year","_","Month",".mx"}]]["data"]],{i,Length[months]}];

(*model*)
cStatic=4;
cDynamic=32;
cEncoder=16;
dStatic={1,301,621};
dDynamic={15,3,59,123};
dLatent=8;
dOutput={1,40,80};

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

DynamicNet=Block[{c=cDynamic,c2=cDynamic+cStatic+dLatent},
        NetGraph[
 <|"latent"->{ReplicateLayer[3200],TransposeLayer[],ReshapeLayer[{dLatent,40,80}]},
   "resize0"->{Table[{ConvolutionLayer[c,{2,3,3}],BatchNormalizationLayer[],ParametricRampLayer[]},2],FlattenLayer[1],ResizeLayer[dOutput[[2;;]]]},
   "cate0"->CatenateLayer[],
   "up0"->{res[c2],res[c2]},
   "output0"->{ConvolutionLayer[c,{1,1}],BatchNormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],BatchNormalizationLayer[],Ramp}|>,
   {NetPort["Dynamics"]->"resize0"->"cate0",
    NetPort["Latent"]->"latent"->"cate0",
        NetPort["Static0"]->"cate0"->"up0"->"output0"->NetPort["P_0"]},
"Dynamics"->dD,
"Static0"->NetExtract[StaticNet,"Down3"],
"Latent"->dLatent]];

discriminator = NetChain[{
    ReshapeLayer[{1,40,80}],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    ConvolutionLayer[64,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,
    ConvolutionLayer[64,{3,3},"Stride"->2],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2],
    FlattenLayer[],BatchNormalizationLayer[],100,Ramp,BatchNormalizationLayer[], 1, ElementwiseLayer["HardSigmoid"]},
  "Input" -> dOutput[[2;;]]]

latentLoss = NetGraph[{ElementwiseLayer[-0.5*# &], TotalLayer[], ElementwiseLayer[-(#^2) &],
   ElementwiseLayer[-Exp[2*#] &], ElementwiseLayer[1 + 2*# &], SummationLayer[]},
  {NetPort["mn"] -> 3,NetPort["sd"] -> 4,NetPort["sd"] -> 5, {3, 4, 5} -> 2 -> 6 -> 1}]

Encoder=NetGraph[<|"seq"->{ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],
                                          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],PoolingLayer[2,2],
                                          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],
                                          ConvolutionLayer[cEncoder,{3,3}],BatchNormalizationLayer[],ParametricRampLayer[],PoolingLayer[2,2],
                                          FlattenLayer[],ReplicateLayer[2]},
                                   "MeanPart"->{PartLayer[1],BatchNormalizationLayer[],LinearLayer[200],Ramp,dLatent},
                                   "StdPart"->{PartLayer[2],BatchNormalizationLayer[],LinearLayer[200],Ramp,dLatent}|>,
        {NetPort["Input"]->"seq","seq"->"MeanPart"->NetPort["mn"],"seq"->"StdPart"->NetPort["sd"]
         },
        "Input"->dOutput]

Reparameterization=NetGraph[<|"LatentLoss"->latentLoss,
                                   "expSd"->ElementwiseLayer[Exp[#]&],
                                   "time"->ThreadingLayer[Times],
                                   "add"->ThreadingLayer[Plus]|>,
         {NetPort["mn"]->NetPort["LatentLoss","mn"],
          NetPort["sd"]->NetPort["LatentLoss","sd"],
          "LatentLoss"->NetPort["KL_Loss"],
          NetPort["sd"]->"expSd"->"time",
          NetPort["Random"]->"time"->"add",
          NetPort["mn"]->"add"},
        "mn"->dLatent,
        "sd"->dLatent,
        "Random"->dLatent]

VAEDynamicNet = NetInsertSharedArrays[DynamicNet, "DynamicNet/"];
LRDynamicNet = NetInsertSharedArrays[DynamicNet, "DynamicNet/"];

VAEEncoder = NetInsertSharedArrays[Encoder,"Encoder/"];
LREncoder = NetInsertSharedArrays[Encoder,"Encoder/"];

SRNet=NetInitialize[NetGraph[<|"VAE_DynamicNet"->VAEDynamicNet,
                "LR_DynamicNet"->LRDynamicNet,
                                "StaticNet"->StaticNet,
                                "VAE_Cate"->CatenateLayer[],
                                "VAE_Discriminator"->NetMapOperator[discriminator],
                                "VAE_Reshape"->ReshapeLayer[{2}],
                                "VAE_Scale" -> ConstantTimesLayer["Scaling" -> {-1, 1}],
                                "VAE_True"->PartLayer[1],
                                "VAE_False"->PartLayer[2],
                                "VAE_MSE"->MeanSquaredLossLayer[],
                                "VAE_Encoder"->VAEEncoder,
                                "Reparameterization"->Reparameterization,

                                "LR_Cate"->CatenateLayer[],
                                "LR_Discriminator"->NetMapOperator[discriminator],
                                "LR_Reshape"->ReshapeLayer[{2}],
                                "LR_Scale" -> ConstantTimesLayer["Scaling" -> {-1, 1}],
                                "LR_True"->PartLayer[1],
                                "LR_False"->PartLayer[2],
                                "LR_Encoder"->LREncoder,
                                "LR_MSE"->MeanSquaredLossLayer[]|>,
        {NetPort["Dynamics"]->NetPort["VAE_DynamicNet","Dynamics"],
         NetPort["Static"]->NetPort["StaticNet","Static"],
         NetPort["StaticNet","Down3"]->NetPort["VAE_DynamicNet","Static0"],
         NetPort["VAE_DynamicNet","P_0"]->"VAE_Cate",
         NetPort["Po_0"]->"VAE_Cate"->"VAE_Discriminator"->"VAE_Reshape"->"VAE_Scale",
         "VAE_Scale"->"VAE_True"->NetPort["VAE_True_Loss"],
         "VAE_Scale"->"VAE_False"->NetPort["VAE_Fake_Loss"],
         NetPort["Po_0"]->"VAE_MSE",
         NetPort["VAE_DynamicNet","P_0"]->"VAE_MSE"->NetPort["VAE_MSE_Loss"],
         NetPort["Po_0"]->NetPort["VAE_Encoder","Input"],
         NetPort["VAE_Encoder","mn"]->NetPort["Reparameterization","mn"],
         NetPort["VAE_Encoder","sd"]->NetPort["Reparameterization","sd"],
         NetPort["VAE_Random"]->NetPort["Reparameterization","Random"],
         NetPort["Reparameterization","Output"]->NetPort["VAE_DynamicNet","Latent"],

         NetPort["Dynamics"]->NetPort["LR_DynamicNet","Dynamics"],
         NetPort["StaticNet","Down3"]->NetPort["LR_DynamicNet","Static0"],
         NetPort["LR_Random"]->NetPort["LR_DynamicNet","Latent"],
         NetPort["LR_DynamicNet","P_0"]->"LR_Cate",
         NetPort["Po_0"]->"LR_Cate"->"LR_Discriminator"->"LR_Reshape"->"LR_Scale",
         "LR_Scale"->"LR_True"->NetPort["LR_True_Loss"],
         "LR_Scale"->"LR_False"->NetPort["LR_Fake_Loss"],

         NetPort["LR_DynamicNet","P_0"]->"LR_Encoder",
         NetPort["LR_Encoder","mn"]->"LR_MSE",
         NetPort["LR_Random"]->"LR_MSE"->NetPort["LR_MSE_Loss"]},
        "Dynamics"->dDynamic,
        "Static"->dStatic,
        "Po_0"->dOutput,
        "VAE_Random"->dLatent,
        "LR_Random"->dLatent]]


months=months[[1;;-25]];

validation=Block[{select=30,day},
  day=Table[RandomSample[Range[Length[p[[i]]]],select],{i,-24,-1}];
  Flatten[Table[<|"Dynamics"->dynamics[[i,;;,(day[[i,j]]-1)*3+1;;day[[i,j]]*3]],
    "Static"->ele,
    "Po_0"->p[[i,day[[i,j]]]]|>,{j,select},{i,-24,-1}]]];

EnergyDistance[x_,y_]:=Block[{a,b,c},
 a=2*ParallelTable[Total[Flatten[(x[[i]]-y[[j]])^2]],{i,Length[x]},{j,Length[y]}];
 Print[Mean[Flatten[a]]];
 b=ParallelTable[Total[Flatten[(x[[i]]-x[[j]])^2]],{i,Length[x]},{j,Length[x]}];
 Print[Mean[Flatten[b]]];
 c=ParallelTable[Total[Flatten[(y[[i]]-y[[j]])^2]],{i,Length[y]},{j,Length[y]}];
 Print[Mean[Flatten[c]]];
 Mean[Flatten[a]]-Mean[Flatten[b]]-Mean[Flatten[c]]]

index=StringSplit[CreateUUID[],"-"][[1]];
globe=Infinity;
gcorr=-Infinity;

(*
ReportSRnet1[net_] :=
  Block[{obser,simu,snet,result,corr},
        obser=validation[[;;,"Po_0"]][[;;,1]];
        snet=NetTake[net,"LR_DynamicNet"];
        simu=Table[snet[<|"Dynamics"->validation[[i,"Dynamics"]],"Static"->validation[[i,"Static"]],"LR_Random"->Table[0,dLatent]|>, TargetDevice->"GPU"],{i,Length[validation]}][[;;,1]];
        corr=Table[Correlation[simu[[;;,i,j]],obser[[;;,i,j]]],{i,40},{j,80}];
        Print[Mean[Select[Flatten[corr],NumberQ]]];
        If[Mean[Select[Flatten[corr],NumberQ]]>gcorr,
          Block[{},
                Export["/data/home/scy0446/run/Code/trained_Bicycle_"<>index<>"_d.mx",net];
                Print["update"];
                Set[gcorr,Mean[Select[Flatten[corr],NumberQ]]];]]];

(*training*)
batch=64;
NetTrain[SRNet,
   {Function[Block[{Nmonth=8,m,d},
        m=RandomSample[Range[Length[months]],Nmonth];
        d=Map[RandomSample[Range[Length[p[[#]]]],batch/Nmonth]&,m];
        <|"Static"->Table[ele,batch],
          "Dynamics"->Flatten[Table[Table[dynamics[[m[[i]]]][[;;,(d[[i,j]]-1)*3+1;;d[[i,j]]*3]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "Po_0"->Flatten[Table[Table[p[[m[[i]]]][[d[[i,j]]]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "LR_Random"->Table[Table[0,dLatent],batch],
          "VAE_Random"->Table[Table[0,dLatent],batch]|>]],
       "RoundLength" -> Total[Map[Length[#]&,p]]},
    LossFunction ->{"VAE_Fake_Loss"->Scaled[0],"VAE_True_Loss"->Scaled[0],"VAE_MSE_Loss"->Scaled[-10],"KL_Loss"->Scaled[0],
                    "LR_Fake_Loss"->Scaled[0],"LR_True_Loss"->Scaled[0],"LR_MSE_Loss"->Scaled[0]},
    TrainingUpdateSchedule -> {"VAE_Encoder","LR_Encoder","StaticNet","VAE_DynamicNet","LR_DynamicNet"},
    LearningRateMultipliers -> {"VAE_Discriminator" -> 0, "LR_Discriminator"->0,
                                "VAE_Encoder"->-1,"LR_Encoder"->-1,
                                "StaticNet"->-1,
                                "VAE_DynamicNet"->-1,"LR_DynamicNet"->-1,
                                "VAE_Scale"->0,"LR_Scale"->0},
    BatchSize -> 64,
    TargetDevice->{"GPU",All},
    MaxTrainingRounds->100,
    Method -> {"ADAM", "Beta1" -> 0.5, "LearningRate" -> 10^-4},
    TrainingProgressReporting->{{Function@ReportSRnet1[#Net], "Interval" -> Quantity[100, "Batches"]},"Print"}];

SRNet=Import["/data/home/scy0446/run/Code/trained_Bicycle_"<>index<>"_d.mx"];
*)

gcorr=-Infinity;
ReportSRnet2[net_] :=
  Block[{obser,simu,snet,result,corr},
        obser=validation[[;;,"Po_0"]][[;;,1]];
        snet=NetTake[net,"LR_DynamicNet"];
        simu=Table[snet[<|"Dynamics"->validation[[i,"Dynamics"]],"Static"->validation[[i,"Static"]],"LR_Random"->RandomReal[NormalDistribution[0,1],dLatent]|>, TargetDevice->"GPU"],{i,Length[validation]}][[;;,1]];
        corr=Table[Correlation[simu[[;;,i,j]],obser[[;;,i,j]]],{i,40},{j,80}];
        Print[Mean[Select[Flatten[corr],NumberQ]]];
        result=EnergyDistance[Normal[obser],Normal[simu]];
        Print[{globe,result}];
        If[result<globe,
          Block[{},
                Export["/data/home/scy0446/run/Code/trained_Bicycle_"<>index<>".mx",net];
                Print["update"];
                Set[globe,result];]]];


ReportSRnet2[net_] :=
  Block[{obser,simu,snet,result,corr},
        obser=validation[[;;,"Po_0"]][[;;,1]];
        snet=NetTake[net,"LR_DynamicNet"];
        simu=Table[snet[<|"Dynamics"->validation[[i,"Dynamics"]],"Static"->validation[[i,"Static"]],"LR_Random"->RandomReal[NormalDistribution[0,1],dLatent]|>, TargetDevice->"GPU"],{i,Length[validation]}][[;;,1]];
        corr=Quiet[Table[Correlation[simu[[;;,i,j]],obser[[;;,i,j]]],{i,40},{j,80}]];
        Print[Mean[Select[Flatten[corr],NumberQ]]];
        Print[{gcorr,Mean[Select[Flatten[corr],NumberQ]]}];
        If[gcorr<Mean[Select[Flatten[corr],NumberQ]],
          Block[{},
                Export["/data/home/scy0446/run/Code/trained_Bicycle_"<>index<>".mx",net];
                Print["update"];
                Set[gcorr,Mean[Select[Flatten[corr],NumberQ]]];]]];

(*training*)
batch=128;
NetTrain[SRNet,
   {Function[Block[{Nmonth=8,m,d},
        m=RandomSample[Range[Length[months]],Nmonth];
        d=Map[RandomSample[Range[Length[p[[#]]]],batch/Nmonth]&,m];
        <|"Static"->Table[ele,batch],
          "Dynamics"->Flatten[Table[Table[dynamics[[m[[i]]]][[;;,(d[[i,j]]-1)*3+1;;d[[i,j]]*3]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "Po_0"->Flatten[Table[Table[p[[m[[i]]]][[d[[i,j]]]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "LR_Random"->Table[RandomReal[NormalDistribution[0,1],dLatent],batch],
          "VAE_Random"->Table[RandomReal[NormalDistribution[0,1],dLatent],batch]|>]],
       "RoundLength" -> Total[Map[Length[#]&,p]]},
    LossFunction ->{"VAE_Fake_Loss"->Scaled[1],"VAE_True_Loss"->Scaled[1],"VAE_MSE_Loss"->Scaled[-100],"KL_Loss"->Scaled[-1],
"Bicycle_GAN_3.m" 279L, 14172C
