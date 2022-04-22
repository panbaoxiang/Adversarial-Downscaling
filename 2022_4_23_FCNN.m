(*import data*)
months=DateRange[{1979,1,1},{2010,12,1},"Month"];
SetDirectory["/data/home/scy0446/run/Data"];
dlat=Import["CONUS/D_CONUS_1979_01.mx"]["lat"];
dlon=Import["CONUS/D_CONUS_1979_01.mx"]["lon"];
plat=Import["N_CONUS_197901.mx"]["lat"];
plon=Import["N_CONUS_197901.mx"]["lon"][[1,;;,1]];
size={16,32};
dInput={15,3,16,32}
dOutput={80,160}

SetDirectory["/data/home/scy0446/run/Data"];
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

length=Length[p];
sep=Round[length*0.8];
(*starting from 2004.8.6*)
(*training=Table[<|"P"->p[[i]],"Dynamics"->dynamics[[i]],"Static"->ele|>,{i,sep}];*)
validation=Table[Block[{start,dPosition,pPosition},
        start={RandomSample[Range[Length[dlat]-size[[1]]]][[1]],RandomSample[Range[Length[dlon]-size[[2]]]][[1]]};
        dPosition={{start[[1]],start[[1]]+size[[1]]-1},{start[[2]],start[[2]]+size[[2]]-1}};
        pPosition={{Position[plat,dlat[[start[[1]]]]+0.25][[1,1]],Position[plat,dlat[[start[[1]]+size[[1]]-1]]-0.25][[1,1]]-1},
           {Position[plon,dlon[[start[[2]]]]-0.25][[1,1]],Position[plon,dlon[[start[[2]]+size[[2]]-1]]+0.25][[1,1]]-1}};
        <|"P"->p[[i,;;,pPosition[[1,1]];;pPosition[[1,2]],pPosition[[2,1]];;pPosition[[2,2]]]],
          "Dynamics"->dynamics[[i,;;,;;,dPosition[[1,1]];;dPosition[[1,2]],dPosition[[2,1]];;dPosition[[2,2]]]],
          "Static"->ele[[;;,pPosition[[1,1]];;pPosition[[1,2]],pPosition[[2,1]];;pPosition[[2,2]]]]|>],{i,sep+1,length},{sample,10}];

training=Table[Block[{start,dPosition,pPosition},
        start={RandomSample[Range[Length[dlat]-size[[1]]]][[1]],RandomSample[Range[Length[dlon]-size[[2]]]][[1]]};
        dPosition={{start[[1]],start[[1]]+size[[1]]-1},{start[[2]],start[[2]]+size[[2]]-1}};
        pPosition={{Position[plat,dlat[[start[[1]]]]+0.25][[1,1]],Position[plat,dlat[[start[[1]]+size[[1]]-1]]-0.25][[1,1]]-1},
           {Position[plon,dlon[[start[[2]]]]-0.25][[1,1]],Position[plon,dlon[[start[[2]]+size[[2]]-1]]+0.25][[1,1]]-1}};
        <|"P"->p[[i,;;,pPosition[[1,1]];;pPosition[[1,2]],pPosition[[2,1]];;pPosition[[2,2]]]],
          "Dynamics"->dynamics[[i,;;,;;,dPosition[[1,1]];;dPosition[[1,2]],dPosition[[2,1]];;dPosition[[2,2]]]],
          "Static"->ele[[;;,pPosition[[1,1]];;pPosition[[1,2]],pPosition[[2,1]];;pPosition[[2,2]]]]|>],{i,sep},{sample,10}];

dInput={15,3,16,32};
dOutput={80,160};
res[c_]:=NetGraph[<|"long"->Flatten[Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],NormalizationLayer[],Ramp},2]][[1;;-2]],
                    "plus"->TotalLayer[],
                     "short"->ConvolutionLayer[c,{1,1}]|>,
   {NetPort["Input"]->"long"->"plus",NetPort["Input"]->"short"->"plus"}]
upres[c_,size_]:=NetGraph[<|"long"->{NormalizationLayer[],Ramp,ResizeLayer[size],ConvolutionLayer[c,{3,3},"PaddingSize"->1],
                                        NormalizationLayer[],Ramp,ConvolutionLayer[c,{3,3},"PaddingSize"->1]},
                            "plus"->TotalLayer[],
                           "short"->{ResizeLayer[size],ConvolutionLayer[c,{1,1}]}|>,
   {NetPort["Input"]->"long"->"plus",NetPort["Input"]->"short"->"plus"}]

StaticNet=Block[{c=8},NetGraph[
 <|"down0"->{res[c],res[c]},
   "down1"->{res[2c],PoolingLayer[2,2],res[2c]},
   "down2"->{res[4c],PoolingLayer[2,2],res[4c]},
   "down3"->{res[8c],PoolingLayer[2,2],res[8c]},
   "down4"->{res[16c],PoolingLayer[2,2],res[16c]}|>,
       {NetPort["Static"]->"down0"->NetPort["Down0"],
        "down0"->"down1"->NetPort["Down1"],
        "down1"->"down2"->NetPort["Down2"],
        "down2"->"down3"->NetPort["Down3"],
        "down3"->"down4"->NetPort["Down4"]},
"Static"->Prepend[dOutput,1]]];

DynamicNet=Block[{c=32}, NetInitialize[NetGraph[<|
        "StaticNet"->StaticNet,
        "DConv"->{Table[{ConvolutionLayer[c,{2,3,3}],NormalizationLayer[],Ramp},{i,2}],FlattenLayer[1],
                         res[c],res[2c],PoolingLayer[2,2],ConvolutionLayer[4c,{2,5}],res[8c],res[16c]},
    "Cate4"->CatenateLayer[],
    "Up4"->upres[8c,{10,20}],
    "Cate3"->CatenateLayer[],
    "Up3"->upres[4c,{20,40}],
    "Cate2"->CatenateLayer[],
    "Up2"->upres[2c,{40,80}],
    "Cate1"->CatenateLayer[],
    "Up1"->upres[c,{80,160}],
    "Cate0"->CatenateLayer[],
    "Up0"->{res[c],ConvolutionLayer[c,{1,1}],NormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],NormalizationLayer[],Ramp}|>,
{NetPort["Static"]->"StaticNet",
 NetPort["Dynamics"]->"DConv"->"Cate4",
 NetPort["StaticNet","Down4"]->"Cate4"->"Up4"->"Cate3",
 NetPort["StaticNet","Down3"]->"Cate3"->"Up3"->"Cate2",
 NetPort["StaticNet","Down2"]->"Cate2"->"Up2"->"Cate1",
 NetPort["StaticNet","Down1"]->"Cate1"->"Up1"->"Cate0",
 NetPort["StaticNet","Down0"]->"Cate0"->"Up0"->NetPort["P"]},
 "Dynamics"->dInput]]]


GlobeLoss=Infinity;
Report[net_,ValidationLoss_]:=Block[{},
        Print[{GlobeLoss,ValidationLoss}];
        If[ValidationLoss<GlobeLoss,Block[{},Print["Update"];
        Set[GlobeLoss,ValidationLoss];
        Export["/data/home/scy0446/run/Code/2023_4_23_Patch_ResNet.mx",net]]]];

batch=256;
trained=NetTrain[DynamicNet,RandomSample[Flatten[training]],
        ValidationSet->Flatten[validation],
        TargetDevice->{"GPU",All},
        BatchSize->batch,
        MaxTrainingRounds->20,
        TrainingProgressReporting->{{Function@Report[#Net,#ValidationLoss], "Interval" -> Quantity[1, "Rounds"]},"Print"}];
~
