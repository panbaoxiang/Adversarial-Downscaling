<<JLink`;
InstallJava[];
ReinstallJava[JVMArguments -> "-Xmx56512m"];
SetSystemOptions["LibraryLinkOptions" -> "TestFloatingPointExceptions" -> False];
vars={"pressfc", "tmpsfc", "pwat", "prmsl", "q500", "q850", "q925", "u200", "u850", "z1000", "z200", "z500", "z850","v200","v850"};
size={16,32};
t=2;
dInput={15,3*t,16,32}
dOutput={80,160}

SetDirectory["/data/home/scy0446/run/China_Project"];
ele=Import["Elevation.mx"]["ele"];
{dlat,plat,dlon,plon}=Normal[Map[Import["NetCDF/199107.nc",{"Datasets",#}]&,{"/latD", "/latP", "/lonD", "/lonP"}]];
data=Block[{files,data,meanvar},
  files=Table[DateString[{year,month,1},{"NetCDF/","Year","Month",".nc"}],{year,1991,1999},{month,6,9}];
  data=Table[Block[{},
        Print[files[[i]]];
        {dynamic,p}=Block[{data},
                data=Table[Import[files[[i,j]],{"Datasets",{"/dynamics","/precipitation"}}],{j,Length[files[[i]]]}];
                {(Map[Transpose,data[[;;,1]]])/.List->Join,(data[[;;,2]])/.List->Join}]],{i,Length[files]}];
  dmeanvar=Block[{tempt=Transpose[Normal[data[[-3;;-1,1]]/.List->Join]]},
        Map[{Mean[Flatten[#]],Sqrt[Variance[Flatten[#]]]}&,tempt]];
  Table[Block[{dynamic,p},
        Print[files[[i]]];
        dynamic=Block[{tempt=Table[(Normal[data[[i,1,;;,v]]]-dmeanvar[[v,1]])/dmeanvar[[v,2]],{v,15}],l},
          tempt=Table[tempt[[;;,i;;If[i>3,-1,-4];;3]],{i,t*3}];
          tempt=NumericArray[tempt,"Real32"];
          Transpose[tempt,{3,2,1,4,5}]];
        p=NumericArray[Map[List,Log[Normal[data[[i,2,t;;]]]+1]],"Real32"];
        {dynamic,p}],{i,Length[data]}]];

allstarts=Block[{a,b},
        a=Range[2,Length[dlat]-size[[1]],Round[size[[1]]/1.12]];
        b=Range[2,Length[dlon]-size[[2]],Round[size[[2]]/1.05]];
        Flatten[Table[{a[[i]],b[[j]]},{i,Length[a]},{j,Length[b]}],1]];

{tYear,vYear}={{1,2,3,5,6,7,8},{4,9}};

validation=Table[Block[{start,dPosition,pPosition},
        start=allstarts[[sample]];
        dPosition={{start[[1]],start[[1]]+size[[1]]-1},{start[[2]],start[[2]]+size[[2]]-1}};
        pPosition={{Position[plat,dlat[[start[[1]]]]+0.25][[1,1]],Position[plat,dlat[[start[[1]]+size[[1]]-1]]-0.25][[1,1]]-1},
           {Position[plon,dlon[[start[[2]]]]-0.25][[1,1]],Position[plon,dlon[[start[[2]]+size[[2]]-1]]+0.25][[1,1]]-1}};
        Table[<|"P"->data[[i,2]][[date,;;,pPosition[[1,1]];;pPosition[[1,2]],pPosition[[2,1]];;pPosition[[2,2]]]],
          "Dynamics"->data[[i,1]][[date,;;,;;,dPosition[[1,1]];;dPosition[[1,2]],dPosition[[2,1]];;dPosition[[2,2]]]],
          "Static"->ele[[;;,pPosition[[1,1]];;pPosition[[1,2]],pPosition[[2,1]];;pPosition[[2,2]]]]|>,{date,Length[data[[i,1]]]}]],{i,vYear},{sample,Length[allstarts]}];

DynamicNet=Import["/data/home/scy0446/run/China_Project/Trained.mx"];

GlobeLoss=Infinity;
Report[net_,ValidationLoss_]:=Block[{},
        Print[{GlobeLoss,ValidationLoss}];
        If[ValidationLoss<GlobeLoss,Block[{},Print["Update"];
        Set[GlobeLoss,ValidationLoss];
        Export["/data/home/scy0446/run/China_Project/trained_"<>StringSplit[CreateUUID[],"-"][[1]]<>".mx",net]]]];

batch=128;
trained=NetTrain[DynamicNet,
    {Function[Block[{Pselect,Tselect},
        Pselect=Table[{RandomSample[Range[2,Length[dlat]-size[[1]]]][[1]],RandomSample[Range[2,Length[dlon]-size[[2]]]][[1]]},batch];
        Tselect=Table[Block[{year=RandomSample[tYear][[1]]},{year,RandomSample[Range[Length[data[[year,1]]]]][[1]]}],batch];
        Table[Block[{start,dPosition,pPosition},
        start=Pselect[[i]];
        dPosition={{start[[1]],start[[1]]+size[[1]]-1},{start[[2]],start[[2]]+size[[2]]-1}};
        pPosition={{Position[plat,dlat[[start[[1]]]]+0.25][[1,1]],Position[plat,dlat[[start[[1]]+size[[1]]-1]]-0.25][[1,1]]-1},
           {Position[plon,dlon[[start[[2]]]]-0.25][[1,1]],Position[plon,dlon[[start[[2]]+size[[2]]-1]]+0.25][[1,1]]-1}};
        <|"P"->data[[Tselect[[i,1]],2]][[Tselect[[i,2]],;;,pPosition[[1,1]];;pPosition[[1,2]],pPosition[[2,1]];;pPosition[[2,2]]]],
          "Dynamics"->data[[Tselect[[i,1]],1]][[Tselect[[i,2]],;;,;;,dPosition[[1,1]];;dPosition[[1,2]],dPosition[[2,1]];;dPosition[[2,2]]]],
          "Static"->ele[[;;,pPosition[[1,1]];;pPosition[[1,2]],pPosition[[2,1]];;pPosition[[2,2]]]]|>],{i,batch}]]],
       "RoundLength" -> 100000},
        ValidationSet->Flatten[validation],
        BatchSize->batch,
        TargetDevice->{"GPU",All},
        MaxTrainingRounds->100,
        Method->{"ADAM","L2Regularization"->10^-5},
        TrainingProgressReporting->{{Function@Report[#Net,#ValidationLoss], "Interval" -> Quantity[1, "Rounds"]},"Print"}];

(*
DynamicNet=trained;
Table[Block[{simu,obser},
simu=Map[trained[#,TargetDevice->"GPU"]&,validation[[year,i]]];
obser=validation[[year,i,;;,"P"]];
Print[{year,i,Correlation[Map[Mean[Flatten[Normal[#]]]&,simu],Map[Mean[Flatten[Normal[#]]]&,obser]]}]],{year,Length[validation]},{i,20}]




res[c_]:=NetGraph[<|"long"->Flatten[Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],NormalizationLayer[],Ramp},2]][[1;;-2]],
                    "plus"->TotalLayer[],
                    "short"->ConvolutionLayer[c,{1,1}]|>,
   {NetPort["Input"]->"long"->"plus",NetPort["Input"]->"short"->"plus"}]
upres[c_]:=NetGraph[<|"long"->{NormalizationLayer[],Ramp,DeconvolutionLayer[c,{2,2},"Stride"->2],ConvolutionLayer[c,{3,3},"PaddingSize"->1],
                                        NormalizationLayer[],Ramp,ConvolutionLayer[c,{3,3},"PaddingSize"->1]},
                            "plus"->TotalLayer[],
                           "short"->{DeconvolutionLayer[c,{2,2},"Stride"->2],ConvolutionLayer[c,{1,1}]}|>,
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
    "Up4"->upres[8c],
    "Cate3"->CatenateLayer[],
    "Up3"->upres[4c],
    "Cate2"->CatenateLayer[],
    "Up2"->upres[2c],
    "Cate1"->CatenateLayer[],
    "Up1"->upres[c],
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
        Export["/data/home/scy0446/run/Code/2022_5_09_Model_1_CNN_t_"<>ToString[t]<>"_"<>StringSplit[CreateUUID[],"-"][[1]]<>".mx",net]]]];

Print["start training"];
batch=128;
trained=NetTrain[DynamicNet,
  {Function[Block[{select,start,dPosition,pPosition},
        select=RandomSample[Range[2,sep],batch];
        start=Table[{RandomSample[Range[Length[dlat]-size[[1]]]][[1]],RandomSample[Range[Length[dlon]-size[[2]]]][[1]]},{batch}];
        dPosition=Table[{{start[[i,1]],start[[i,1]]+size[[1]]-1},{start[[i,2]],start[[i,2]]+size[[2]]-1}},{i,batch}];
        pPosition=Table[{{Position[plat,dlat[[start[[i,1]]]]+0.25][[1,1]],Position[plat,dlat[[start[[i,1]]+size[[1]]-1]]-0.25][[1,1]]-1},
           {Position[plon,dlon[[start[[i,2]]]]-0.25][[1,1]],Position[plon,dlon[[start[[i,2]]+size[[2]]-1]]+0.25][[1,1]]-1}},{i,batch}];
        <|"Dynamics"->Table[Transpose[dynamics[[(select[[i]]-t)*3+1;;select[[i]]*3]]][[;;,;;,dPosition[[i,1,1]];;dPosition[[i,1,2]],dPosition[[i,2,1]];;dPosition[[i,2,2]]]],{i,batch}],
          "Static"->Table[ele[[;;,pPosition[[i,1,1]];;pPosition[[i,1,2]],pPosition[[i,2,1]];;pPosition[[i,2,2]]]],{i,batch}],
          "P"->Table[p[[select[[i]]]][[;;,pPosition[[i,1,1]];;pPosition[[i,1,2]],pPosition[[i,2,1]];;pPosition[[i,2,2]]]],{i,batch}]|>]],
       "RoundLength" -> sep*5},
        ValidationSet->Flatten[validation],
        TargetDevice->{"GPU",All},
        BatchSize->batch,
        MaxTrainingRounds->50,
        Method->{"ADAM","L2Regularization"->10^-5},
        TrainingProgressReporting->{{Function@Report[#Net,#ValidationLoss], "Interval" -> Quantity[1, "Rounds"]},"Print"}];
~
~
~
~
~
~
~
~
~
