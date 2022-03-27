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
p=Table[Block[{},Print[months[[i]]];Import["N_"<>DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["data"]],{i,Length[months]}];

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
  day=Table[RandomSample[Range[Length[p[[i,1]]]],select],{i,-24,-1}];
  Flatten[Table[<|"Dynamics"->dynamics[[i,;;,(day[[i,j]]-1)*3+1;;day[[i,j]]*3]],
    "Static"->ele,
    "P_0"->p[[i,1,day[[i,j]]]],
    "P_1"->p[[i,2,day[[i,j]]]],
    "P_2"->p[[i,3,day[[i,j]]]],
    "P_3"->p[[i,4,day[[i,j]]]]|>,{j,select},{i,-24,-1}]]];

cStatic=4;
cDynamic=32;

StaticNet=Block[{c=cStatic},
        NetGraph[
 <|"resize"->{ResizeLayer[{320,640}],ConvolutionLayer[c,{1,1}],BatchNormalizationLayer[], ParametricRampLayer[]},
   "down1"->{Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],BatchNormalizationLayer[],ParametricRampLayer[]},3],PoolingLayer[2,2]},
   "down2"->{Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],BatchNormalizationLayer[],ParametricRampLayer[]},3],PoolingLayer[2,2]},
   "down3"->{Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],BatchNormalizationLayer[],ParametricRampLayer[]},3],PoolingLayer[2,2]}|>,
        {NetPort["Static"]->"resize"->NetPort["Down0"],
        "resize"->"down1"->NetPort["Down1"],
        "down1"->"down2"->NetPort["Down2"],
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
   "output0"->{ConvolutionLayer[c,{1,1}],BatchNormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],BatchNormalizationLayer[],Ramp},

   "resize1"->DeconvolutionLayer[c,{2,2},2],
   "cate1"->CatenateLayer[],
   "up1"->{res[c2],res[c2]},
   "output1"->{ConvolutionLayer[c,{1,1}],BatchNormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],BatchNormalizationLayer[],Ramp},

   "resize2"->DeconvolutionLayer[c,{2,2},2],
   "cate2"->CatenateLayer[],
   "up2"->{res[c2],res[c2]},
   "output2"->{ConvolutionLayer[c,{1,1}],BatchNormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],BatchNormalizationLayer[],Ramp},

   "resize3"->DeconvolutionLayer[c,{2,2},2],
   "cate3"->CatenateLayer[],
   "up3"->{res[c2],res[c2]},
   "output3"->{ResizeLayer[{301,621}],ConvolutionLayer[c,{1,1}],BatchNormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],BatchNormalizationLayer[],Ramp}|>,

        {NetPort["Dynamics"]->"resize0"->"cate0",
         NetPort["Static0"]->"cate0"->"up0"->"output0"->NetPort["P_0"],

         "up0"->"resize1"->"cate1",
         NetPort["Static1"]->"cate1"->"up1"->"output1"->NetPort["P_1"],

         "up1"->"resize2"->"cate2",
         NetPort["Static2"]->"cate2"->"up2"->"output2"->NetPort["P_2"],

         "up2"->"resize3"->"cate3",
         NetPort["Static3"]->"cate3"->"up3"->"output3"->NetPort["P_3"]},
"Dynamics"->{15,3,59,123}]]

SRNet=NetInitialize[NetGraph[<|"DynamicNet"->DynamicNet,
                                "StaticNet"->StaticNet|>,
        {NetPort["Dynamics"]->NetPort["DynamicNet","Dynamics"],
         NetPort["Static"]->NetPort["StaticNet","Static"],
         NetPort["StaticNet","Down0"]->NetPort["DynamicNet","Static3"],
         NetPort["StaticNet","Down1"]->NetPort["DynamicNet","Static2"],
         NetPort["StaticNet","Down2"]->NetPort["DynamicNet","Static1"],
         NetPort["StaticNet","Down3"]->NetPort["DynamicNet","Static0"]},
        "Dynamics"->{15,3,59,123},
        "Static"->{1,301,621}]]
SRNet=Import["/data/home/scy0446/run/Code/trained_Res_f2e86701.mx"];

globalmse={Infinity,Infinity,Infinity,Infinity};
index=StringSplit[CreateUUID[],"-"][[1]];
ReportSRnet[net_] :=
  Block[{obserP0,obserP1,obserP2,obserP3,simuP0,simuP1,simuP2,simuP3,simu,mse,corr},
        obserP0=validation[[;;,"P_0"]];
        obserP1=validation[[;;,"P_1"]];
        obserP2=validation[[;;,"P_2"]];
        obserP3=validation[[;;,"P_3"]];
        simu=Map[net[<|"Dynamics"->#[["Dynamics"]],"Static"->#[["Static"]]|>,TargetDevice->"GPU"]&,validation];
        {simuP0,simuP1,simuP2,simuP3}=Map[simu[[;;,#]]&,{"P_0","P_1","P_2","P_3"}];
        {simuP0,simuP1,simuP2,simuP3}=Normal[{simuP0,simuP1,simuP2,simuP3}];
        {obserP0,obserP1,obserP2,obserP3}=Normal[{obserP0,obserP1,obserP2,obserP3}];
        mse={Mean[Flatten[(simuP0-obserP0)^2]],Mean[Flatten[(simuP1-obserP1)^2]],
             Mean[Flatten[(simuP2-obserP2)^2]],Mean[Flatten[(simuP3-obserP3)^2]]};
        Print[{mse,globalmse}];
        If[Total[mse]<Total[globalmse],
          Block[{},
                Export["/data/home/scy0446/run/Code/trained_Res_"<>index<>".mx",net];
                (*
                corr=Quiet[{Table[Correlation[simuP0[[;;,1,i,j]],obserP0[[;;,1,i,j]]],{i,Dimensions[simuP0][[3]]},{j,Dimensions[simuP0][[4]]}],
                      Table[Correlation[simuP1[[;;,1,i,j]],obserP1[[;;,1,i,j]]],{i,Dimensions[simuP1][[3]]},{j,Dimensions[simuP1][[4]]}],
                      Table[Correlation[simuP2[[;;,1,i,j]],obserP2[[;;,1,i,j]]],{i,Dimensions[simuP2][[3]]},{j,Dimensions[simuP2][[4]]}],
                      Table[Correlation[simuP3[[;;,1,i,j]],obserP3[[;;,1,i,j]]],{i,Dimensions[simuP3][[3]]},{j,Dimensions[simuP3][[4]]}]}];
                Print[Map[Mean[Select[Flatten[#],NumberQ]]&,corr]];
                *)
                Print["update"];
                Set[globalmse,mse];]]];

batch=128;
NetTrain[SRNet,
    {Function[Block[{Nmonth=8,m,d},
        m=RandomSample[Range[Length[months]],Nmonth];
        d=Map[RandomSample[Range[Length[p[[#,1]]]],batch/Nmonth]&,m];
        <|"Static"->Table[ele,batch],
          "Dynamics"->Flatten[Table[Table[dynamics[[m[[i]]]][[;;,(d[[i,j]]-1)*3+1;;d[[i,j]]*3]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "P_0"->Flatten[Table[Table[p[[m[[i]]]][[1]][[d[[i,j]]]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "P_1"->Flatten[Table[Table[p[[m[[i]]]][[2]][[d[[i,j]]]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "P_2"->Flatten[Table[Table[p[[m[[i]]]][[3]][[d[[i,j]]]],{j,Length[d[[i]]]}],{i,Length[m]}]],
          "P_3"->Flatten[Table[Table[p[[m[[i]]]][[4]][[d[[i,j]]]],{j,Length[d[[i]]]}],{i,Length[m]}]]|>]],
       "RoundLength" -> Total[Map[Length[#[[1]]]&,p]]},
    BatchSize -> batch,
    TargetDevice->{"GPU",All},
    ValidationSet->validation,
    MaxTrainingRounds->400,
    TrainingProgressReporting->{{Function@ReportSRnet[#Net], "Interval" -> Quantity[50, "Batches"]},"Print"}];
~
~
