(*regression model*)
SetDirectory["/data/home/scy0446/run/Data"];
months=DateRange[{2004,1,1},{2006,12,1},"Month"];
p=Table[Block[{},Print[months[[i]]];Import["N_"<>DateString[months[[i]],{"CONUS_","Year","Month",".mx"}]]["data"]],{i,Length[months]}];

ele=NumericArray[{Log[Normal[Import["/data/home/scy0446/run/Data/CONUS/Elevation.mx"]["data"]]+1.]},"Real32"];
dmeanvar=Import["/data/home/scy0446/run/Data/CONUS/meanvar_dynamics.mx"]["data"];
SetDirectory["/data/home/scy0446/run/Data/CONUS"];
var={"pressfc","prmsl","pwat","q500","q850","q925","tmpsfc","u200","u850","v200","v850","z1000","z200","z500","z850"};

{dlat,dlon}=Block[{tempt=Import[DateString[{1979,1,1},{"D_CONUS_","Year","_","Month",".mx"}]]},{tempt["lat"],tempt["lon"]}];

dynamics=Table[Block[{},Print[months[[i]]];Import[DateString[months[[i]],{"D_CONUS_","Year","_","Month",".mx"}]]["data"]],{i,Length[months]}];

validation=Block[{select=30,day},
  day=Table[RandomSample[Range[Length[p[[i,1]]]],select],{i,-24,-1}];
  Flatten[Table[<|"Dynamics"->dynamics[[i,;;,(day[[i,j]]-1)*3+1;;day[[i,j]]*3]],
    "Static"->ele,
    "P_0"->p[[i,1,day[[i,j]]]],
    "P_1"->p[[i,2,day[[i,j]]]],
    "P_2"->p[[i,3,day[[i,j]]]],
    "P_3"->p[[i,4,day[[i,j]]]]|>,{j,select},{i,-24,-1}]]];

net=SRNet=Import["/data/home/scy0446/run/Code/trained_Res.mx"];
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
        Print[mse]

corr=Quiet[{Table[Correlation[simuP0[[;;,1,i,j]],obserP0[[;;,1,i,j]]],{i,Dimensions[simuP0][[3]]},{j,Dimensions[simuP0][[4]]}],
                      Table[Correlation[simuP1[[;;,1,i,j]],obserP1[[;;,1,i,j]]],{i,Dimensions[simuP1][[3]]},{j,Dimensions[simuP1][[4]]}],
                      Table[Correlation[simuP2[[;;,1,i,j]],obserP2[[;;,1,i,j]]],{i,Dimensions[simuP2][[3]]},{j,Dimensions[simuP2][[4]]}],
                      Table[Correlation[simuP3[[;;,1,i,j]],obserP3[[;;,1,i,j]]],{i,Dimensions[simuP3][[3]]},{j,Dimensions[simuP3][[4]]}]}];
Print[Map[Mean[Select[Flatten[#],NumberQ]]&,corr]];

Export["/data/home/scy0446/run/Regression_Result.mx",
 <|"simu"->Map[NumericArray[#,"Real32"]&,{simuP0,simuP1,simuP2,simuP3}],
   "obser"->Map[NumericArray[#,"Real32"]&,{obserP0,obserP1,obserP2,obserP3}],
   "description"->"mm/3h from 2004-2006 for validation set"|>];
