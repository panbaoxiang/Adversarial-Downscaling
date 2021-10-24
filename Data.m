SetDirectory["/Volumes/lambda/CFSR"];
vars={"z1000","z850","z500","z200","q925","q850","q500","prmsl"};

dates=DateRange[{1979,1,1},{2011,3,1},"Month"];
script=Flatten[Table[Block[{date=DateString[dates[[i]],{"Year","Month"}]},"wget -nc https://www.ncei.noaa.gov/data/climate-forecast-system/access/reanalysis/time-series/"<>date<>"/"<>vars[[j]]<>".gdas."<>date<>".grb2"],{i,Length[dates]},{j,Length[vars]}]];
Export["reanalysis_1979-2011.sh",script,"Table"]

dates=DateRange[{2011,4,1},{2020,9,1},"Month"];
script=Flatten[Table[Table[DateString[dates[[i]],{"wget -nc https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-analysis/time-series/","Year","/","Year","Month","/",vars[[j]],".gdas.","Year","Month",".grib2"}],{j,Length[vars]}],{i,Length[dates]}]];
Export["reanalysis_2011-2021.sh",script,"Table"]

<<JLink`;
InstallJava[];
ReinstallJava[JVMArguments -> "-Xmx6512m"];

vars={"z1000","z850","z500","z200","q925","q850","q500","prmsl"};
dates=DateRange[{1979,1,1},{2021,9,1},"Month"];
names=Map[Import,Table[FileNames[vars[[i]]<>"*"<>DateString[dates[[1]],{"Year","Month"}]<>"*"][[1]],{i,Length[vars]}]][[;;,1]];
{lat,lon}=Import["z1000.gdas.197901.grb2",{"Datasets",{"lat","lon"}}];
scope={{22., 51.}, {233., 294.}};

position={{Position[lat,scope[[1,1]]][[1,1]],Position[lat,scope[[1,2]]][[1,1]]},{Position[lon,scope[[2,1]]][[1,1]],Position[lon,scope[[2,2]]][[1,1]]}}

Table[Block[{files},
	<<JLink`;InstallJava[];ReinstallJava[JVMArguments -> "-Xmx6512m"];
	files=Flatten[Table[FileNames[vars[[i]]<>"*"<>DateString[dates[[d]],{"Year","Month"}]<>"*grb2"],{i,Length[vars]}]];
	Print[Length[files]];Print[dates[[d]]];
	If[And[Not[FileExistsQ[DateString[dates[[d]],{"CONUS_","Year","_","Month",".mx"}]]],Length[files]==Length[vars]],
     Block[{data},
	 data=Table[Block[{tempt=Import[files[[i]],{"Datasets",names[[i]]}]},ArrayReshape[tempt,DeleteCases[Dimensions[tempt],1]]],{i,Length[names]}];
	 data=NumericArray[data[[;;,;;,position[[1,2]];;position[[1,1]],position[[2,1]];;position[[2,2]]]],"Real32"];
	 Export[DateString[dates[[d]],{"CONUS_","Year","_","Month",".mx"}],<|"data"->data,"description"->vars,"lat"->lat[[position[[1,2]];;position[[1,1]]]],"lon"->lon[[position[[2,1]];;position[[2,2]]]]|>];]]],{d,RandomSample[Range[Length[dates]]]}];
