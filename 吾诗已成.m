SetDirectory["/data/home/scy0446/run/Data"];
months=DateRange[{1979,1,1},{2010,12,1},"Month"];
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

StaticNet=Block[{c=cStatic},NetGraph[<|"resize"->{ResizeLayer[{320,640}],ConvolutionLayer[c,{1,1}],NormalizationLayer[], ParametricRampLayer[]},
   "down1"->{Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],NormalizationLayer[],ParametricRampLayer[]},3],PoolingLayer[2,2]},
   "down2"->{Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],NormalizationLayer[],ParametricRampLayer[]},3],PoolingLayer[2,2]},
   "down3"->{Table[{ConvolutionLayer[c,{3,3},"PaddingSize"->1],NormalizationLayer[],ParametricRampLayer[]},3],PoolingLayer[2,2]}|>,
        {NetPort["Static"]->"resize"->NetPort["Down0"],
        "resize"->"down1"->NetPort["Down1"],
        "down1"->"down2"->NetPort["Down2"],
        "down2"->"down3"->NetPort["Down3"]},"Static"->dP]];

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
                           "thread"->ThreadingLayer[Plus],
                           "latent"->{res[16],res[32],res[64],res[128],res[256]},
                           "static"->StaticNet,
                           "thread2"->ThreadingLayer[Times],
                           "rep"->ReplicateLayer[{256,5,10}],
                           "output0"->{ConvolutionLayer[64,{1,1}],NormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],NormalizationLayer[],Ramp},
                           "output1"->{ConvolutionLayer[64,{1,1}],NormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],NormalizationLayer[],Ramp},
                           "output2"->{ConvolutionLayer[64,{1,1}],NormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],NormalizationLayer[],Ramp},
                           "output3"->{ResizeLayer[dP[[2;;]]],ConvolutionLayer[64,{1,1}],NormalizationLayer[],Ramp,ConvolutionLayer[1,{1,1}],NormalizationLayer[],Ramp}|>,
   {"merge"->NetPort["contract_1","Input"],
   NetPort["contract_1","Pooling"]->"contract_2",
   NetPort["contract_2","Pooling"]->"contract_3",
   NetPort["contract_3","Pooling"]->"thread",
   NetPort["Latent"]->"latent"->"thread2"->"thread"->"ubase"->NetPort["expand_3","Input"],
   NetPort["Switch"]->"rep"->"thread2",
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
   "expand_-3"->"output3"->NetPort["P3"],
   "expand_-2"->"output2"->NetPort["P2"],
   "expand_-1"->"output1"->NetPort["P1"],
   "expand_0"->"output0"->NetPort["P0"]}]]

EncoderNet=NetGraph[<|"long"->{ResizeLayer[{320,640}],res[64],PoolingLayer[2,2],
                                                     res[64],PoolingLayer[2,2],res[64],PoolingLayer[2,2],
                                                     res[64],PoolingLayer[2,2],res[64],PoolingLayer[2,2],res[64],PoolingLayer[2,2]},
                     "part1"->{ConvolutionLayer[64,{1,1}],NormalizationLayer[],Ramp,ConvolutionLayer[dLatent,{1,1}]},
                     "part2"->{ConvolutionLayer[64,{1,1}],NormalizationLayer[],Ramp,ConvolutionLayer[dLatent,{1,1}]}|>,
 {NetPort["Input"]->"long"->"part1"->NetPort["mn"],"long"->"part2"->NetPort["sd"]},"Input"->dP];

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

discriminator[d_,size_:dP]:=Block[{dis},
        dis=Append[Table[{ConvolutionLayer[32,{3,3},"Stride"->1],BatchNormalizationLayer[],Ramp,PoolingLayer[2,2]},d],
                {FlattenLayer[],BatchNormalizationLayer[],150,Ramp, 1, ElementwiseLayer["HardSigmoid"]}];
        NetGraph[<|"cat"->CatenateLayer[0],
                "scale" ->{FlattenLayer[], ConstantTimesLayer["Scaling" -> {-1, 1}]},
                "dis"->NetMapOperator[dis],
                "true"->PartLayer[1],
                "fake"->PartLayer[2]|>,
                {NetPort["Input1"]->"cat",NetPort["Input2"]->"cat"->"dis"->"scale"->"true"->NetPort["True"],"scale"->"fake"->NetPort["Fake"]},
    "Input1"->size,"Input2"->size]]

DiscriminatorNet0=discriminator[3,{1,40,80}];
DiscriminatorNet1=discriminator[4,{1,80,160}];
DiscriminatorNet2=discriminator[5,{1,160,320}];
DiscriminatorNet3=discriminator[6,{1,301,621}];


dynamic=NetInsertSharedArrays[DynamicNet,"dshare"];
encoder=NetInsertSharedArrays[EncoderNet,"eshare"];
sigma=7.0*10^-2;

net=NetInitialize[NetGraph[<|
                           "dynamic"->dynamic,
                           "encoder"->encoder,
                           "reparameterization"->Reparameterization,
                           "P0"->ResizeLayer[{40,80}],
                           "P1"->ResizeLayer[{80,160}],
                           "P2"->ResizeLayer[{160,320}],

                           "MSE10"->MeanSquaredLossLayer[],
                           "MSE11"->MeanSquaredLossLayer[],
                           "MSE12"->MeanSquaredLossLayer[],
                           "MSE13"->MeanSquaredLossLayer[],

                           "D10"->DiscriminatorNet0,
                           "D11"->DiscriminatorNet1,
                           "D12"->DiscriminatorNet2,
                           "D13"->DiscriminatorNet3,

                           "dynamic2"->dynamic,
                           "MSE20"->MeanSquaredLossLayer[],
                           "MSE21"->MeanSquaredLossLayer[],
                           "MSE22"->MeanSquaredLossLayer[],
                           "MSE23"->MeanSquaredLossLayer[],
                           "Hinge0"->ElementwiseLayer[Max[#,sigma]-sigma&],
                           "Hinge1"->ElementwiseLayer[Max[#,sigma]-sigma&],
                           "Hinge2"->ElementwiseLayer[Max[#,sigma]-sigma&],
                           "Hinge3"->ElementwiseLayer[Max[#,sigma]-sigma&],

                           "D20"->DiscriminatorNet0,
                           "D21"->DiscriminatorNet1,
                           "D22"->DiscriminatorNet2,
                           "D23"->DiscriminatorNet3,

                           "encoder2"->encoder,
                           "latentloss"->MeanSquaredLossLayer[]|>,
 {NetPort["P"]->"encoder",
  NetPort["encoder","mn"]->NetPort["reparameterization","mn"],
  NetPort["encoder","sd"]->NetPort["reparameterization","sd"],
  NetPort["reparameterization","Output"]->NetPort["dynamic","Latent"],

  NetPort["P"]->"P0"->"MSE10"->NetPort["MSE0"],
  NetPort["dynamic","P0"]->"MSE10",
  NetPort["P"]->"P1"->"MSE11"->NetPort["MSE1"],
  NetPort["dynamic","P1"]->"MSE11",
  NetPort["P"]->"P2"->"MSE12"->NetPort["MSE2"],
  NetPort["dynamic","P2"]->"MSE12",
  NetPort["P"]->"MSE13"->NetPort["MSE3"],
  NetPort["dynamic","P3"]->"MSE13",

  "P0"->NetPort["D10","Input1"],
  NetPort["dynamic","P0"]->NetPort["D10","Input2"],
  "P1"->NetPort["D11","Input1"],
  NetPort["dynamic","P1"]->NetPort["D11","Input2"],
  "P2"->NetPort["D12","Input1"],
  NetPort["dynamic","P2"]->NetPort["D12","Input2"],
  NetPort["P"]->NetPort["D13","Input1"],
  NetPort["dynamic","P3"]->NetPort["D13","Input2"],

  NetPort["Random2"]->NetPort["dynamic2","Latent"],
  "P0"->"MSE20"->"Hinge0"->NetPort["Hinge0"],
  NetPort["dynamic2","P0"]->"MSE20",
  "P1"->"MSE21"->"Hinge1"->NetPort["Hinge1"],
  NetPort["dynamic2","P1"]->"MSE21",
  "P2"->"MSE22"->"Hinge2"->NetPort["Hinge2"],
  NetPort["dynamic2","P2"]->"MSE22",
  NetPort["P"]->"MSE23"->"Hinge3"->NetPort["Hinge3"],
  NetPort["dynamic2","P3"]->"MSE23",

  "P0"->NetPort["D20","Input1"],
  NetPort["dynamic2","P0"]->NetPort["D20","Input2"],
  "P1"->NetPort["D21","Input1"],
  NetPort["dynamic2","P1"]->NetPort["D21","Input2"],
  "P2"->NetPort["D22","Input1"],
  NetPort["dynamic2","P2"]->NetPort["D22","Input2"],
  NetPort["P"]->NetPort["D23","Input1"],
  NetPort["dynamic2","P3"]->NetPort["D23","Input2"],

  NetPort["dynamic","P3"]->"encoder2",
  NetPort["encoder2","mn"]->"latentloss",
  NetPort["Random2"]->"latentloss"->NetPort["Latent_Loss"]}]]


length=Length[p];
sep=Round[length*0.8];
training=Table[<|"P"->p[[i]],"Dynamics"->dynamics[[i]]|>,{i,sep}];
validation=Table[<|"P"->p[[i]],"Dynamics"->dynamics[[i]]|>,{i,sep,length}];


batch=64;
GlobeLoss=Infinity;
Report[net_,ValidationLoss_]:=Block[{},Print[{GlobeLoss,ValidationLoss}];If[ValidationLoss<GlobeLoss,Block[{},Print["Update"];Set[GlobeLoss,ValidationLoss];Export["/data/home/scy0446/run/Code/2022_4_14_Stage1.mx",net]]]];

trained=NetTrain[net,
        {Function[Block[{select},
        select=RandomSample[Range[sep],batch];
        <|"Dynamics"->training[[select,"Dynamics"]],
          "Static"->Table[ele,batch],
          "P"->training[[select,"P"]],
          "Switch"->Table[0,batch],
          "Random"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],batch],
          "Random2"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],batch]|>]],
       "RoundLength" -> sep},
        MaxTrainingRounds->40,
        LossFunction ->{"Hinge0"->Scaled[1],"Hinge1"->Scaled[1],"Hinge2"->Scaled[1],"Hinge3"->Scaled[1],
                        "MSE0"->Scaled[1],"MSE1"->Scaled[1],"MSE2"->Scaled[1],"MSE3"->Scaled[1],
                        "KL_Loss"->Scaled[0],"Latent_Loss"->Scaled[0],
                        "True1"->Scaled[0],"Fake1"->Scaled[0],
                        "True2"->Scaled[0],"Fake2"->Scaled[0],
                        "True3"->Scaled[0],"Fake3"->Scaled[0],
                        "True4"->Scaled[0],"Fake4"->Scaled[0],
                        "True5"->Scaled[0],"Fake5"->Scaled[0],
                        "True6"->Scaled[0],"Fake6"->Scaled[0],
                        "True7"->Scaled[0],"Fake7"->Scaled[0],
                        "True8"->Scaled[0],"Fake8"->Scaled[0]},
        LearningRateMultipliers -> {"encoder"->-0,"encoder2"->0,"dynamic"->1,"dynamic2"->1,
                                    "D10"->0,"D11"->0,"D12"->0,"D13"->0,
                                    "D20"->0,"D21"->0,"D22"->0,"D23"->0},
        TargetDevice->{"GPU",All},
        Method -> {"ADAM", "Beta1" -> 0.5, "LearningRate" -> 10^-4},
        BatchSize->batch,
        TrainingProgressReporting->{{Function@Report[#Net,#ValidationLoss], "Interval" -> Quantity[1, "Rounds"]},"Print"}];


(*
net=Import["/data/home/scy0446/run/Code/2022_4_14_Stage1.mx"];
trained=NetTrain[net,
        {Function[Block[{select},
        select=RandomSample[Range[sep],batch];
        <|"Dynamics"->training[[select,"Dynamics"]],
          "Static"->Table[ele,batch],
          "P"->training[[select,"P"]],
          "Switch"->Table[1,batch],
          "Random"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],batch],
          "Random2"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],batch]|>]],
       "RoundLength" -> sep},
        MaxTrainingRounds->200,
        LossFunction ->{"Hinge0"->Scaled[-1],"Hinge1"->Scaled[-1],"Hinge2"->Scaled[-1],"Hinge3"->Scaled[-1],
                        "MSE0"->Scaled[-1],"MSE1"->Scaled[-1],"MSE2"->Scaled[-1],"MSE3"->Scaled[-1],
                        "KL_Loss"->Scaled[-.0001],"Latent_Loss"->Scaled[-.01],
                        "True1"->Scaled[0.1],"Fake1"->Scaled[0.1],
                        "True2"->Scaled[0.1],"Fake2"->Scaled[0.1],
                        "True3"->Scaled[0.1],"Fake3"->Scaled[0.1],
                        "True4"->Scaled[0.1],"Fake4"->Scaled[0.1],
                        "True5"->Scaled[0.1],"Fake5"->Scaled[0.1],
                        "True6"->Scaled[0.1],"Fake6"->Scaled[0.1],
                        "True7"->Scaled[0.1],"Fake7"->Scaled[0.1],
                        "True8"->Scaled[0.1],"Fake8"->Scaled[0.1]},
        LearningRateMultipliers -> {"encoder"->-1,"encoder2"->0,"dynamic"->-1,"dynamic2"->-1,
                                    "D10"->1,"D11"->1,"D12"->1,"D13"->1,
                                    "D20"->1,"D21"->1,"D22"->1,"D23"->1,
                                    {"D10","scale"}->0,{"D11","scale"}->0,{"D12","scale"}->0,{"D13","scale"}->0,
                                    {"D20","scale"}->0,{"D21","scale"}->0,{"D22","scale"}->0,{"D23","scale"}->0},
        TargetDevice->{"GPU",All},
        Method -> {"ADAM", "Beta1" -> 0.5, "LearningRate" -> 10^-4, "GradientClipping" -> {"D10"->1*10^-5,"D11"->1*10^-5,"D12"->1*10^-5,"D13"->1*10^-5,"D20"->1*10^-5,"D21"->1*10^-5,"D22"->1*10^-5,"D23"->1*10^-5}},
        BatchSize->batch];


net=trained;
(*Export["/data/home/scy0446/run/Code/2022_4_14_Stage2.mx",trained];*)
net=NetExtract[net,"dynamic"];
m=1;
l=RandomSample[Range[Length[validation]],200];
obser=validation[[l,"P"]];
simu=Table[Print[i];
        net[<|"Latent"->Table[RandomReal[NormalDistribution[0,1],{dLatent,5,10}],m],
                "Dynamics"->Table[validation[[i,"Dynamics"]],m],
                "Switch"->Table[0.,m],
                "Static"->Table[ele,m]|>,TargetDevice->"GPU"],{i,l}];
Export["/data/home/scy0446/run/simu.mx",simu];
msimu=Map[Mean,Normal[simu[[;;,"P3"]]]];
obser=Normal[obser];
corr=Quiet[Table[Correlation[msimu[[;;,1,i,j]],obser[[;;,1,i,j]]],{i,Dimensions[obser][[3]]},{j,Dimensions[obser][[4]]}]];
Print[Mean[Select[Flatten[corr],NumberQ]]]
Export["/data/home/scy0446/run/tempt.mx",{msimu,obser}]
*)
~
~
~
~
~
~
~
~
~
