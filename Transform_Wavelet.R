library(waveslim)


args = commandArgs(trailingOnly = TRUE)
infile <- as.numeric(args[1])
Level=infile

infile <- as.numeric(args[2])
num_=infile

infile <- as.numeric(args[3])
class_type=infile

infile <- as.numeric(args[4])
Ts_tr=infile


if (class_type==0) { S="neut_"} else {S="sweep_"}

if (Ts_tr==0) { R="train_"} else {R="test_"}

M <- read.csv(paste0("./Data/CSV_files/",S,R,"Processed_1.csv"), header = TRUE, row.names = 1)
wS<-dwt.2d(as.matrix(M), Level)
d<-t(unlist(wS))
write.table(d, file = paste0("./Data/Wavelets_",S,R,".csv"), sep = ",", append = FALSE,row.names = FALSE, quote = FALSE, col.names = FALSE)


for (i in 2:num_)
{
  M <- read.csv(paste0("./Data/CSV_files/",S,R,"Processed_", i, ".csv"), header = TRUE, row.names = 1)
  wS<-dwt.2d(as.matrix(M), Level)
  d<-t(unlist(wS))
  write.table(d, file = paste0("./Data/Wavelets_",S,R,".csv"), sep = ",", append = TRUE,row.names = FALSE, quote = FALSE, col.names = FALSE)
}

