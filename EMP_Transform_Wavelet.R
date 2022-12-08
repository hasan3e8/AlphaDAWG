library(waveslim)


args = commandArgs(trailingOnly = TRUE)
infile <- as.numeric(args[1])
Level=infile

infile <- as.numeric(args[2])
num_= infile

M <- read.csv(paste0("./Data/CSV_files/output_1.csv"), header = TRUE, row.names = 1)
wS<-dwt.2d(as.matrix(M), Level)
d<-t(unlist(wS))
write.table(d, file = paste0("./Data/EMP_Wavelets_.csv"), sep = ",", append = FALSE,row.names = FALSE, quote = FALSE, col.names = FALSE)


for (i in 2:num_)
{
  M <- read.csv(paste0("./Data/CSV_files/output_", i, ".csv"), header = TRUE, row.names = 1)
  wS<-dwt.2d(as.matrix(M), Level)
  d<-t(unlist(wS))
  write.table(d, file = paste0("./Data/EMP_Wavelets_.csv"), sep = ",", append = TRUE,row.names = FALSE, quote = FALSE, col.names = FALSE)
}

