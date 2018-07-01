#library
library(mxnet)
library(jpeg)

Path <- "Data/Pikachu/2. prosess data/"

############### TEST ################
#Read jpeg file and test
img <- readJPEG(source = paste0(Path, "jpeg_file/", 77,".jpeg"))

#Read label
LABEL <- read.table(file = paste0(Path, "label_file.txt"), header = T, stringsAsFactors = FALSE)

par(mar = c(0, 0, 0, 0))
plot(NA, xlim = 0:1, ylim = 0:1,xaxs="i", yaxs="i", xaxt = "n", yaxt = "n", bty = "n")
rasterImage( img, 0, 0, 1, 1, interpolate = FALSE)

#
rect( xleft = LABEL$xleft_1[77], xright = LABEL$xright_1[77],
      ybottom = LABEL$ybottom_1[77], ytop = LABEL$ytop_1[77], col = '#00A80050', lwd = 1.5)


