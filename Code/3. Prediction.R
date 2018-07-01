## Library

library(mxnet)
library(OpenImageR)

## load data

label_dat <- read.table('Data/Pikachu/2. prosess data/label_file.txt', header = TRUE)
img_array <- array(0, dim = c(256, 256, 3, nrow(label_dat)))

for (i in 1:nrow(label_dat)) {img_array[,,,i] <- readImage(paste0('Data/Pikachu/2. prosess data/jpeg_file/', label_dat[i,1], '.jpeg'))}

Train.img_array <- img_array[,,,1:1000]
Train.box_info <- label_dat[1:1000,3:6]

Test.img_array <- img_array[,,,1001:1088]
Test.box_info <- label_dat[1001:1088,3:6]

# load trained model
# MOBILE_V2_yolo-0000.params is well trained model
# you may test that model you traning by yourself

YOLO_model <- mx.model.load("Model/yolo yu-sheng/MOBILE_V2_yolo", 0)
#YOLO_model <- mx.model.load("Model/yolo yu-sheng/MOBILE_V2_yolo", your model version)

# Predict by yolol model
# you may change training set or testing set if you want

YOLO_pred <- mxnet:::predict.MXFeedForwardModel(model = YOLO_model, X = Train.img_array[,,,1:10])
#YOLO_pred <- mxnet:::predict.MXFeedForwardModel(model = YOLO_model, X = Test.img_array[,,,1:10])

## Decode ##

SAMPLE <- 3

encode_array <- YOLO_pred[,,,SAMPLE]

IMG_array <- Train.img_array[,,,SAMPLE]

# Show predict result
encode_array

# cut is the threshold value of The probability of an object in the yolo grid.
# Note : some sample prediction probaility is low. you may try lower cut value.
cut <- 0.2

box_info <- NULL

for (i in 1:dim(encode_array)[1]) {
  for (j in 1:dim(encode_array)[2]) {
    if (encode_array[i,j,1] > cut) {
      new_box <- data.frame(xleft_1 = (j - 1 + encode_array[i,j,3] - encode_array[i,j,5]/2)/dim(encode_array)[2],
                            xright_1 = (j - 1 + encode_array[i,j,3] + encode_array[i,j,5]/2)/dim(encode_array)[2],
                            ybottom_1 = 1 - (i - 1 + encode_array[i,j,2] + encode_array[i,j,4]/2)/dim(encode_array)[1],
                            ytop_1 = 1 - (i - 1 + encode_array[i,j,2] - encode_array[i,j,4]/2)/dim(encode_array)[1])
      box_info <- rbind(box_info, new_box)
    }
  }
}

# Show predict box 

box_info

## Show plot  ##

par(mar=rep(0,4))
plot(NA, xlim = c(0.04, 0.96), ylim = c(0.04, 0.96), xaxt = "n", yaxt = "n", bty = "n")
img = as.raster(IMG_array)
rasterImage(img, 0, 0, 1, 1, interpolate=FALSE)


for (i in 1:nrow(box_info)) {
  rect(xleft = box_info[i,1], xright = box_info[i,2],
       ybottom = box_info[i,3], ytop = box_info[i,4],
       col = '#00A80050', lwd = 1.5)
}


n.grid <- 7

for (i in 1:n.grid) {
  if (i != n.grid) {
    abline(a = i/n.grid, b = 0, col = 'red', lwd = 1.5)
    abline(v = i/n.grid, col = 'red', lwd = 1.5)
  }
  for (j in 1:n.grid) {
    text((i-0.5)/n.grid, 1-(j-0.5)/n.grid, paste0('(', j, ', ', i, ')'), col = 'red')
  }
}