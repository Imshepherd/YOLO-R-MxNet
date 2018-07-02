#####################!!!# Due to files size #!!!######################
#  Please start code at 'Code/1. Data pipline/1. load jpg_text data.R'
######################################################################

#Get data from url: 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/'
# Down load thoes filex into "Data/Pikachu/1. raw data/" #
#Train imgrec sets : 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/train.rec' 
#Train imgidx sets : 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/train.idx' 
#Valid imgrec sets : 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/val.rec'


#library
library(mxnet)
library(abind)

#Set path
input.path <- "Data/Pikachu/1. raw data/"

#parameter
data_shape <- c(256, 256, 3)
batch_size <- 32

# Load data
# data is in iterator. get data out and transform into jpeg
#mx.io.ImageDetRecordIter dont have parameter to read .idx file and max_attempts=200
train_iter <- mx.io.ImageDetRecordIter(path.imgrec = paste0(input.path, "train.rec"),
                                       batch.size = batch_size,
                                       data.shape = data_shape,
                                       shuffle = TRUE,
                                       mean = TRUE,
                                       rand.crop = 1,
                                       min.crop.object.coverages = 0.95) 

val_iter <- mx.io.ImageDetRecordIter(path.imgrec = paste0(input.path, "val.rec"),
                                     batch.size = batch_size,
                                     data.shape = data_shape,
                                     shuffle = FALSE,
                                     mean = TRUE)

#####################################################################
#Training data into R files.

train_iter$iter.next()
train_iter$reset()

Train_list = list()

Train_list[[length(Train_list)+1]] <- train_iter$value()

for (i in 1:10000){
  while ( train_iter$iter.next() ){
    Train_list[[length(Train_list)+1]] <- train_iter$value()
    if (sum(as.array(Train_list[[length(Train_list)]]$label)[9,]) > 0){
      stop('find none pikachu in data.')
    }
  }
}

#Get data
Pikachu_train_data <- array(NA, dim = c(256, 256, 3, 32 * length(Train_list)))

for (i in 1:length(Train_list)){
  Pikachu_train_data[,,,(1+32*(i-1)):(32*i)] <- as.array(Train_list[[i]]$data)
}

#Rotate data
for (i in 1:dim(Pikachu_train_data)[4]){
  Pikachu_train_data[,,1,i] <- t(Pikachu_train_data[,,1,i])
  Pikachu_train_data[,,2,i] <- t(Pikachu_train_data[,,2,i])
  Pikachu_train_data[,,3,i] <- t(Pikachu_train_data[,,3,i])
}

#Get Label
Pikachu_train_label_old <- array(NA, dim = c(4, 32 * length(Train_list)))

for (i in 1:length(Train_list)){
  Pikachu_train_label_old[,(1+32*(i-1)):(32*i)] <- as.array(Train_list[[i]]$label)[10:13,]
}  

#Rotate label

Pikachu_train_label_old[2, ] = 1 - Pikachu_train_label_old[2, ]
Pikachu_train_label_old[4, ] = 1 - Pikachu_train_label_old[4, ]

#Make order of label into xleft, xright, ybottom, ytop 

Pikachu_train_label <- array(NA, dim = c(4, 32 * length(Train_list)))

Pikachu_train_label[1, ] <- Pikachu_train_label_old[1, ]
Pikachu_train_label[2, ] <- Pikachu_train_label_old[3, ]
Pikachu_train_label[3, ] <- Pikachu_train_label_old[4, ]
Pikachu_train_label[4, ] <- Pikachu_train_label_old[2, ]

#####################################################################
#Visualization

par(mar = c(0, 0, 0, 0))
plot(NA, xlim = 0:1, ylim = 0:1,xaxs="i", yaxs="i", xaxt = "n", yaxt = "n", bty = "n")
rasterImage( Pikachu_train_data[,,,1]/255, 0, 0, 1, 1, interpolate = FALSE)

rect( xleft = Pikachu_train_label[1, 1], xright = Pikachu_train_label[2, 1],
      ybottom = Pikachu_train_label[3, 1], ytop = Pikachu_train_label[4, 1], col = '#00A80050', lwd = 1.5)


##################################################
########## Same as Validation data below #########
##################################################
#Vaildation data into R files.

val_iter$iter.next()
val_iter$reset()

val_list = list()

val_list[[length(val_list)+1]] <- val_iter$value()

for (i in 1:10000){
  while (val_iter$iter.next()){
    val_list[[length(val_list)+1]] <- val_iter$value()
    if (sum(as.array(val_list[[length(val_list)]]$label)[9,]) > 0){
      stop('find none pikachu in data.')
    }
  }
}

#Get data
Pikachu_val_data <- array(NA, dim = c(256, 256, 3, 32 * length(val_list)))

for (i in 1:length(val_list)){
  Pikachu_val_data[,,,(1+32*(i-1)):(32*i)] <- as.array(val_list[[i]]$data)
}

#Rotate data
for (i in 1:dim(Pikachu_val_data)[4]){
  Pikachu_val_data[,,1,i] <- t(Pikachu_val_data[,,1,i])
  Pikachu_val_data[,,2,i] <- t(Pikachu_val_data[,,2,i])
  Pikachu_val_data[,,3,i] <- t(Pikachu_val_data[,,3,i])
}

#Get Label
Pikachu_val_label_old <- array(NA, dim = c(4, 32 * length(val_list)))

for (i in 1:length(val_list)){
  Pikachu_val_label_old[,(1+32*(i-1)):(32*i)] <- as.array(val_list[[i]]$label)[10:13,]
}  

#Rotate label

Pikachu_val_label_old[2, ] = 1 - Pikachu_val_label_old[2, ]
Pikachu_val_label_old[4, ] = 1 - Pikachu_val_label_old[4, ]

#Make order of label into xleft, xright, ybottom, ytop 

Pikachu_val_label <- array(NA, dim = c(4, 32 * length(val_list)))

Pikachu_val_label[1, ] <- Pikachu_val_label_old[1, ]
Pikachu_val_label[2, ] <- Pikachu_val_label_old[3, ]
Pikachu_val_label[3, ] <- Pikachu_val_label_old[4, ]
Pikachu_val_label[4, ] <- Pikachu_val_label_old[2, ]

#####################################################################
#Save out data
#Bind train and validation

Pikach_data <- abind(Pikachu_train_data, Pikachu_val_data)
Pikach_label <- cbind(Pikachu_train_label, Pikachu_val_label)

save(Pikach_data, Pikach_label, file = paste0(input.path, "Pikach_data.RData"))

###########################################
########## Write as jpeg file #############
###########################################

#Set_path
input.path2 <- "Data/Pikachu/1. raw data/"
output.path2 <- "Data/Pikachu/2. prosess data/"

#load data
load(paste0(input.path2, "Pikach_data.RData"))

#Save out jpeg
for (i in 1:dim(Pikach_data)[4]){
  writeJPEG(Pikach_data[,,,i]/255, target = paste0(output.path2, "jpeg_file/", i,".jpeg"), quality =1, bg = "white")
}

#Write text file for label
Pikach_label <- data.frame(JPEG_File_ID = 1:dim(Pikach_data)[4],
                           Class_1 = rep(1, dim(Pikach_data)[4]),
                           xleft_1 = Pikach_label[1, ],
                           xright_1 = Pikach_label[2, ],
                           ybottom_1 = Pikach_label[3, ],
                           ytop_1 = Pikach_label[4, ])

write.table(Pikach_label, file = paste0(output.path2, "label_file.txt"), sep = "\t", row.names = F)


