## Library

library(mxnet)
library(OpenImageR)
library(magrittr)

##0. function

Show_img <- function (img, box_info = NULL, show_grid = TRUE, n.grid = 7) {
  
  par(mar=rep(0,4))
  plot(NA, xlim = c(0.04, 0.96), ylim = c(0.04, 0.96), xaxt = "n", yaxt = "n", bty = "n")
  img = as.raster(img)
  rasterImage(img, 0, 0, 1, 1, interpolate=FALSE)
  
  if (!is.null(box_info)) {
    for (i in 1:nrow(box_info)) {
      rect(xleft = box_info[i,1], xright = box_info[i,2],
           ybottom = box_info[i,3], ytop = box_info[i,4],
           col = '#00A80050', lwd = 1.5)
    }
  }
  
  if (show_grid) {
    for (i in 1:n.grid) {
      if (i != n.grid) {
        abline(a = i/n.grid, b = 0, col = 'red', lwd = 1.5)
        abline(v = i/n.grid, col = 'red', lwd = 1.5)
      }
      for (j in 1:n.grid) {
        text((i-0.5)/n.grid, 1-(j-0.5)/n.grid, paste0('(', j, ', ', i, ')'), col = 'red')
      }
    }
  }
  
}

Encode_fun <- function (box_info, n.grid = 7) {
  
  out_array <- array(0, dim = c(7, 7, 6, nrow(box_info)))
  
  rescale_box_info <- box_info*n.grid
  
  for (i in 1:nrow(box_info)) {
    
    ROW_vec <- 7 - c(rescale_box_info[i,4], rescale_box_info[i,3])
    COL_vec <- c(rescale_box_info[i,1], rescale_box_info[i,2])
    
    out_array[ceiling(mean(ROW_vec)),ceiling(mean(COL_vec)),1,i] <- 1
    out_array[ceiling(mean(ROW_vec)),ceiling(mean(COL_vec)),2,i] <- mean(ROW_vec) %% 1
    out_array[ceiling(mean(ROW_vec)),ceiling(mean(COL_vec)),3,i] <- mean(COL_vec) %% 1
    out_array[ceiling(mean(ROW_vec)),ceiling(mean(COL_vec)),4,i] <- diff(ROW_vec)
    out_array[ceiling(mean(ROW_vec)),ceiling(mean(COL_vec)),5,i] <- diff(COL_vec)
    out_array[ceiling(mean(ROW_vec)),ceiling(mean(COL_vec)),6,i] <- 1
    
  }
  
  return(out_array)
  
}

Decode_fun <- function (encode_array, cut = 0.5) {
  
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
  
  return(box_info)
  
}


##1. load data

label_dat <- read.table('Data/Pikachu/2. prosess data/label_file.txt', header = TRUE)
img_array <- array(0, dim = c(256, 256, 3, nrow(label_dat)))

for (i in 1:nrow(label_dat)) {img_array[,,,i] <- readImage(paste0('Data/Pikachu/2. prosess data/jpeg_file/', label_dat[i,1], '.jpeg'))}

Train.img_array <- img_array[,,,1:1000]
Train.box_info <- label_dat[1:1000,3:6]



#1-1. Test data ( Use frist sample for testing)

par(mar=rep(0,4))
plot(NA, xlim = c(0.04, 0.96), ylim = c(0.04, 0.96), xaxt = "n", yaxt = "n", bty = "n")
img <- as.raster(Train.img_array[,,,1])
rasterImage(img, 0, 0, 1, 1, interpolate=FALSE)

points(Train.box_info$xleft_1[1], Train.box_info$ybottom_1[1])
points(Train.box_info$xright_1[1], Train.box_info$ytop_1[1])

rect( xleft = Train.box_info$xleft_1[1], xright = Train.box_info$xright_1[1],
      ybottom = Train.box_info$ybottom_1[1], ytop = Train.box_info$ytop_1[1], col = '#00A80050', lwd = 1.5)



#1-1. Test YOLO encoding

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

box_info = Train.box_info[1, ]

out_array <- array(0, dim = c(7, 7, 6, nrow(box_info)))

rescale_box_info <- box_info*n.grid

for (i in 1:nrow(box_info)) {
  
  ROW_vec <- 7 - c(rescale_box_info[i,4], rescale_box_info[i,3])
  COL_vec <- c(rescale_box_info[i,1], rescale_box_info[i,2])
  
  out_array[ceiling(mean(ROW_vec)),ceiling(mean(COL_vec)),1,i] <- 1
  out_array[ceiling(mean(ROW_vec)),ceiling(mean(COL_vec)),2,i] <- mean(ROW_vec) %% 1
  out_array[ceiling(mean(ROW_vec)),ceiling(mean(COL_vec)),3,i] <- mean(COL_vec) %% 1
  out_array[ceiling(mean(ROW_vec)),ceiling(mean(COL_vec)),4,i] <- diff(ROW_vec)
  out_array[ceiling(mean(ROW_vec)),ceiling(mean(COL_vec)),5,i] <- diff(COL_vec)
  out_array[ceiling(mean(ROW_vec)),ceiling(mean(COL_vec)),6,i] <- 1
  
}

print(out_array)



##2. Custom iterator

my_iterator_core <- function (batch_size, aug = TRUE) {
  
  batch = 0
  batch_per_epoch = nrow(Train.box_info)/batch_size
  
  reset = function() {batch <<- 0}
  
  iter.next = function() {
    batch <<- batch+1
    if (batch > batch_per_epoch) {return(FALSE)} else {return(TRUE)}
  }
  
  value = function() {
    
    idx = 1:batch_size + (batch - 1) * batch_size
    idx[idx > nrow(Train.box_info)] = sample(1:nrow(Train.box_info), sum(idx > nrow(Train.box_info)))
    
    batch.box_info <- Train.box_info[idx, ]
    
    if (aug) {
      
      random.row <- sample(0:32, 1)
      random.col <- sample(0:32, 1)
      
      data = mx.nd.array(Train.img_array[random.row+1:224,random.col+1:224,,idx])
      batch.box_info[,3:4] <- (1 - (1 - batch.box_info[,3:4])*8/7 + random.row/256)
      batch.box_info[,1:2] <- batch.box_info[,1:2]*8/7 - random.col/256
      
    } else {
      
      data = mx.nd.array(Train.img_array[,,,idx])
      
    }
    
    label = mx.nd.array(Encode_fun(box_info = batch.box_info))
    
    return(list(data = data, label = label))
  }
  
  return(list(reset = reset, iter.next = iter.next, value = value, batch_size = batch_size, batch = batch))
}

my_iterator_func <- setRefClass("Custom_Iter",
                                fields = c("iter", "batch_size", "aug"),
                                contains = "Rcpp_MXArrayDataIter",
                                methods = list(
                                  initialize = function(iter, batch_size = 100, aug = TRUE){
                                    .self$iter <- my_iterator_core(batch_size = batch_size, aug = aug)
                                    .self
                                  },
                                  value = function(){
                                    .self$iter$value()
                                  },
                                  iter.next = function(){
                                    .self$iter$iter.next()
                                  },
                                  reset = function(){
                                    .self$iter$reset()
                                  },
                                  finalize=function(){
                                  }
                                )
)

#2-1. Test Iterator

my_iter <- my_iterator_func(iter = NULL, batch_size = 10, aug = TRUE)

my_iter$reset()
for (i in 1:sample(20, 1)) {my_iter$iter.next()}
test <- my_iter$value()

Show_img(img = as.array(test$data)[,,,1], box_info = Decode_fun(as.array(test$label)[,,,1]))
print(i)
print(as.array(test$label)[,,,1])


##3. Define the model architecture
##   Use pre-trained model and fine tuning

#3-1. Mobile Net V2

Pre_Trained_model <- mx.model.load('Model/Pre-trained model/mobilev2', 0)

Mobile_V2_symbol <- Pre_Trained_model$symbol

Mobile_V2_All_layer <- Mobile_V2_symbol$get.internals()

Mobile_V2_Last_ConV <- which(Mobile_V2_All_layer$outputs == "conv6_3_linear_output" ) %>% Mobile_V2_All_layer$get.output(.)


#3-2. Convolution layer for speific mission and training new para

YOLO_ConV_1 <- mx.symbol.Convolution(data = Mobile_V2_Last_ConV, kernel = c(3, 3), stride = c(1, 1), pad = c(1, 1),
                                     no.bias = TRUE, num.filter = 320, name = "YOLO_ConV_1")

YOLO_BN_1 <- mx.symbol.BatchNorm(data = YOLO_ConV_1, eps = 1e-4, fix_gamma = FALSE,
                                 momentum = 0.9, use_global_stats = FALSE,
                                 name = 'YOLO_BN_1')

YOLO_Relu_1 <- mx.symbol.Activation(data = YOLO_BN_1, act.type = "relu", name = "YOLO_Relu_1")

YOLO_ConV_2 <- mx.symbol.Convolution(data = YOLO_Relu_1, kernel = c(3, 3), stride = c(1, 1), pad = c(1, 1),
                                     no.bias = TRUE, num.filter = 1280, name = "YOLO_ConV_2")

YOLO_BN_2 <- mx.symbol.BatchNorm(data = YOLO_ConV_2, eps = 1e-4, fix_gamma = FALSE,
                                 momentum = 0.9, use_global_stats = FALSE,
                                 name = 'YOLO_BN_2')

YOLO_Relu_2 <- mx.symbol.Activation(data = YOLO_BN_2, act.type = "relu", name = "YOLO_Relu_2")

YOLO_ConV_fc_1 <- mx.symbol.Convolution(data = YOLO_Relu_2, kernel = c(1, 1), stride = c(1, 1), pad = c(0, 0),
                                        num.filter = 6, name = "YOLO_ConV_fc_1")

#3-3. Custom Loss Active function

YOLO_inter_split <- mx.symbol.SliceChannel(data = YOLO_ConV_fc_1, num_outputs = 6, 
                                           axis = 1, squeeze_axis = FALSE, name = "YOLO_inter_split")

new_list <- list()
new_list[[1]] <- YOLO_inter_split[[1]]
new_list[[2]] <- YOLO_inter_split[[2]]
new_list[[3]] <- YOLO_inter_split[[3]]
new_list[[4]] <- mx.symbol.Activation(YOLO_inter_split[[4]], act.type = 'relu', name = "yolo_map_4")
new_list[[5]] <- mx.symbol.Activation(YOLO_inter_split[[5]], act.type = 'relu', name = "yolo_map_5")
new_list[[6]] <- YOLO_inter_split[[6]] #mx.symbol.softmax(data = YOLO_inter_split[[6]], axis = 1, name = 'yolo_map_6')

YOLO_MAP <- mx.symbol.concat(data = new_list, num.args = 6, dim = 1, name = 'yolo_map')

#3-4. YOLO multi-part loss function

label <- mx.symbol.Variable(name = "label")

YOLO_split <- mx.symbol.SliceChannel(data = YOLO_MAP, num_outputs = 6, 
                                     axis = 1, squeeze_axis = FALSE, name = "YOLO_split")

label_split <- mx.symbol.SliceChannel(data = label, num_outputs = 6, 
                                      axis = 1, squeeze_axis = FALSE, name = "label_split")

# Predict_IoU
Predict_IoU_obj <- mx.symbol.square(data = (label_split[[1]] * (YOLO_split[[1]] - label_split[[1]])), name = "Predict_IoU_obj")
Predict_IoU_noobj <- mx.symbol.square(data = ((1 - label_split[[1]]) * (YOLO_split[[1]] - label_split[[1]])), name = "Predict_IoU_noobj")

Predict_IoU_obj_sum <- mx.symbol.sum_axis(data = Predict_IoU_obj, axis = c(3, 2), keepdims = T, name = 'Predict_IoU_obj_sum')
Predict_IoU_noobj_sum <- mx.symbol.sum_axis(data = Predict_IoU_noobj, axis = c(3, 2), keepdims = T, name = 'Predict_IoU_noobj_sum')

# Anchors box parameter
Predict_x_square <- mx.symbol.square(data = (YOLO_split[[2]] - label_split[[2]]), name = "Predict_x_square")
Predict_y_square <- mx.symbol.square(data = (YOLO_split[[3]] - label_split[[3]]), name = "Predict_y_square")

Predict_xy_sum <- mx.symbol.sum_axis(data = (Predict_x_square + Predict_y_square) * label_split[[1]], axis = c(3, 2), keepdims = T, name = 'Predict_xy_sum')

Predict_w_square <- mx.symbol.square(data = (mx.symbol.sqrt(YOLO_split[[4]]) - mx.symbol.sqrt(label_split[[4]])), name = "Predict_w_square")
Predict_h_square <- mx.symbol.square(data = (mx.symbol.sqrt(YOLO_split[[5]]) - mx.symbol.sqrt(label_split[[5]])), name = "Predict_h_square")

Predict_wh_sum <- mx.symbol.sum_axis(data = (Predict_w_square + Predict_h_square) * label_split[[1]], axis = c(3, 2), keepdims = T, name = 'Predict_wh_sum')

# Preduct_class
Preduct_class <- mx.symbol.square(data = (YOLO_split[[6]] - label_split[[6]]), name = "Preduct_class")
Preduct_class_sum <- mx.symbol.sum_axis(data = Preduct_class * label_split[[1]], axis = c(3, 2), keepdims = T, name = 'Preduct_class_sum')

coord_lamda <- 5; noobj_lamda <- .5

individual_YOLO <- (coord_lamda * Predict_xy_sum + 
                    coord_lamda * Predict_wh_sum + 
                    Predict_IoU_obj_sum +
                    noobj_lamda * Predict_IoU_noobj_sum +
                    Preduct_class_sum )

average_YOLO <- mx.symbol.mean(data = individual_YOLO, axis = c(3, 2, 1, 0), keepdims = FALSE,
                               name = 'average_YOLO')

FINAL_YOLO <- mx.symbol.MakeLoss(data = average_YOLO, name = 'FINAL_YOLO')


##4. initiate Parameter for model
new_arg <- mxnet:::mx.model.init.params(symbol = FINAL_YOLO, 
                                        input.shape = list(data = c(224, 224, 3, 10), 
                                                           label = c(7, 7, 6, 10)), 
                                        output.shape = NULL, initializer = mxnet:::mx.init.Xavier(rnd_type = "uniform", magnitude = 2.24), 
                                        ctx = CTX)
#4-1. Bind Pre-trained Parameter into model

Pre_trained_ARG <- Pre_Trained_model$arg.params
Pre_trained_AUX <- Pre_Trained_model$aux.params

ARG_in_net_name <- names(Pre_trained_ARG) %>% .[. %in% names(new_arg$arg.params)]  # remove paramter does not in model
AUX_in_net_name <- names(Pre_trained_AUX) %>% .[. %in% names(new_arg$aux.params)]  # remove paramter does not in model

for (i in 1:length(ARG_in_net_name)){
  new_arg$arg.params[names(new_arg$arg.params) == ARG_in_net_name[i]] <- Pre_trained_ARG[names(Pre_trained_ARG) == ARG_in_net_name[i]]
}

for (i in 1:length(AUX_in_net_name)){
  new_arg$arg.params[names(new_arg$arg.params) == AUX_in_net_name[i]] <- Pre_trained_AUX[names(Pre_trained_AUX) == AUX_in_net_name[i]]
}

ARG.PARAMS <- new_arg$arg.params
AUX.PARAMS <- new_arg$aux.params

#4-2. Define layer to fixed in Mobile V2

Layer_to_fixed <- ARG_in_net_name[!(grepl("conv6_", ARG_in_net_name, fixed = T) #| 
                                 # grepl("conv5_", ARG_in_net_name, fixed = T) |
                                 # grepl("conv4_", ARG_in_net_name, fixed = T) |
                                 # grepl("conv3_", ARG_in_net_name, fixed = T)
)]

# Note :
# to see output shape in MObile V2
# mx.symbol.infer.shape(Mobile_V2_All_layer, data = c(224, 224, 3, 1))$out.shapes


##5. Custom callback function
#    Custom eval metric loss due to custom loss function
my.eval.metric.loss <- mx.metric.custom(
  name = "multi_part_loss",
  function(label, pred) {
    return(as.array(pred))
  }
)

#   Custom eval metric loss due to fix layer
my.callback_epoch <- function (prefix = 'Model/yolo/yolo_v1',
                               out_symbol_name = 'yolo_map_output',
                               fixed.params = NULL,
                               period = 1) {
  function(iteration, nbatch, env, verbose = TRUE) {
    if (iteration%%period == 0) {
      env_model <- env$model
      env_all_layers <- env_model$symbol$get.internals()
      final_symbol <- which(env_all_layers$outputs == out_symbol_name) %>% env_all_layers$get.output()
      model_write_out <- list(symbol = final_symbol,
                        arg.params = env_model$arg.params,
                        aux.params = env_model$aux.params)
      model_write_out[[2]] <- append(model_write_out[[2]], fixed.params)
      class(model_write_out) <- "MXFeedForwardModel"
      mx.model.save(model_write_out, prefix, iteration)
      if (verbose) {
        message(sprintf("Model checkpoint saved to %s-%04d.params", prefix, iteration))
      }
    }
    return(TRUE)
  }
}



#6. Model Training

n.cpu = 4
device.cpu <- lapply(0:(n.cpu-1), function(i) {mx.cpu(i)})
CTX = device.cpu # mx.gpu(0)

YOLO_model <- mx.model.FeedForward.create(FINAL_YOLO,
                                          X = my_iter, 
                                          ctx = CTX, 
                                          begin.round = 1, num.round = 300,
                                          array.batch.size = 10, 
                                          learning.rate = 0.0005, momentum = 0.9, wd = 1e-5,
                                          fixed.param = Layer_to_fixed,
                                          arg.params = ARG.PARAMS, aux.params = AUX.PARAMS,
                                          eval.metric = my.eval.metric.loss,
                                          batch.end.callback = mx.callback.log.speedometer(batch_size = 10, frequency = 10),
                                          epoch.end.callback = my.callback_epoch(prefix = "Model/Yolo/MOBILE_V2_yolo",
                                                                                 out_symbol_name = YOLO_MAP$outputs,
                                                                                 fixed.params = ARG.PARAMS[names(ARG.PARAMS) %in% Layer_to_fixed],
                                                                                 period = 1))





