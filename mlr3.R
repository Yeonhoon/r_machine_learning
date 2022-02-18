
require(mlr3verse)
require(data.table)

# Object creation: <Class>$new() or mlr3::tsk()
penguins <- TaskClassif$new(id='penguins',
                            backend=palmerpenguins::penguins,
                            target='species')
task <- tsk('iris')
task2 <- TaskClassif$new(id='iris',
                         backend = iris,
                         target='Species')



# pre-defined datasets in mlr3

mlr_tasks
as.data.table(mlr_tasks) #pxc: POSIXct


# task object로 할 수 있는 기능들
task$ncol
task$feature_names # feature 이름
task$class_names # target의 class 이름
task$target_names
task$truth(rows = 200)
task$row_ids
task$data(rows = 150, cols='Species')
task$select(cols='island') # 특정 feature 선택
task$filter(rows=1:10) # rows filter
task$task_type
task$col_roles


# Dictionaries -----------------------------------------------------------

# Task: mlr_tasks => tsk()
# Learner mlr_learners => lrn()
# Measure: mlr_measures => msr()
# Resampling mlr_resamplings => rsml()


require(mlr3)
mlr_learners
?mlr_measures
mlr3misc::dictionary_sugar(tsk())

# EDA ---------------------------------------------------------------------

summary(as.data.table(task))
mlr3viz::autoplot(task)
mlr3viz::autoplot(task, type='pairs')
mlr3viz::autoplot(task, type='duo')

# learner for training ---------------------------------------------------------------

mlr3misc::dictionary_sugar(lrn()) 
mlr_learners
#classif.featureless
lnr('')

data('Sonar', package = 'mlbench')
task <- as_task_classif(Sonar, target='Class', positive="M")
learner <-  lrn("classif.rpart", cp=.01, predict_type='prob')

# learner$param_set$values$cp <- 0.02
# as.data.table(learner$param_set)
# hyperparameter 변경
learner$param_set$values = list(maxdepth=1, xval=0)

learner$train(task, row_ids = train_set)




# Thresholding ----------------------------------------------------------------

pred <- learner$train(task)$predict(task)
mlr3misc::dictionary_sugar_get(msr())
measures <- msrs(c('classif.tpr','classif.tnr','classif.acc'))
pred$confusion
pred$score(measures)

pred$set_threshold(0.2)
pred$confusion
pred$score(measures)


# Train, Prediction, and Assessment ---------------------------------------

task <- tsk('penguins')
learner <- lrn('classif.rpart')

# train_test_split

train_set <- sample(task$nrow, 0.8*task$nrow)
test_set <- setdiff(seq_len(task$nrow), train_set)

# train the learner

learner$model
learner$train(task, row_ids=train_set)

# prediction 

prediction <- learner$predict(task, row_ids = test_set)
prediction

prediction$confusion

# changing the predict type: acc to prob

learner$predict_type <- 'prob'
prediction <- learner$predict(task, row_ids = test_set)
prediction$prob
prediction$response

autoplot(prediction)
measure <- msr('classif.acc') # accuracy
prediction$score(measure)


# ROC ---------------------------------------------------------------------
data('Sonar',package = "mlbench")
task <- as_task_classif(Sonar,target='Class',positive='M')
learner <- lrn('classif.rpart', predict_type='prob')
pred <- learner$train(task)$predict(task)
pred$set_threshold(0.1)
CM <- pred$confusion
CM

roc_curve <- autoplot(pred, type='roc')
prc_curve <- autoplot(pred, type='prc')

roc_curve


# Resampling: CV ------------------------------------------------------------
# 종류
# cv, loo(leave one out), repeated_cv, 
# bootstrapping, subsampling, holdout, insampling, custom

as.data.table(mlr_resamplings)

resampling <- rsmp('cv',folds=5)
# resampling$param_set$values <- list(ratio=.8)
print(resampling)

resampling$instantiate(task)
str(resampling$train_set(1))
str(resampling$test_set(1))

rr = resample(task, learner, resampling, store_models = T)
print(rr)
rr$score(measure)

# cv한 이후의 measures
rr$aggregate(msr('classif.ce'))

# visualization
autoplot(rr, measure = msr('classif.auc'))
autoplot(rr, type='roc')
rr$filter(1)
autoplot(rr, type = "prediction")


# benchmarking ------------------------------------------------------------

# Comparing the performance of different learners

design <- benchmark_grid(
  task = tsks(c('spam','german_credit','sonar')),
  learners = lrns(c('classif.ranger','classif.rpart','classif.featureless'),
                  predict_type = 'prob', predict_sets= c('train','test')),
  resamplings = rsmps('cv',folds=5)
)
print(design)

bmr <- benchmark(design)

measures <- list(
  msr('classif.auc',predict_sets='train', id='auc_train'),
  msr('classif.auc',id='auc_test')
)
tab <- bmr$aggregate(measures)

rank <- tab[,.(learner_id, rank_train = rank(-auc_train),
               rank_test = rank(-auc_test)),by=task_id]
ranks <- rank[,.(mrank_train = mean(rank_train), mrank_test = mean(rank_test)),by=learner_id]

ranks[order(mrank_test)]

# Plotting
require(ggplot2)
autoplot(bmr) + ggplot2::theme(axis.text.x = ggplot2::element_text(angle=45, hjust=1))
bmr_german <- bmr$clone()$filter(task_id = 'german_credit')
bmr_sonar <- bmr$clone()$filter(task_id = 'sonar')
autoplot(bmr_german, type='roc') + theme(legend.position = 'bottom')
autoplot(bmr_sonar, type='roc') + theme(legend.position = 'bottom')

# extracting resample results

tab <- bmr$aggregate(measures)
rr <- tab[learner_id =='classif.ranger']$resample_result[[1]]

measure <- msr('classif.auc')
rr$aggregate(measure)


# converting and merging --------------------------------------------------

# resampling 모델들을 benchmark result로 converting 한 뒤, 
# combine()을 활용하여 merge 가능

task <- tsk('iris')
resampling <- rsmp('cv')$instantiate(task)

rr1 <- resample(task,lrn('classif.rpart'),resampling)
rr2 <- resample(task,lrn('classif.featureless'),resampling)

bmr1 <- as_benchmark_result(rr1)
bmr2 <- as_benchmark_result(rr2)

bmr1
bmr1$combine(bmr2)