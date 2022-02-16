
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
# data('Sonar',package = "mlbench")
task <- as_task_classif(Sonar,target='Class',positive='M')
leaner <- lrn('classif.rpart', predict_type='prob')
pred <- learner$train(task)$predict(task)
pred$set_threshold(0.1)
CM <- pred$confusion
CM

roc_curve <- autoplot(pred, type='roc')
prc_curve <- autoplot(pred, type='prc')

roc_curve


# Resampling: CV ------------------------------------------------------------

cv5=rsmp('cv', folds=5)

rr = resample(task, learner, cv5)
rr$score(measure)

rr_table = as.data.table(rr)

# resample ==> aggregate
rr$aggregate(measure)
