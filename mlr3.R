require(tidyverse)
install.packages('tidymodels')

require(tidymodels)

install.packages('caret')

require(mlr3)
require(data.table)

# Object creation: <Class>$new()
penguins <- TaskClassif$new(id='penguins',
                            backend=palmerpenguins::penguins,
                            target='species')
task <- tsk('iris')

penguins$ncol
penguins$feature_names
penguins$target_names
penguins$truth(rows = 200)
penguins$data(rows = 200, cols='sex')
penguins$select(cols='island') # 특정 feature 선택
penguins$filter(rows=1:10)


# Dictionaries -----------------------------------------------------------

# Task: mlr_tasks => tsk()
# Learner mlr_learners => lrn()
# Measure: mlr_measures => msr()
# Resampling mlr_resamplings => rsml()


# learner: training model
learner <-  lrn("classif.rpart", cp=.01)
as.data.table(learner$param_set)


# training ---------------------------------------------------------------
train_set <- sample(x = task$nrow, size = 0.8 * task$nrow)
test_set <- setdiff(seq_len(task$nrow), train_set)


# hyperparameter 변경
learner$param_set$values = list(maxdepth=1, xval=0)

learner$train(task, row_ids = train_set)

#prediction --------------------------------------------

prediction <- learner$predict(task, row_ids = test_set)

prediction$confusion
prediction$score()

measure <- msr('classif.acc') # accuracy
prediction$score(measure)

# Resampling: CV ------------------------------------------------------------

cv5=rsmp('cv', folds=5)

rr = resample(task, learner, cv5)
rr$score(measure)

rr_table = as.data.table(rr)

# resample ==> aggregate
rr$aggregate(measure)
