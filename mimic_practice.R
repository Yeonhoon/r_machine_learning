

# import packages ---------------------------------------------------------

require(mlr3verse)


# EDA ---------------------------------------------------------------------
names(df_comp)
str(df_comp)
df_ml <- df_comp[,stay_id:=NULL]
df_ml_raw <- copy(temp)
df_ml_raw
df_ml[,afib_within_7days:=as.factor(afib_within_7days)]
df_ml_raw[,afib_within_7days:=as.factor(afib_within_7days)]
task <- as_task_classif(df_ml , target = 'afib_within_7days', positive='1')
task_raw <- as_task_classif(df_ml_raw , target = 'afib_within_7days', positive='1')

# exclude features
task$select(setdiff(task$feature_names,c('t2e','obs_duration','inhos_duration','ethnicity',
                                         'mortality_90','inhos_mortality','icuunit')))
task_raw$select(setdiff(task_raw$feature_names,c('t2e','obs_duration','inhos_duration','ethnicity',
                                                 'mortality_90','inhos_mortality','icuunit')))
task$feature_names
task_raw$feature_names

# preprocessing -----------------------------------------------------------

require(mlr3pipelines)
task$feature_types
task_raw$
  dictionary_sugar(po())

## pipe Ops
# encode: factor, ordered, character features to int
## options: one-hot, treatment ...
# scale: normalization for numeric features\
# scalerange: min-max scaler
# smote: balancing 
# impute(median, mode, mean) : imputation by median


fct_to_int <- po("encode", method = "one-hot", affect_columns = selector_type("factor"))
scale <- po('scale')

xgb <- fct_to_int %>>% po('scale') %>>% 
  po('learner', learner= lrn('classif.xgboost', predict_type='prob',
                             predict_sets=c('train','test')))
require(mlr3learners.lightgbm)
lgbm <- fct_to_int %>>%
  po('learner', learner= lrn('classif.lightgbm', predict_type='prob',
                             predict_sets=c('train','test')))

rf <- fct_to_int %>>% po('learner', learner= lrn('classif.ranger', predict_type='prob',
                                                 predict_sets=c('train','test')))


xgb_learner <- as_learner(xgb)
lgbm_learner <- as_learner(lgbm)
rf_learner <- as_learner(rf)

# xgb_learner <- as_learner(xgb)

# xgb_learner$train(task, row_ids = train_set)
rf

grid <- benchmark_grid(
  tasks = task,
  learners = c(xgb_learner,lgbm_learner, rf_learner),
  resamplings = rr
)


bmr <- benchmark(grid, store_models = T)
bmr$score(measures) %>% as.data.table() %>% View()
bmr$aggregate(measures)
autoplot(bmr, type='roc')+theme(legend.position = 'bottom')

# trainining --------------------------------------------------------------

train_set <- sample(task$nrow, 0.8*task$nrow)
test_set <- setdiff(seq_len(task$nrow), y=train_set)
learner <- lrn('classif.ranger', predict_type='prob')
print(learner)


# prediction --------------------------------------------------------------
pred <- learner$train(task,row_ids=train_set)$predict(task, row_ids=test_set)

pred$set_threshold(0.1)
measures <- msrs(c('classif.acc','classif.auc','classif.prauc','classif.fbeta'))
pred$score(measures)
pred$confusion
autoplot(pred, type='prc')
roc_curve <- autoplot(predict, type='roc');roc_curve
prc_curve <- autoplot(predict, type='prc');prc_curve

# resampling ---------------------------------------------------------------

rr <- rsmp('cv',folds=10)
rr$instantiate(task)

resampling <- resample(task, learner, rr, store_model=T)
resampling$score(measures)

resampling$aggregate(msr('classif.prauc'))

resampling$prediction(predict_sets = 'test')


# Parameter tuning --------------------------------------------------------

require(mlr3tuning)
dictionary_sugar(trm())

as.data.table(learner$param_set) %>% View()
search_space = ps(
  max.depth = p_int(10,100),
  num.trees = p_int(1,10),
  min.node.size = p_int(10,100)
)

instance <- TuningInstanceMultiCrit$new(
  task = task,
  learner = learner,
  resampling = rr,
  search_space = search_space,
  measure = measures,
  terminator = trm('evals',n_evals=20)
)

# set tuner
at <- AutoTuner$new(
  # task = task,
  learner = learner,
  resampling = rsmp('cv',folds=5),
  search_space = search_space,
  measure = msr('classif.ce'),
  terminator = trm('evals',n_evals=20),
  tuner = tnr('random_search')
)


at$train(task, row_ids = train_set)
at$archive
at$tuning_result

pred <- at$predict(task, row_ids=test_set)
pred$score(measures)
autoplot(pred, type='roc')
pred$confusion


# compare with whole model
grid <- benchmark_grid(
  task = task,
  learner = list(at, learner),
  resampling = rr
)

bmr <- benchmark(grid,store_models=T)
bmr$aggregate(msrs(c('classif.ce','time_train')))

dictionary_sugar(po())
??collapsefactors



# importance filter -------------------------------------------------------


