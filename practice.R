require(mlr3verse)
require(data.table)


# import data -------------------------------------------------------------

getwd()
data <- fread('ml/df_comp.csv')



# create task -------------------------------------------------------------
require(mlr3verse)
data[,afib_within_7days := as.factor(afib_within_7days)]
str(data)
target <- as.data.table(task$feature_types)[type=='character',id]
data[,(target):=lapply(.SD, function(x) as.factor(ifelse(x=='Yes',1,
                                                         ifelse(x=='No',0,x)))), .SDcols=target]

data[,`:=`(
  sex = as.factor(ifelse(sex=='Female',0,1)),
  race = as.factor(ifelse(race=='African American',0,1)),
  age60_ = as.factor(ifelse(age60_=='<60',0,1)),
  cvicu = as.factor(ifelse(icuunit=='CVICU',1,0)),
  icuunit = NULL,
  ethnicity = NULL,
  obs_duration = NULL,
  stay_id = NULL,
  t2e = NULL,
  inhos_duration = NULL,
  inhos_mortality = NULL,
  ventil_after_afib_duration_day = NULL,
  ventil_before_afib_duration_day = NULL
)]
str(data)
task <- mlr3::TaskClassif$new(id='mimic',
                    backend=data,
                    target='afib_within_7days') # task: need to be factor



# preprocessing -----------------------------------------------------------

require(mlr3pipelines)
require(mlr3learners)
require(iml)
# character or factor ==> numeric or one-hot encoding
# method = 'one-hot' : mutate a new column for each levels
# method = 'treatment: create n-1 columns leaving out the first factor of each factor variables
fac_to_int <- po('encode', method='treatment', affect_columns=selector_type('factor'))

graph <- fac_to_int %>>%
  po('learner', learner = lrn('classif.xgboost', predict_type='prob', 
                              predict_sets=c('train','test')))
glrn <- as_learner(graph)


train_set = sample(task$nrow, 0.8*task$nrow)
test_set = setdiff(seq_len(task$nrow), train_set)


glrn$train(task, row_ids = train_set)
pred = glrn$predict(task, row_ids=test_set)
pred$confusion
measures = msrs(c('classif.auc',
                  'classif.prauc',
                  'classif.recall',
                  'classif.acc',
                  'classif.fbeta'))
pred$score(measures)
dictionary_sugar(msrs())
autoplot(pred, type='roc')


# resampling
rr <- rsmp('cv',folds=10)
rr$instantiate(task)

resampling <- resample(task, glrn, rr, store_models = T)
resampling$score(measures)
resampling$aggregate(msr('classif.auc'))
autoplot(resampling, measure = msr('classif.fbeta'))
autoplot(resampling, type='roc') + theme(legend.position = 'bottom')


# benchmarking ------------------------------------------------------------

design <- benchmark_grid(
  
)

# tuning -----------------------------------
require(mlr3tuning)
dictionary_sugar(trm())

# Checking hyperparameter
# learner$param_set
glrn$param_set %>% as.data.table() %>% View()

search_space = ps(
  classif.xgboost.eta = p_dbl(lower = 0.001, upper = 1),
  classif.xgboost.booster = p_fct(c('gbtree','gblinear','dart')),
  nroudns = p_int(lower=1, upper=16, tags='budget')
)
search_space

# 1. creating instance

measures <- msrs(c('classif.ce','time_train'))

instance = TuningInstanceMultiCrit$new(
  task = task,
  learner = glrn,
  resampling = rsmp('holdout'),
  search_space = search_space,
  measure = measures,
  terminator = trm('evals', n_evals=20)
)
glrn$param_set %>% as.data.table() %>% View()
instance

# 2. creating tuner
mlr3tuning::tnrs('grid_search')
tuner <- mlr3tuning::tnr('grid_search')

tuner$optimize(instance)
instance$result_y
instance$archive
instance$archive$benchmark_result$score(msr('classif.fbeta'))

glrn$param_set$values <- instance$result_learner_param_vals[[1]]# best result
instance$result_learner_param_vals

# automate tuning

at = AutoTuner$new(
  learner = glrn,
  resampling = rsmp('cv'),
  measure = msr('classif.ce'),
  search_space = search_space,
  terminator = trm('evals',n_evals=20),
  tuner = mlr3tuning::tnr('random_search')
)

at$train(task, row_ids=train_set)

at$archive
at$tuning_result

pred = at$predict(task, row_ids=test_set)
pred$confusion
pred$score(msrs(c('classif.auc','classif.fbeta')))
autoplot(pred, type='prc') + autoplot(pred, type='roc')


# Nested Resampling

outer_resampling = rsmp('holdout')

rr = resample(task = task, learner=at,
              resampling = outer_resampling,
              store_models = T)

extract_inner_tuning_results(rr)
rr$score()
rr$aggregate()

# feature selection
#   1) filtering:  assign an importance value to each feature
# require(praznik)
dictionary_sugar(flt())
filter = flt('importance') 

filter$calculate(task)
as.data.table(filter)


#   2) wrapper

instance = FSelectInstanceSingleCrit$new(
  task = task,
  learner = glrn,
  resampling = rsmp('holdout'),
  measure = measure,
  terminator = trm('evals', n_evals=20)
)

instance

fselector = fs('random_search')

lgr::get_logger("bbotk")$set_threshold("warn") # reducing logging output
fselector$optimize(instance)

instance$result_feature_set
instance$result_y


# automating feature selection

at = AutoFSelector$new(
  learner = glrn,
  resampling = rsmp('holdout'),
  measure = msr('classif.ce'),
  terminator = trm('evals',n_evals=20),
  fselector = fselector
)



# compare with whole model
grid = benchmark_grid(
  task = task,
  learner = list(at,glrn),
  resampling = rsmp('cv', folds=5)
)

bmr = benchmark(grid, store_models = T)
bmr$aggregate(msrs(c("classif.ce", "time_train")))

#   3) embedded



# Interpretation ----------------------------------------------------------

model <- Predictor$new(glrn, data=data[train_set, ], y='afib_within_7days')
effect <- FeatureImp$new(model, loss='ce')
effect
plot_train = plot(effect, features = num_features)
plot_train

model <- Predictor$new(glrn, data=data[test_set, ],y='afib_within_7days')
effect <- FeatureImp$new(model, loss='ce')
plot_test = plot(effect, features = num_features)

require(patchwork)
plot_train + plot_test


predict <- glrn$predict(task, row_ids=test_set)
predict$score(measures = msrs(c('classif.auc','classif.acc',
                                'classif.recall','classif.prauc')))



x <- data %>% select(-afib_within_7days)


