[![Build Status](https://travis-ci.com/gyrdym/ml_algo.svg?branch=master)](https://travis-ci.com/gyrdym/ml_algo)
[![Coverage Status](https://coveralls.io/repos/github/gyrdym/ml_algo/badge.svg?branch=master)](https://coveralls.io/github/gyrdym/ml_algo?branch=master)
[![pub package](https://img.shields.io/pub/v/ml_algo.svg)](https://pub.dartlang.org/packages/ml_algo)
[![Gitter Chat](https://badges.gitter.im/gyrdym/gyrdym.svg)](https://gitter.im/gyrdym/)

# Machine learning algorithms with dart

**Table of contents**
- [What for is the library?](#what-is-the-ml_algo-for)
- [The library's structure](#the-librarys-structure)
- [Usage](#usage)

## What is the ml_algo for?

The main purpose of the library - to give developers, interested both in Dart language and data science, native Dart 
implementation of machine learning algorithms. This library targeted to dart vm, so, to get smoothest experience with 
the lib, please, do not use it in a browser.

Following algorithms are implemented:
- Linear regression:
    - Gradient descent algorithm (batch, mini-batch, stochastic) with ridge regularization
    - Lasso regression (feature selection model)

- Linear classifier:
    - Logistic regression (with "one-vs-all" multiclass classification)
    - Softmax regression
    
## The library's structure

To provide main purposes of machine learning, the library exposes the following classes:

- [MLData](https://github.com/gyrdym/ml_algo/blob/master/lib/src/data_preprocessing/ml_data/ml_data.dart). Factory, that creates instances of 
different adapters for data. For example, one can create a csv reader, that makes work with csv data easier: you just 
need to point, where your dataset resides and then get features and labels in convenient data science friendly format.

- [CrossValidator](https://github.com/gyrdym/ml_algo/blob/master/lib/src/model_selection/cross_validator/cross_validator.dart). Factory, that creates 
instances of a cross validator. In a few words, this entity allows researchers to fit different [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) of machine learning
algorithms, assessing prediction quality on different parts of a dataset. [Wiki article](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) about cross validation process. 

- [LinearClassifier.logisticRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/linear_classifier.dart). A class,
that performs simplest linear classification. If you want to use this classifier for your data, please, make sure, that 
your data is [linearly separably](https://en.wikipedia.org/wiki/Linear_separability). Multiclass classification is also
supported (see [ovr classification](https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest))

- [LinearClassifier.softmaxRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/linear_classifier.dart). 
A class, that performs simplest linear multiclass classification. As well as for logistic regression, if you want to use 
this classifier for your data, please, make sure, that your data is [linearly separably](https://en.wikipedia.org/wiki/Linear_separability).

- [LinearRegressor.gradient](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/linear_regressor.dart). An algorithm, 
that performs geometry-based linear regression using [gradient vector](https://en.wikipedia.org/wiki/Gradient) of a cost 
function.

- [LinearRegressor.lasso](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/linear_regressor.dart) An algorithm, 
that performs feature selection along with regression process. It uses [coordinate descent optimization]() and [subgradient vector]() 
instead of [gradient descent optimization]() and [gradient vector]() like in `LinearRegressor.gradient` to provide 
regression. If you want to decide, which features are less important - go ahead and use this regressor. 

## Usage

### Real life example

Let's classify records from well-known dataset - [Pima Indians Diabets Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
via [Logistic regressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/linear_classifier.dart)

Import all necessary packages: 

````dart  
import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
````

Read `csv`-file `pima_indians_diabetes_database.csv` with test data. You can use csv from the library's 
[datasets directory](https://github.com/gyrdym/ml_algo/tree/master/datasets):
````dart
final data = MLData.fromCsvFile('datasets/pima_indians_diabetes_database.csv');
final features = await data.features;
final labels = await data.labels;
````

Data in this file is represented by 768 records and 8 features. Processed features are contained in a data structure of 
`MLMatrix` type and processed labels are contained in a data structure of `MLVector` type. To get 
more information about these types, please, visit [ml_linal repo](https://github.com/gyrdym/ml_linalg)

Then, we should create an instance of `CrossValidator` class for fitting [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) 
of our model
````dart
final validator = CrossValidator.KFold(numberOfFolds: 5);
````

All are set, so, we can perform our classification. For better hyperparameters fitting, let's create a loop in order to 
try each value of a chosen hyperparameter in a defined range:
````dart
final step = 0.001;
final limit = 0.6;
double maxAccuracy = -double.infinity;
double bestLearningRate = 0.0;
for (double rate = step; rate < limit; rate += step) {
  // ...
}
````    
Let's create a logistic regression classifier instance with stochastic gradient descent optimizer in the loop's body:
````dart
final logisticRegressor = LinearClassifier.logisticRegressor(
        iterationsLimit: 100,
        initialLearningRate: rate,
        learningRateType: LearningRateType.constant);
````

Evaluate our model via accuracy metric:
````dart
final accuracy = validator.evaluate(logisticRegressor, featuresMatrix, labels, MetricType.accuracy);
if (accuracy > maxAccuracy) {
  maxAccuracy = accuracy;
  bestLearningRate = rate;
}
````

Let's print score:
````dart
print('best accuracy on classification: ${(maxAccuracy * 100).toFixed(2)}');
print('best learning rate: ${bestLearningRate.toFixed(3)}');
````

Best model parameters search takes much time so far, so be patient. After the search is over, we will see something like 
this:

````
best acuracy on classification: 67.0%
best learning rate: 0.155
````

All the code above all together:
````dart
import 'dart:async';

import 'package:ml_algo/ml_algo.dart';

Future<double> logisticRegression() async {
  final data = CsvMLData.fromFile('datasets/pima_indians_diabetes_database.csv');
  final features = await data.features;
  final labels = await data.labels;

  final validator = CrossValidator.kFold(numberOfFolds: 5);

  final step = 0.001;
  final limit = 0.6;

  double maxAccuracy = -double.infinity;
  double bestLearningRate = 0.0;

  for (double rate = step; rate < limit; rate += step) {
    final logisticRegressor = LinearClassifier.logisticRegressor(
      iterationsLimit: 100,
      initialLearningRate: rate,
      learningRateType: LearningRateType.constant);
    final accuracy = validator.evaluate(logisticRegressor, features, labels, MetricType.accuracy);
    if (accuracy > maxAccuracy) {
      maxAccuracy = accuracy;
      bestLearningRate = rate;
    }
  }

  print('best accuracy on classification: ${(maxAccuracy * 100).toFixed(2)}');
  print('best learning rate: ${bestLearningRate.toFixed(3)}');
}
````

For more examples please see [examples folder](https://github.com/gyrdym/dart_ml/tree/master/example)

### Contacts
If you have questions, feel free to write me on 
 - [Facebook](https://www.facebook.com/ilya.gyrdymov)
 - [Linkedin](https://www.linkedin.com/in/gyrdym/)
