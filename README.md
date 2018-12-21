[![Build Status](https://travis-ci.com/gyrdym/ml_algo.svg?branch=master)](https://travis-ci.com/gyrdym/ml_algo)
[![pub package](https://img.shields.io/pub/v/ml_algo.svg)](https://pub.dartlang.org/packages/ml_algo)
[![Gitter Chat](https://badges.gitter.im/gyrdym/gyrdym.svg)](https://gitter.im/gyrdym/)

# Machine learning algorithms with dart

**Table of contents**
- [Current state of the library](#current-state-of-the-library)
- [Key entities of the library](#key-entities-of-the-library)
- [Usage](#usage)

## Current state of the library

The main purpose of the library - give developers, interested both in Dart language and data science, native Dart 
implementation of machine learning algorithms. This library targeted to dart vm, so, to get smoothest experience with 
the lib, please, do not use it in a browser.

Following algorithms are implemented:
- Linear regression:
    - Gradient descent algorithm (batch, mini-batch, stochastic) with ridge regularization
    - Lasso regression (feature selection model)

- Linear classifier:
    - Logistic regression (with "one-vs-all" multinomial classification)
    
## Key entities of the library

To provide main purposes of machine learning, the library exposes the following classes:

-  [Float32x4CsvMLData](https://github.com/gyrdym/ml_algo/blob/master/lib/float32x4_csv_ml_data.dart). Factory, 
that creates instances of a csv reader. The reader makes work with csv data easier: you just need to point, where 
your dataset resides and then get features and labels in convenient data science friendly format. As you see, the reader 
converts data into sequence of numbers of [Float32x4](https://api.dartlang.org/stable/2.1.0/dart-typed_data/Float32x4-class.html) 
type, which makes machine learning process [faster](https://www.dartlang.org/articles/dart-vm/simd);

- [Float32x4CrossValidator](https://github.com/gyrdym/ml_algo/blob/master/lib/float32x4_cross_validator.dart). Factory, that creates instances of a cross validator. In a few words, this entity allows researchers to fit
different [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) of machine learning
algorithms, assessing prediction quality on different parts of a dataset. [Wiki article](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) about cross validation process. 

- [LogisticRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/logistic_regression.dart). A class,
that performs simplest linear classification. If you want to use this classifier for your data, please, make sure, that 
your data is [linearly separably](https://en.wikipedia.org/wiki/Linear_separability)

- [GradientRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/gradient.dart). A class, that 
performs geometry-based linear regression using [gradient vector](https://en.wikipedia.org/wiki/Gradient) of a cost 
function.

- [LassoRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/lasso.dart) A class, that performs
feature selection along with regression process. It uses [coordinate descent optimization]() and [subgradient vector]() 
instead of [gradient descent optimization]() and [gradient vector]() like in `GradientRegressor` to provide regression.
If you want to decide, which features are less important - go ahead and use this regressor. 

## Usage

### Real life example

Let's classify records from well-known dataset - [Pima Indians Diabets Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
via [Logistic regressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/logistic_regression.dart)

Import all necessary packages: 

````dart  
import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
````

Read `csv`-file `pima_indians_diabetes_database.csv` with test data. You can use csv from the library's 
[datasets directory](https://github.com/gyrdym/ml_algo/tree/master/datasets):
````dart
final data = Float32x4CsvMLData.fromFile('datasets/pima_indians_diabetes_database.csv');
final features = await data.features;
final labels = await data.labels;
````

Data in this file is represented by 768 records and 8 features. Processed features are contained in a data structure of 
`MLMatrix<Float32x4>` type and processed labels are contained in a data structure of `MLVector<Float32x4>` type. To get 
more information about these types, please, visit [ml_linal repo](https://github.com/gyrdym/ml_linalg)

Then, we should create an instance of `CrossValidator` class for fitting [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) 
of our model
````dart
final validator = CrossValidator.KFold();
````

All are set, so, we can perform our classification. For better hyperparameters fitting, let's create a loop in order to 
try each value of a chosen hyperparameter in a defined range:
````dart
final step = 0.001;
final limit = 0.6;
double minError = double.infinity;
double bestLearningRate = 0.0;
for (double rate = step; rate < limit; rate += step) {
  // ...
}
````    
Let's create a logistic regression classifier instance with stochastic gradient descent optimizer in the loop's body:
````dart
final logisticRegressor = LogisticRegressor(
        iterationLimit: 100,
        learningRate: rate,
        batchSize: 1,
        learningRateType: LearningRateType.constant,
        fitIntercept: true);
````

Evaluate our model via accuracy metric:
````dart
final error = validator.evaluate(logisticRegressor, featuresMatrix, labels, MetricType.accuracy);
if (error < minError) {
  minError = error;
  bestLearningRate = rate;
}
````

Let's print score:
````dart
print('best error on classification: ${(minError * 100).toFixed(2)}');
print('best learning rate: ${bestLearningRate.toFixed(3)}');
````

Best model parameters search takes much time so far, so be patient. After the search is over, we will see something like 
this:

````
best error on classification: 35.5%
best learning rate: 0.155
````

All the code above all together:
````dart
import 'dart:async';

import 'package:ml_algo/ml_algo.dart';

Future<double> logisticRegression() async {
  final data = Float32x4CsvMLData.fromFile('datasets/pima_indians_diabetes_database.csv');
  final features = await data.features;
  final labels = await data.labels;

  final validator = CrossValidator.kFold(numberOfFolds: 7);

  final step = 0.001;
  final limit = 0.6;

  double minError = double.infinity;
  double bestLearningRate = 0.0;

  for (double rate = step; rate < limit; rate += step) {
    final logisticRegressor = LogisticRegressor(
        iterationLimit: 100,
        learningRate: rate,
        batchSize: 1,
        learningRateType: LearningRateType.constant,
        fitIntercept: true);
    final error = validator.evaluate(logisticRegressor, features, labels, MetricType.accuracy);
    if (error < minError) {
      minError = error;
      bestLearningRate = rate;
    }
  }

  print('best error on classification: ${(minError * 100).toFixed(2)}');
  print('best learning rate: ${bestLearningRate.toFixed(3)}');
}
````

For more examples please see [examples folder](https://github.com/gyrdym/dart_ml/tree/master/example)

### Contacts
If you have questions, feel free to write me on 
 - [Facebook](https://www.facebook.com/ilya.gyrdymov)
 - [Linkedin](https://www.linkedin.com/in/gyrdym/)
