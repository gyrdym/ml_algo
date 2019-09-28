[![Build Status](https://travis-ci.com/gyrdym/ml_algo.svg?branch=master)](https://travis-ci.com/gyrdym/ml_algo)
[![Coverage Status](https://coveralls.io/repos/github/gyrdym/ml_algo/badge.svg?branch=master)](https://coveralls.io/github/gyrdym/ml_algo?branch=master)
[![pub package](https://img.shields.io/pub/v/ml_algo.svg)](https://pub.dartlang.org/packages/ml_algo)
[![Gitter Chat](https://badges.gitter.im/gyrdym/gyrdym.svg)](https://gitter.im/gyrdym/)

# Machine learning algorithms with dart

**Table of contents**
- [What for is the library?](#what-is-the-ml_algo-for)
- [The library's structure](#the-librarys-structure)
- [Examples](#examples)
    - [Logistic regression](#logistic-regression)
    - [Softmax regression](#softmax-regression)
    - [KNN regression](#k-nearest-neighbour-regression)

## What is the ml_algo for?

The main purpose of the library - to give developers, interested both in Dart language and data science, native Dart 
implementation of machine learning algorithms. This library targeted to dart vm, so, to get smoothest experience with 
the lib, please, do not use it in a browser.

**Following algorithms are implemented:**
- *Linear regression:*
    - Gradient descent based linear regression
    - Coordinate descent based linear regression

- *Linear classifier:*
    - Logistic regression
    - Softmax regression
    
- *Non-linear classifier:*
    - Decision tree classifier (majority-based prediction)
    
- *Non-parametric regression:*
    - KNN regression
    
## The library's structure

- #### Model selection
    - [CrossValidator](https://github.com/gyrdym/ml_algo/blob/master/lib/src/model_selection/cross_validator/cross_validator.dart). 
    Factory, that creates instances of a cross validator. Cross validation allows researchers to fit different 
    [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) of machine learning algorithms, 
    assessing prediction quality on different parts of a dataset. 

- #### Classification algorithms
    - ##### Linear classification
        - ###### Logistic regression
            An algorithm, that performs linear binary classification. 
            - [LogisticRegressor.gradient](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/linear/logistic_regressor/logistic_regressor.dart). 
            Logistic regression with gradient ascent optimization of log-likelihood cost function. To use this kind 
            of classifier your data have to be [linearly separable](https://en.wikipedia.org/wiki/Linear_separability).
            
            - [LogisticRegressor.coordinate](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/linear/logistic_regressor/logistic_regressor.dart). 
            Not implemented yet. Logistic regression with coordinate descent optimization of negated log-likelihood cost 
            function. Coordinate descent allows to do feature selection (aka `L1 regularization`) To use this kind of 
            classifier your data have to be [linearly separable](https://en.wikipedia.org/wiki/Linear_separability).
        
        - ##### Softmax regression
            An algorithm, that performs linear multiclass classification.
            - [SoftmaxRegressor.gradient](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/linear/softmax_regressor/softmax_regressor.dart). 
            Softmax regression with gradient ascent optimization of log-likelihood cost function. To use this kind 
            of classifier your data have to be [linearly separable](https://en.wikipedia.org/wiki/Linear_separability).
            
            - [SoftmaxRegressor.coordinate](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/linear/softmax_regressor/logistic_regressor.dart). 
            Not implemented yet. Softmax regression with coordinate descent optimization of negated log-likelihood cost 
            function. As in case of logistic regression, coordinate descent allows to do feature selection (aka 
            `L1 regularization`) To use this kind of classifier your data have to be 
            [linearly separable](https://en.wikipedia.org/wiki/Linear_separability).
    - ##### Nonlinear classification
        - ###### Decision tree classifier
            - [DecisionTreeClassifier.majority](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/non_linear/decision_tree/decision_tree_classifier.dart)

- #### Regression algorithms
    - ##### Linear regression
        - [LinearRegressor.gradient](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/linear_regressor.dart). A 
        well-known algorithm, that performs linear regression using [gradient vector](https://en.wikipedia.org/wiki/Gradient) of a cost 
        function.

        - [LinearRegressor.coordinate](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/linear_regressor.dart) An algorithm, 
        that uses coordinate descent in order to find optimal value of a cost function. Coordinate descent allows to 
        perform feature selection along with regression process (This technique often calls `Lasso regression`). 
    
    - ##### Nonlinear regression
        - [ParameterlessRegressor.knn](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/non_parametric_regressor.dart)
        An algorithm, that makes prediction for each new observation based on first `k` closest observations from training data.
        It has quite high computational complexity, but in the same time it may easily catch non-linear pattern of the data. 

## Examples

### Logistic regression

Let's classify records from well-known dataset - [Pima Indians Diabets Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
via [Logistic regressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/linear_classifier.dart)

Import all necessary packages. First, it's needed to ensure, if you have `ml_preprocessing` and `ml_dataframe` package 
in your dependencies:

````
dependencies:
  ml_dataframe: ^0.0.8
  ml_preprocessing: ^5.0.1
````

We need these repos to parse raw data in order to use it farther. For more details, please,
visit [ml_preprocessing](https://github.com/gyrdym/ml_preprocessing) repository page.

````dart  
import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
````

Download dataset from [Pima Indians Diabets Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database) and 
read it (of course, you should provide a proper path to your downloaded file): 

````dart
final dataFrame = await fromCsv('datasets/pima_indians_diabetes_database.csv', headerExists: true);
````

Data in this file is represented by 768 records and 8 features. 9th column is a label column, it contains either 0 or 1 
on each row. This column is our target - we should predict a class label for each observation. Therefore, we 
should point, where to get label values.
 
Then, we should create an instance of `CrossValidator` class for fitting [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
of our model
````dart
final validator = CrossValidator.KFold(numberOfFolds: 5);
````

All are set, so, we can do our classification.

Evaluate our model via accuracy metric:
````dart
final accuracy = validator.evaluate((trainFeatures, trainLabels) => 
    LogisticRegressor.gradient(
        dataFrame,
        targetName: 'class variable (0 or 1)',
        initialLearningRate: .8,
        iterationsLimit: 500,
        batchSize: 768,
        fitIntercept: true,
        interceptScale: .1,
        learningRateType: LearningRateType.constant), 
    features, labels, MetricType.accuracy);
````

Let's print score:
````dart
print('accuracy on classification: ${accuracy.toStringAsFixed(2)}');
````

We will see something like this:

````
acuracy on classification: 0.77
````

All the code above all together:
````dart
import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

Future main() async {
  final dataFrame = await fromCsv('datasets/pima_indians_diabetes_database.csv', headerExists: true);
  final validator = CrossValidator.kFold(numberOfFolds: 5);
  final accuracy = validator.evaluate((trainFeatures, trainLabels) => 
    LogisticRegressor.gradient(
        trainFeatures, trainLabels,
        initialLearningRate: .8,
        iterationsLimit: 500,
        batchSize: 768,
        fitIntercept: true,
        interceptScale: .1,
        learningRateType: LearningRateType.constant), 
    features, labels, MetricType.accuracy);

  print('accuracy on classification: ${accuracy.toStringFixed(2)}');
}
````

### Softmax regression
Let's classify another famous dataset - [Iris dataset](https://www.kaggle.com/uciml/iris). Data in this csv is separated into 3 classes - therefore we need
to use different approach to data classification - [Softmax regression](http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/).

As usual, start with data preparation. Before we start, we should update our pubspec's dependencies with [xrange](https://github.com/gyrdym/xrange)` 
library: 

````
dependencies:
    ...
    xrange: ^0.0.5
    ...
````

Download the file and read it:

````Dart
final data = DataFrame.fromCsv('datasets/iris.csv',
    labelName: 'Species',
    columns: [ZRange.closed(1, 5)],
    categories: {
      'Species': CategoricalDataEncoderType.oneHot,
    },
);

final features = await data.features;
final labels = await data.labels;
````

The csv database has 6 columns, but we need to get rid of the first column, because it contains just ID of every 
observation - it's absolutely useless data. So, as you may notice, we provided a columns range to exclude ID-column:

````Dart
columns: [ZRange.closed(1, 5)]
````

Also, since the label column 'Species' has categorical data, we encoded it to numerical format:

````Dart
categories: {
  'Species': CategoricalDataEncoderType.oneHot,
},
````

Next step - create a cross validator instance:

````Dart
final validator = CrossValidator.kFold(numberOfFolds: 5);
````

Evaluate quality of prediction:

````Dart
final accuracy = validator.evaluate((trainFeatures, trainLabels) => 
      LinearClassifier.softmaxRegressor(
          trainFeatures, trainLabels,
          initialLearningRate: 0.03,
          iterationsLimit: null,
          minWeightsUpdate: 1e-6,
          randomSeed: 46,
          learningRateType: LearningRateType.constant
      ), features, labels, MetricType.accuracy);

print('Iris dataset, softmax regression: accuracy is '
  '${accuracy.toStringAsFixed(2)}'); // It yields 0.93
````

Gather all the code above all together:

````Dart
import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:xrange/zrange.dart';

Future main() async {
  final data = DataFrame.fromCsv('datasets/iris.csv',
    labelName: 'Species',
    columns: [ZRange.closed(1, 5)],
    categories: {
      'Species': CategoricalDataEncoderType.oneHot,
    },
  );

  final features = await data.features;
  final labels = await data.labels;
  final validator = CrossValidator.kFold(numberOfFolds: 5);
  final accuracy = validator.evaluate((trainFeatures, trainLabels) => 
      LinearClassifier.softmaxRegressor(
          trainFeatures, trainLabels,
          initialLearningRate: 0.03,
          iterationsLimit: null,
          minWeightsUpdate: 1e-6,
          randomSeed: 46,
          learningRateType: LearningRateType.constant
      ), features, labels, MetricType.accuracy);

  print('Iris dataset, softmax regression: accuracy is '
      '${accuracy.toStringAsFixed(2)}');
}
````

### K nearest neighbour regression

Let's do some prediction with a well-known non-parametric regression algorithm - k nearest neighbours. Let's take a 
state of the art dataset - [boston housing](https://www.kaggle.com/c/boston-housing).

As usual, import all necessary packages

````dart
import 'dart:async';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
import 'package:xrange/zrange.dart';
````

and download and read the data

````dart
final data = DataFrame.fromCsv('lib/_datasets/housing.csv',
    headerExists: false,
    fieldDelimiter: ' ',
    labelIdx: 13,
);
````

As you can see, the dataset is headless, that means, that there is no a descriptive line in the beginning of the file,
hence we can just use the index-based approach to point, where the outcomes column resides (13 index in our case)

Extract features and labels

````dart
// As in example above, it's needed to normalize the matrix column-wise to reach computational stability and provide 
// uniform scale for all the values in the column
final features = (await data.features).mapColumns((column) => column.normalize());
final labels = await data.labels;
````

Create a cross-validator instance

````dart
final validator = CrossValidator.kFold(numberOfFolds: 5);
````

Let the `k` parameter be equal to `4`.

Assess a knn regressor with the chosen `k` value using MAPE metric

````dart
final error = validator.evaluate((trainFeatures, trainLabels) => 
  ParameterlessRegressor.knn(trainFeatures, trainLabels, k: 4), features, labels, MetricType.mape);
````

Let's print our error

````dart
print('MAPE error on k-fold validation: ${error.toStringAsFixed(2)}%'); // it yields approx. 6.18
````

### Contacts
If you have questions, feel free to write me on 
 - [Facebook](https://www.facebook.com/ilya.gyrdymov)
 - [Linkedin](https://www.linkedin.com/in/gyrdym/)
