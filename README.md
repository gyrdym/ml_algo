[![Build Status](https://travis-ci.com/gyrdym/ml_algo.svg?branch=master)](https://travis-ci.com/gyrdym/ml_algo)
[![Coverage Status](https://coveralls.io/repos/github/gyrdym/ml_algo/badge.svg?branch=master)](https://coveralls.io/github/gyrdym/ml_algo?branch=master)
[![pub package](https://img.shields.io/pub/v/ml_algo.svg)](https://pub.dartlang.org/packages/ml_algo)
[![Gitter Chat](https://badges.gitter.im/gyrdym/gyrdym.svg)](https://gitter.im/gyrdym/)

# Machine learning algorithms with dart

## What is the ml_algo for?

The main purpose of the library - to give developers, interested both in Dart language and data science, native Dart 
implementation of machine learning algorithms. This library targeted to dart vm, so, to get smoothest experience with 
the lib, please, do not use it in a browser.

## The library's content

- #### Model selection
    - [CrossValidator](https://github.com/gyrdym/ml_algo/blob/master/lib/src/model_selection/cross_validator/cross_validator.dart). 
    Factory, that creates instances of cross validators. Cross validation allows researchers to fit different 
    [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) of machine learning algorithms, 
    assessing prediction quality on different parts of a dataset. 

- #### Classification algorithms
    - [LogisticRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/logistic_regressor/logistic_regressor.dart). 
    A class, that performs linear binary classification of data. To use this kind of classifier your data have to be 
    [linearly separable](https://en.wikipedia.org/wiki/Linear_separability).

    - [SoftmaxRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/softmax_regressor/softmax_regressor.dart). 
    A class, that performs linear multiclass classification of data. To use this kind of classifier your data have to be 
    [linearly separable](https://en.wikipedia.org/wiki/Linear_separability).
        
    - [DecisionTreeClassifier](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/decision_tree_classifier/decision_tree_classifier.dart)
    A class, that performs classification, using decision trees. May work with data with non-linear patterns.
    
    - [KnnClassifier](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/knn_classifier/knn_classifier.dart)
    A class, that performs classification, using `k nearest neighbours algorithm` - it makes prediction basing on 
    first `k` closest observations to the given one.

- #### Regression algorithms
    - [LinearRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/linear_regressor/linear_regressor.dart). A 
    class, that finds a linear pattern in training data and predicts a real numbers depending on the pattern. 

    - [KnnRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/knn_regressor/knn_regressor.dart)
    A class, that makes prediction for each new observation basing on first `k` closest observations from 
    training data. It may catch non-linear pattern of the data. 

## Examples

### Logistic regression

Let's classify records from well-known dataset - [Pima Indians Diabets Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
via [Logistic regressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/linear_classifier.dart)

Import all necessary packages. First, it's needed to ensure, if you have `ml_preprocessing` and `ml_dataframe` package 
in your dependencies:

````
dependencies:
  ml_dataframe: ^0.0.11
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
final samples = await fromCsv('datasets/pima_indians_diabetes_database.csv', headerExists: true);
````

Data in this file is represented by 768 records and 8 features. 9th column is a label column, it contains either 0 or 1 
on each row. This column is our target - we should predict a class label for each observation. The column's name is
`class variable (0 or 1)`. Let's store it:

````dart
final targetColumnName = 'class variable (0 or 1)';
````
 
Then, we should create an instance of `CrossValidator` class for fitting [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
of our model. We should pass training data (our `samples` variable), a list of target column names (in our case it's 
just a name stored in `targetColumnName` variable) and a number of folds into CrossValidator constructor.
 
````dart
final validator = CrossValidator.KFold(samples, [targetColumnName], numberOfFolds: 5);
````

All are set, so, we can do our classification.

Evaluate our model via accuracy metric:

````dart
final accuracy = validator.evaluate((samples, targetNames) => 
    LogisticRegressor(
        samples,
        targetNames[0], // remember, we provided a list of just a single name
        optimizerType: LinearOptimizerType.gradient,  
        initialLearningRate: .8,
        iterationsLimit: 500,
        batchSize: samples.rows.length,
        fitIntercept: true,
        interceptScale: .1,
        learningRateType: LearningRateType.constant
    ), MetricType.accuracy);
````

Let's print the score:
````dart
print('accuracy on classification: ${accuracy.toStringAsFixed(2)}');
````

We will see something like this:

````
acuracy on classification: 0.77
````

All the code above all together:

````dart
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

Future main() async {
  final samples = await fromCsv('datasets/pima_indians_diabetes_database.csv', headerExists: true);
  final targetColumnName = 'class variable (0 or 1)';
  final validator = CrossValidator.KFold(samples, [targetColumnName], numberOfFolds: 5);
  final accuracy = validator.evaluate((samples, targetNames) => 
      LogisticRegressor(
          samples,
          targetNames[0], // remember, we provide a list of just a single name
          optimizerType: LinearOptimizerType.gradient,  
          initialLearningRate: .8,
          iterationsLimit: 500,
          batchSize: 768,
          fitIntercept: true,
          interceptScale: .1,
          learningRateType: LearningRateType.constant
      ), MetricType.accuracy);

  print('accuracy on classification: ${accuracy.toStringFixed(2)}');
}
````

### K nearest neighbour regression

Let's do some prediction with a well-known non-parametric regression algorithm - k nearest neighbours. Let's take a 
state of the art dataset - [boston housing](https://www.kaggle.com/c/boston-housing).

As usual, import all necessary packages

````dart
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
````

and download and read the data

````dart
final samples = await fromCsv('lib/_datasets/housing.csv',
    headerExists: false,
    fieldDelimiter: ' ',
);
````

As you can see, the dataset is headless, that means, that there is no a descriptive line in the beginning of the file.
So, we may use an autogenerated header in order to point, from what column we should take our target labels:

```dart
print(samples.header);
``` 

It will output the following:
````dart
(col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10, col_11, col_12, col_13)
````

Our target is `col_13`. Let's store it:

````dart
final targetColumnName = 'col_13';
````

Let's create a cross-validator instance:

````dart
final validator = CrossValidator.KFold(samples, [targetColumnName], numberOfFolds: 5);
````

Let the `k` parameter be equal to `4`.

Assess a knn regressor with the chosen `k` value using MAPE metric

````dart
final error = validator.evaluate((samples, targetNames) => 
  KnnRegressor(samples, targetNames[0], 4), MetricType.mape);
````

Let's print our error

````dart
print('MAPE error on k-fold validation: ${error.toStringAsFixed(2)}%'); // it yields approx. 6.18
````

### Contacts
If you have questions, feel free to write me on 
 - [Facebook](https://www.facebook.com/ilya.gyrdymov)
 - [Linkedin](https://www.linkedin.com/in/gyrdym/)
