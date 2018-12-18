[![Build Status](https://travis-ci.com/gyrdym/ml_algo.svg?branch=master)](https://travis-ci.com/gyrdym/ml_algo)
[![pub package](https://img.shields.io/pub/v/ml_algo.svg)](https://pub.dartlang.org/packages/ml_algo)
[![Gitter Chat](https://badges.gitter.im/gyrdym/gyrdym.svg)](https://gitter.im/gyrdym/)

# Machine learning algorithms with dart

Following algorithms are implemented:
- Linear regression:
    - gradient descent algorithm (batch, mini-batch, stochastic) with ridge regularization
    - lasso regression (feature selection model)

- Linear classifier:
    - Logistic regression (with "one-vs-all" multinomial classification)
    
## Usage

### Real life example

Let's classify records from well-known dataset - [Pima Indians Diabets Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
via [Logistic regressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/logistic_regression.dart)

Import all necessary packages: 

````dart  
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:csv/csv.dart' as csv;
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/linalg.dart';
````

Read `csv`-file `pima_indians_diabetes_database.csv` with test data. You can use csv from the library's 
[datasets directory](https://github.com/gyrdym/ml_algo/tree/master/datasets):
````dart
final csvCodec = csv.CsvCodec(eol: '\n');
final input = File('datasets/pima_indians_diabetes_database.csv').openRead();
final fields = (await input.transform(utf8.decoder).transform(csvCodec.decoder).toList()).sublist(1);
````

Data in this file is represented by 768 records and 8 features. Let's do some data preprocessing.
Let's extract features from the data. Declare utility function `extractFeatures`:
````dart
final extractFeatures = (List<Object> item) => item.map((Object feature) => (feature as num).toDouble()).toList();
````

...and get all features:
```dart
final features = fields.map((List item) => extractFeatures(item.sublist(0, item.length - 1))).toList(growable: false);
```

For now, our features are just a list. For further processing we should create a matrix, based on the feature list:
````dart
final featuresMatrix = Float32x4Matrix.from(features);
````

To get more information about `Float32x4Matrix`, please, see [ml_linal repo](https://github.com/gyrdym/ml_linalg)

Also, we need to extract labels (last element of each line)
````dart
final labels = Float32x4Vector.from(fields.map((List<dynamic> item) => (item.last as num).toDouble()));
````

Then, we should create an instance of `CrossValidator` class for fitting [hyper parameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) 
of our model
````dart
final validator = CrossValidator<Float32x4>.KFold();
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
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:csv/csv.dart' as csv;
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_linalg/linalg.dart';

Future<double> main() async {
  final csvCodec = csv.CsvCodec(eol: '\n');
  final input = File('datasets/pima_indians_diabetes_database.csv').openRead();
  final fields = (await input.transform(utf8.decoder).transform(csvCodec.decoder).toList()).sublist(1);
  final extractFeatures = (List<Object> item) => item.map((Object feature) => (feature as num).toDouble()).toList();
  final features = fields.map((List item) => extractFeatures(item.sublist(0, item.length - 1))).toList(growable: false);
  final featuresMatrix = Float32x4Matrix.from(features);
  final labels = Float32x4Vector.from(fields.map((List<dynamic> item) => (item.last as num).toDouble()));
  final validator = CrossValidator<Float32x4>.kFold(numberOfFolds: 7);
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
    final error = validator.evaluate(logisticRegressor, featuresMatrix, labels, MetricType.accuracy);
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
