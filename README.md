[![Build Status](https://travis-ci.com/gyrdym/ml_algo.svg?branch=master)](https://travis-ci.com/gyrdym/ml_algo)

# Machine learning algorithms with dart

Following algorithms are implemented:
- Linear regression:
    - gradient descent models (batch, mini-batch, stochastic) with ridge regularization
    - lasso model (feature selection model)

- Linear classifier:
    - Logistic regression (with "one-vs-all" multinomial classification)
    
## Usage

### A simple usage example (Linear regression with stochastic gradient descent):

Import all necessary packages: 

````dart  
import 'dart:io';
import 'dart:async';
import 'dart:convert';
import 'package:ml_algo/ml_algo.dart';
import 'package:csv/csv.dart' as csv;
````

Read `csv`-file `advertising.csv` with test data:
````dart
final csvCodec = csv.CsvCodec(eol: '\n');
final input = File('example/datasets/advertising.csv').openRead();
final fields = (await input.transform(utf8.decoder)
  .transform(csvCodec.decoder).toList())
  .sublist(1);
````

Data in this file is represented by 200 lines, every line contains 4 elements. First 3 elements of every line are features and the last one is label.
Let's extract features from the data. Declare utility method `extractFeatures`, that extracts 3 elements from every line:
````dart
List<double> extractFeatures(List<dynamic> item) => item.sublist(0, 3)
      .map((dynamic feature) => (feature as num).toDouble())
      .toList();
````

...and finally get all features:
```dart
final features = fields
  .map(extractFeatures)
  .toList(growable: false);
```

...and labels (last element of a every line)
````dart
final labels = Float32x4VectorFactory.from(fields.map((List<dynamic> item) => (item.last as num).toDouble()));
````

Create an instance of `CrossValidator` class for evaluating quality of our predictor
````dart
final validator = CrossValidator<Float32x4>.KFold();
````

Create a linear regressor instance with stochastic gradient descent optimizer:
````dart
final sgdRegressor = GradientRegressor(type: GradientType.stochastic, iterationLimit: 100000,
                         learningRate: 1e-5, learningRateType: LearningRateType.constant);
````

Evaluate our model via MAPE-metric:
````dart
final scoreMAPE = validator.evaluate(sgdRegressor, Float32x4Matrix.from(features), labels, metric: MetricType.mape);
````

Let's print score:
````dart
print("score (MAPE): ${scoreMAPE}");
````

We will see something like this:
````
score (MAPE): 31.221150755882263
````

For more examples please see [examples folder](https://github.com/gyrdym/dart_ml/tree/master/example)
