# Machine learning with dart

Following models are implemented:
- Linear regression:
    - with stochastic gradient descent
    - with mini batch gradient descent
    - with batch gradient descent

- Linear classifier:
    - Logistic regression
    
## Usage

### A simple usage example (Linear regression with stochastic gradient descent):

Import all necessary packages: 

````dart  
import 'dart:io';
import 'dart:async';
import 'dart:convert';
import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;
````

Read `csv`-file `advertising.csv` with test data:
````dart
csv.CsvCodec csvCodec = new csv.CsvCodec();
Stream<List<int>> input = new File('example/advertising.csv').openRead();
List<List<num>> fields = (await input.transform(UTF8.decoder)
  .transform(csvCodec.decoder).toList() as List<List<num>>)
  .sublist(1);
````

Data in this file is represented by 200 lines, every line contains 4 elements. First 3 elements of every line are features and the last is label.
Let's extract features from the data. Declare utility method `extractFeatures`, which extracts 3 elements of an every line:
````dart
List<double> extractFeatures(item) => item.sublist(0, 3)
  .map((num feature) => feature.toDouble())
  .toList();
````

...and finally get all features:
```dart
List<Float32x4Vector> features = fields
  .map((List<num> item) => new Float32x4Vector.from(extractFeatures(item)))
  .toList(growable: false);
```

...and labels (last element of a every line)
````dart
List<double> labels = fields.map((List<num> item) => item.last.toDouble()).toList(growable: false);
````

Create an instance of `CrossValidator` class for evaluating quality of our predictor
````dart
final validator = new CrossValidator.KFold();
````

Create a linear regressor instance with stochastic gradient descent optimizer:
````dart
final sgdRegressor = new GradientRegressor();
````

Evaluate our model via RMSE-metric (default metric for cross validation):
````dart
final scoreRMSE = validator.evaluate(sgdRegressor, features, labels, metric: MetricType.RMSE);
````

...and via MAPE-metric:
````dart
final scoreMAPE = validator.evaluate(sgdRegressor, features, labels, metric: MetricType.MAPE);
````

Let's print score:
````dart
print("score (RMSE): ${scoreRMSE.mean()}");
print("score (MAPE): ${scoreMAPE.mean()}");
````

We will see something like this:
````
score (RMSE): 4.91429797944094
score (MAPE): 31.221150755882263
````

For more examples please see [examples folder](https://github.com/gyrdym/dart_ml/tree/master/example)
