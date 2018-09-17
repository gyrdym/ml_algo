### master is broken, sorry, I work on that

# Machine learning with dart

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

Data in this file is represented by 200 lines, every line contains 4 elements. First 3 elements of every line are features and the last one is label.
Let's extract features from the data. Declare utility method `extractFeatures`, that extracts 3 elements from every line:
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
final sgdRegressor = new GradientRegressor(type: GradientType.Stochastic);
````

Evaluate our model via MAPE-metric:
````dart
final scoreMAPE = validator.evaluate(sgdRegressor, features, labels, metric: MetricType.MAPE);
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
