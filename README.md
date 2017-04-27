# Machine learning with dart

Only linear regression is available now (with Batch, mini-Batch and Stochastic gradient descent optimizers)

## Usage

### A simple usage example:

Import all necessary packages: 

````dart  
import 'dart:io';
import 'dart:async';
import 'dart:convert';
import 'package:dart_ml/dart_ml.dart';
import 'package:csv/csv.dart' as csv;
````

Read a `csv`-file `advertising.csv` with a test data:
````dart
csv.CsvCodec csvCodec = new csv.CsvCodec();
Stream<List<int>> input = new File('example/advertising.csv').openRead();
List<List<num>> fields = (await input.transform(UTF8.decoder)
  .transform(csvCodec.decoder).toList() as List<List<num>>)
  .sublist(1);
````

Data in this file is represented as 200 lines, every line contains 4 elements. 3 elements of every line are features and the last - label.  
Let's extract features from this data. Declare utility method `extractFeatures`, that extracts 3 elements of a every line: 
````dart
List<double> extractFeatures(item) => item.sublist(0, 3)
  .map((num feature) => feature.toDouble())
  .toList();
````

...and finally get all features:
```dart
List<TypedVector> allFeatures = fields
  .map((List<num> item) => new TypedVector.from(extractFeatures(item)))
  .toList(growable: false);
```

...and all labels (last element of a every line)
````dart
TypedVector allLabels = new TypedVector.from(fields.map((List<num> item) => item.last.toDouble()).toList());
````

Split features and labels into train and test samples:
````dart
Map<DataCategory, List<VectorInterface>> splittedFeatures = DataTrainTestSplitter.splitMatrix(allFeatures, .6);
Map<DataCategory, VectorInterface> splitedLabels = DataTrainTestSplitter.splitVector(allLabels, .6);

List<TypedVector> trainFeatures = splittedFeatures[DataCategory.TRAIN];
TypedVector trainLabels = splitedLabels[DataCategory.TRAIN];

List<TypedVector> testFeatures = splittedFeatures[DataCategory.TEST];
TypedVector testLabels = splitedLabels[DataCategory.TEST];
````

Create RMSE and MAPE estimator instances for evaluating a forecast quality:
````dart
RMSEEstimator rmseEstimator = new RMSEEstimator();
MAPEEstimator mapeEstimator = new MAPEEstimator();
````

Create linear regressor instance with stochastic gradient descent optimizer: 
````dart
SGDLinearRegressor sgdRegressor = new SGDLinearRegressor();
````

Train our model (third argument - initial vector for weights):
````dart
int dimension = trainFeatures.first.length;
sgdRegressor.train(trainFeatures, trainLabels, new TypedVector.filled(dimension, 0.0));
print("SGD regressor weights: ${sgdRegressor.weights}");
````

...and make a forecast (second argument - vector for storing predicted labels):
````dart
VectorInterface sgdPrediction = sgdRegressor.predict(testFeatures, new TypedVector.filled(testFeatures.length, 0.0));
````

Let's print an forecast evaluation:
````dart
print("SGD regressor, rmse (test) is: ${rmseEstimator.calculateError(sgdPrediction, testLabels)}");
print("SGD regressor, mape (test) is: ${mapeEstimator.calculateError(sgdPrediction, testLabels)}\n");
````

We will see something like this:
````
SGD regressor, rmse (test) is: 4.91429797944094
SGD regressor, mape (test) is: 31.221150755882263
````