[![Build Status](https://github.com/gyrdym/ml_algo/workflows/CI%20pipeline/badge.svg)](https://github.com/gyrdym/ml_algo/actions?query=branch%3Amaster+)
[![Coverage Status](https://coveralls.io/repos/github/gyrdym/ml_algo/badge.svg?branch=master)](https://coveralls.io/github/gyrdym/ml_algo?branch=master)
[![pub package](https://img.shields.io/pub/v/ml_algo.svg)](https://pub.dartlang.org/packages/ml_algo)
[![Gitter Chat](https://badges.gitter.im/gyrdym/gyrdym.svg)](https://gitter.im/gyrdym/)

# Machine learning algorithms for Dart developers - ml_algo library

The library is a part of the ecosystem:

- [ml_algo library](https://github.com/gyrdym/ml_algo) - implementation of popular machine learning algorithms 
- [ml_preprocessing library](https://github.com/gyrdym/ml_preprocessing) - a library for data preprocessing
- [ml_linalg library](https://github.com/gyrdym/ml_linalg) - a library for linear algebra 
- [ml_dataframe library](https://github.com/gyrdym/ml_dataframe)- a library for storing and manipulating data 

**Table of contents**

- [What is ml_algo for](#what-is-ml_algo-for)
- [The library content](#the-library-content)
- [Examples](#examples)
    - [Logistic regression](#logistic-regression)
    - [Linear regression](#linear-regression)
    - [Decision tree-based classification](#decision-tree-based-classification)
    - [KDTree-based data retrieval](#kdtree-based-data-retrieval)
- [Models retraining](#models-retraining)
- [Notes on gradient-based optimisation algorithms](#a-couple-of-words-about-linear-models-which-use-gradient-optimisation-methods)



## What is ml_algo for?

The main purpose of the library is to give native Dart implementation of machine learning algorithms to those who are 
interested both in Dart language and data science. This library aims at Dart VM and Flutter, it's impossible to use 
it in web applications.

## The library content

- #### Model selection
    - [CrossValidator](https://github.com/gyrdym/ml_algo/blob/master/lib/src/model_selection/cross_validator/cross_validator.dart). 
    A factory that creates instances of cross validators. Cross-validation allows researchers to fit different 
    [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) of machine learning algorithms 
    assessing prediction quality on different parts of a dataset. 

- #### Classification algorithms
    - [LogisticRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/logistic_regressor/logistic_regressor.dart). 
    A class that performs linear binary classification of data. To use this kind of classifier your data has to be 
    [linearly separable](https://en.wikipedia.org/wiki/Linear_separability).

    - [SoftmaxRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/softmax_regressor/softmax_regressor.dart). 
    A class that performs linear multiclass classification of data. To use this kind of classifier your data has to be 
    [linearly separable](https://en.wikipedia.org/wiki/Linear_separability).
        
    - [DecisionTreeClassifier](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/decision_tree_classifier/decision_tree_classifier.dart)
    A class that performs classification using decision trees. May work with data with non-linear patterns.
    
    - [KnnClassifier](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/knn_classifier/knn_classifier.dart)
    A class that performs classification using `k nearest neighbours algorithm` - it makes predictions based on 
    the first `k` closest observations to the given one.

- #### Regression algorithms
    - [LinearRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/linear_regressor/linear_regressor.dart). 
    A general class for finding a linear pattern in training data and predicting outcomes as real numbers.
    
    - [LinearRegressor.lasso](https://github.com/gyrdym/ml_algo/blob/85f1e2f19b946beb2b594a62e0e3c999d1c31608/lib/src/regressor/linear_regressor/linear_regressor.dart#L219)
    Implementation of the linear regression algorithm based on coordinate descent with lasso regularisation
    
    - [LinearRegressor.SGD](https://github.com/gyrdym/ml_algo/blob/c0ffc71676c1ad14927448fe9bbf984a425ce27a/lib/src/regressor/linear_regressor/linear_regressor.dart#L322)
    Implementation of the linear regression algorithm based on stochastic gradient descent with L2 regularisation
     
    - [KnnRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/knn_regressor/knn_regressor.dart)
    A class that makes predictions for each new observation based on the first `k` closest observations from 
    training data. It may catch non-linear patterns of the data.
    
- #### Clustering and retrieval algorithms
    - [KDTree](https://github.com/gyrdym/ml_algo/blob/master/lib/src/retrieval/kd_tree/kd_tree.dart) An algorithm for
    efficient data retrieval.
    
For more information on the library's API, please visit the [API reference](https://pub.dev/documentation/ml_algo/latest/ml_algo/ml_algo-library.html) 

## Examples

### Logistic regression

Let's classify records from a well-known dataset - [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
via [Logistic regressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/logistic_regressor/logistic_regressor.dart)

**Important note:**

Please pay attention to problems that classifiers and regressors exposed by the library solve. For e.g., 
[Logistic regressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/logistic_regressor/logistic_regressor.dart)
solves only **binary classification** problems, and that means that you can't use this classifier with a dataset 
with more than two classes, keep that in mind - in order to find out more about regressors and classifiers, please refer to
the [API documentation](https://pub.dev/documentation/ml_algo/latest/ml_algo/ml_algo-library.html) of the package

Import all necessary packages. First, it's needed to ensure if you have `ml_preprocessing` and `ml_dataframe` packages 
in your dependencies:

````
dependencies:
  ml_dataframe: ^1.0.0
  ml_preprocessing: ^7.0.2
````

We need these repos to parse raw data in order to use it further. For more details, please
visit [ml_preprocessing](https://github.com/gyrdym/ml_preprocessing) repository page. 

**Important note:**

Regressors and classifiers exposed by the library do not handle strings, booleans and nulls, they can only deal with 
numbers! You necessarily need to convert all the improper values of your dataset to numbers, please refer to [ml_preprocessing](https://github.com/gyrdym/ml_preprocessing)
library to find out more about data preprocessing.

````dart  
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
````

### Read a dataset's file

Download the dataset from [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

#### For a desktop application: 

Just provide a proper path to your downloaded file and use a function-factory `fromCsv` from `ml_dataframe` package to 
read the file:

````dart
final samples = await fromCsv('datasets/pima_indians_diabetes_database.csv');
````

#### For a flutter application:

Be sure that you have ml_dataframe package version at least 1.0.0 and ml_algo package version at least 16.0.0 
in your pubspec.yaml:

````
dependencies:
  ...
  ml_algo: ^16.0.0
  ml_dataframe: ^1.0.0
  ...
````

Then it's needed to add the dataset to the flutter assets by adding the following config in the pubspec.yaml:

````
flutter:
  assets:
    - assets/datasets/pima_indians_diabetes_database.csv
````

You need to create the assets directory in the file system and put the dataset's file there. After that you 
can access the dataset:

```dart
import 'package:flutter/services.dart' show rootBundle;
import 'package:ml_dataframe/ml_dataframe.dart';

final rawCsvContent = await rootBundle.loadString('assets/datasets/pima_indians_diabetes_database.csv');
final samples = DataFrame.fromRawCsv(rawCsvContent);
```

### Prepare datasets for training and testing

Data in this file is represented by 768 records and 8 features. The 9th column is a label column, it contains either 0 or 1 
on each row. This column is our target - we should predict a class label for each observation. The column's name is
`class variable (0 or 1)`. Let's store it:

````dart
final targetColumnName = 'class variable (0 or 1)';
````

Now it's the time to prepare data splits. Since we have a smallish dataset (only 768 records), we can't afford to
split the data into just train and test sets and evaluate the model on them, the best approach in our case is Cross-Validation. 
According to this, let's split the data in the following way using the library's [splitData](https://github.com/gyrdym/ml_algo/blob/master/lib/src/model_selection/split_data.dart) 
function:

```dart
final splits = splitData(samples, [0.7]);
final validationData = splits[0];
final testData = splits[1];
```

`splitData` accepts a `DataFrame` instance as the first argument and ratio list as the second one. Now we have 70% of our
data as a validation set and 30% as a test set for evaluating generalization errors.

### Set up a model selection algorithm 

Then we may create an instance of `CrossValidator` class to fit the [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
of our model. We should pass validation data (our `validationData` variable), and a number of folds into CrossValidator 
constructor.
 
````dart
final validator = CrossValidator.kFold(validationData, numberOfFolds: 5);
````

Let's create a factory for the classifier with desired hyperparameters. We have to decide after the cross-validation 
if the selected hyperparameters are good enough or not:

```dart
final createClassifier = (DataFrame samples) =>
  LogisticRegressor(
    samples
    targetColumnName,
    optimizerType: LinearOptimizerType.gradient,
    iterationsLimit: 90,
    learningRateType: LearningRateType.timeBased,
    batchSize: samples.rows.length,
    probabilityThreshold: 0.7,
  );
```

Let's describe our hyperparameters:
- `optimizerType` - a type of optimization algorithm that will be used to learn coefficients of our model, this time we
decided to use a vanilla gradient ascent algorithm
- `iterationsLimit` - number of learning iterations. The selected optimization algorithm (gradient ascent in our case) will 
be cyclically run this amount of times
- `learningRateType` - a strategy for learning rate update. In our case, the learning rate will decrease after every 
iteration
- `batchSize` - the size of data (in rows) that will be used per each iteration. As we have a really small dataset we may use
full-batch gradient ascent, that's why we used `samples.rows.length` here - the total amount of data.
- `probabilityThreshold` - lower bound for positive label probability

If we want to evaluate the learning process more thoroughly, we may pass `collectLearningData` argument to the classifier
constructor:

```dart
final createClassifier = (DataFrame samples) =>
  LogisticRegressor(
    ...,
    collectLearningData: true,
  );
```

This argument activates collecting costs per each optimization iteration, and you can see the cost values right after 
the model creation.

### Evaluate the performance of the model

Assume, we chose really good hyperparameters. In order to validate this hypothesis let's use CrossValidator instance 
created before:

````dart
final scores = await validator.evaluate(createClassifier, MetricType.accuracy);
````

Since the CrossValidator instance returns a [Vector](https://github.com/gyrdym/ml_linalg/blob/master/lib/vector.dart) of scores as a result of our predictor evaluation, we may choose
any way to reduce all the collected scores to a single number, for instance, we may use Vector's `mean` method:

```dart
final accuracy = scores.mean();
```  

Let's print the score:
````dart
print('accuracy on k fold validation: ${accuracy.toStringAsFixed(2)}');
````

We can see something like this:

````
accuracy on k fold validation: 0.65
````

Let's assess our hyperparameters on the test set in order to evaluate the model's generalization error:

```dart
final testSplits = splitData(testData, [0.8]);
final classifier = createClassifier(testSplits[0]);
final finalScore = classifier.assess(testSplits[1], MetricType.accuracy);
```

The final score is like:

```dart
print(finalScore.toStringAsFixed(2)); // approx. 0.75
```

If we specified `collectLearningData` parameter, we may see costs per each iteration in order to evaluate how our cost 
changed from iteration to iteration during the learning process:

```dart
print(classifier.costPerIteration);
```

### Write the model to a json file

Seems, our model has a good generalization ability, and that means we may use it in the future.
To do so we may store the model in a file as JSON:

```dart
await classifier.saveAsJson('diabetes_classifier.json');
```

After that we can simply read the model from the file and make predictions:

```dart
import 'dart:io';

final fileName = 'diabetes_classifier.json';
final file = File(fileName);
final encodedModel = await file.readAsString();
final classifier = LogisticRegressor.fromJson(encodedModel);
final unlabelledData = await fromCsv('some_unlabelled_data.csv');
final prediction = classifier.predict(unlabelledData);

print(prediction.header); // ('class variable (0 or 1)')
print(prediction.rows); // [ 
                        //   (1),
                        //   (0),
                        //   (0),
                        //   (1),
                        //   ...,
                        //   (1),
                        // ]
```

Please note that all the hyperparameters that we used to generate the model are persisted as the model's read-only 
fields, and we can access them anytime:

```dart
print(classifier.iterationsLimit);
print(classifier.probabilityThreshold);
// and so on
``` 

<details>
<summary>All the code for a desktop application:</summary>

````dart
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

void main() async {
  final samples = await fromCsv('datasets/pima_indians_diabetes_database.csv', headerExists: true);
  final targetColumnName = 'class variable (0 or 1)';
  final splits = splitData(samples, [0.7]);
  final validationData = splits[0];
  final testData = splits[1];
  final validator = CrossValidator.kFold(validationData, numberOfFolds: 5);
  final createClassifier = (DataFrame samples) =>
    LogisticRegressor(
      samples
      targetColumnName,
      optimizerType: LinearOptimizerType.gradient,
      iterationsLimit: 90,
      learningRateType: LearningRateType.timeBased,
      batchSize: samples.rows.length,
      probabilityThreshold: 0.7,
    );
  final scores = await validator.evaluate(createClassifier, MetricType.accuracy);
  final accuracy = scores.mean();
  
  print('accuracy on k fold validation: ${accuracy.toStringAsFixed(2)}');

  final testSplits = splitData(testData, [0.8]);
  final classifier = createClassifier(testSplits[0], targetNames);
  final finalScore = classifier.assess(testSplits[1], targetNames, MetricType.accuracy);
  
  print(finalScore.toStringAsFixed(2));

  await classifier.saveAsJson('diabetes_classifier.json');
}
````
</details>

<details>
<summary>All the code for a flutter application:</summary>

````dart
import 'package:flutter/services.dart' show rootBundle;
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

void main() async {
  final rawCsvContent = await rootBundle.loadString('assets/datasets/pima_indians_diabetes_database.csv');
  final samples = DataFrame.fromRawCsv(rawCsvContent);
  final targetColumnName = 'class variable (0 or 1)';
  final splits = splitData(samples, [0.7]);
  final validationData = splits[0];
  final testData = splits[1];
  final validator = CrossValidator.kFold(validationData, numberOfFolds: 5);
  final createClassifier = (DataFrame samples) =>
    LogisticRegressor(
      samples
      targetColumnName,
      optimizerType: LinearOptimizerType.gradient,
      iterationsLimit: 90,
      learningRateType: LearningRateType.timeBased,
      batchSize: samples.rows.length,
      probabilityThreshold: 0.7,
    );
  final scores = await validator.evaluate(createClassifier, MetricType.accuracy);
  final accuracy = scores.mean();
  
  print('accuracy on k fold validation: ${accuracy.toStringAsFixed(2)}');

  final testSplits = splitData(testData, [0.8]);
  final classifier = createClassifier(testSplits[0], targetNames);
  final finalScore = classifier.assess(testSplits[1], targetNames, MetricType.accuracy);
  
  print(finalScore.toStringAsFixed(2));

  await classifier.saveAsJson('diabetes_classifier.json');
}
````
</details>

### Linear regression

Let's try to predict house prices using linear regression and the famous [Boston Housing](https://www.kaggle.com/c/boston-housing) dataset.
The dataset contains 13 independent variables and 1 dependent variable - `medv` which is the target one (you can find
the dataset in [e2e/_datasets/housing.csv](https://github.com/gyrdym/ml_algo/blob/master/e2e/_datasets/housing.csv)).

Again, first we need to download the file and create a dataframe. The dataset is headless, we may either use autoheader or provide our own header. 
Let's use autoheader in our example:

#### For a desktop application: 

Just provide a proper path to your downloaded file and use a function-factory `fromCsv` from `ml_dataframe` package to 
read the file:

```dart
final samples = await fromCsv('datasets/housing.csv', headerExists: false, columnDelimiter: ' ');
``` 

#### For a flutter application:

It's needed to add the dataset to the flutter assets by adding the following config in the pubspec.yaml:

````
flutter:
  assets:
    - assets/datasets/housing.csv
````

You need to create the assets directory in the file system and put the dataset's file there. After that you 
can access the dataset:

```dart
import 'package:flutter/services.dart' show rootBundle;
import 'package:ml_dataframe/ml_dataframe.dart';

final rawCsvContent = await rootBundle.loadString('assets/datasets/housing.csv');
final samples = DataFrame.fromRawCsv(rawCsvContent, fieldDelimiter: ' ');
```

### Prepare the dataset for training and testing

Data in this file is represented by 505 records and 13 features. The 14th column is a target. Since we use autoheader, the
target's name is autogenerated and it is `col_13`. Let's store it in a variable:

````dart
final targetName = 'col_13';
````

then let's shuffle the data:

```dart
samples.shuffle();
```

Now it's the time to prepare data splits. Let's split the data into train and test subsets using the library's [splitData](https://github.com/gyrdym/ml_algo/blob/master/lib/src/model_selection/split_data.dart) 
function:

```dart
final splits = splitData(samples, [0.8]);
final trainData = splits[0];
final testData = splits[1];
```

`splitData` accepts a `DataFrame` instance as the first argument and ratio list as the second one. Now we have 80% of our
data as a train set and 20% as a test set.

Let's train the model:

```dart
final model = LinearRegressor(trainData, targetName);
```

By default, `LinearRegressor` uses a closed-form solution to train the model. One can also use a different solution type,
e.g. stochastic gradient descent algorithm: 

```dart
final model = LinearRegressor.SGD(
  samples
  targetName,
  iterationLimit: 90,
);
```

or linear regression based on coordinate descent with Lasso regularization:

```dart
final model = LinearRegressor.lasso(
  samples
  targetName,
  iterationLimit: 90,
);
```

Next, we should evaluate performance of our model:

```dart
final error = model.assess(testData, MetricType.mape);

print(error);
``` 

If we are fine with the error, we can save the model for the future use:

```dart
await model.saveAsJson('housing_model.json');
```

Later we may use our trained model for prediction:

```dart
import 'dart:io';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

final file = File('housing_model.json');
final encodedModel = await file.readAsString();
final model = LinearRegressor.fromJson(encodedModel);
final unlabelledData = await fromCsv('some_unlabelled_data.csv');
final prediction = model.predict(unlabelledData);

print(prediction.header);
print(prediction.rows);
```

<details>
<summary>All the code for a desktop application:</summary>

````dart
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

void main() async {
  final samples = (await fromCsv('datasets/housing.csv', headerExists: false, columnDelimiter: ' '))
    ..shuffle();
  final targetName = 'col_13';
  final splits = splitData(samples, [0.8]);
  final trainData = splits[0];
  final testData = splits[1];
  final model = LinearRegressor(trainData, targetName);
  final error = model.assess(testData, MetricType.mape);
  
  print(error);

  await classifier.saveAsJson('housing_model.json');
}
````
</details>

<details>
<summary>All the code for a flutter application:</summary>

````dart
import 'package:flutter/services.dart' show rootBundle;
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

void main() async {
  final rawCsvContent = await rootBundle.loadString('assets/datasets/pima_indians_diabetes_database.csv');
  final samples = DataFrame.fromRawCsv(rawCsvContent, fieldDelimiter: ' ')
    ..shuffle();
  final targetName = 'col_13';
  final splits = splitData(samples, [0.8]);
  final trainData = splits[0];
  final testData = splits[1];
  final model = LinearRegressor(trainData, targetName);
  final error = model.assess(testData, MetricType.mape);
    
  print(error);
  
  await classifier.saveAsJson('housing_model.json');
}
````
</details>

### Decision tree-based classification

Let's try to classify data from a well-known [Iris](https://www.kaggle.com/datasets/uciml/iris) dataset using a non-linear algorithm - [decision trees](https://en.wikipedia.org/wiki/Decision_tree)

First, you need to download the data and place it in a proper place in your file system. To do so you should follow the
instructions which are given in the [Logistic regression](#logistic-regression) section.

After loading the data, it's needed to preprocess it. We should drop the `Id` column since the column doesn't make sense. 
Also, we need to encode the 'Species' column - originally, it contains 3 repeated string labels, to feed it to the classifier
it's needed to convert the labels into numbers:

```dart
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';

void main() async {
    final samples = (await fromCsv('path/to/iris/dataset.csv'))
      .shuffle()
      .dropSeries(seriesNames: ['Id']);
    
    final pipeline = Pipeline(samples, [
      encodeAsIntegerLabels(
        featureNames: ['Species'], // Here we convert strings from 'Species' column into numbers
      ),
    ]);
}
```

Next, let's create a model:

```dart
final model = DecisionTreeClassifier(
  processed,
  'Species',
  minError: 0.3,
  minSamplesCount: 5,
  maxDepth: 4,
);
``` 

As you can see, we specified 3 hyperparameters: `minError`, `minSamplesCount` and `maxDepth`. Let's look at the 
parameters in more detail:

- `minError`. A minimum error on a tree node. If the error is less than or equal to the value, the node is considered a leaf.
- `minSamplesCount`. A minimum number of samples on a node. If the number of samples is less than or equal to the value, the node is considered a leaf.
- `maxDepth`. A maximum depth of the resulting decision tree. Once the tree reaches the `maxDepth`, all the level's nodes are considered leaves.

All the parameters serve as stopping criteria for the tree building algorithm.

Now we have a ready to use model. As usual, we can save the model to a JSON file:

```dart
await model.saveAsJson('path/to/json/file.json');
```

Unlike other models, in the case of a decision tree, we can visualise the algorithm result - we can save the model as an SVG file:

```dart
await model.saveAsSvg('path/to/svg/file.svg');
```

Once we saved it, we can open the file through any image viewer, e.g. through a web browser. An example of the 
resulting SVG image:

<p align="center">
    <img height="600" src="https://raw.github.com/gyrdym/ml_algo/master/e2e/decision_tree_classifier/iris_tree.svg?sanitize=true"> 
</p>

### KDTree-based data retrieval

Let's take a look at another field of machine learning - data retrieval. The field is represented by a family of algorithms,
one of them is `KDTree` which is exposed by the library.

`KDTree` is an algorithm that divides the whole search space into partitions in form of the binary tree which makes it 
efficient to retrieve data.

Let's retrieve some data points through a kd-tree built on the [Iris](https://www.kaggle.com/datasets/uciml/iris) dataset.

First, we need to prepare the data. To do so, it's needed to load the dataset. For this purpose, we may use 
[loadIrisDataset](https://pub.dev/documentation/ml_dataframe/latest/ml_dataframe/loadIrisDataset.html) function from [ml_dataframe](https://pub.dev/packages/ml_dataframe). The function returns prefilled with the Iris data DataFrame instance:

```dart
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

void main() async {
  final originalData = await loadIrisDataset();
}
```

Since the dataset contains `Id` column that doesn't make sense and `Species` column that contains text data, we need to
drop these columns:

```dart
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

void main() async {
  final originalData = await loadIrisDataset();
  final data = originalData.dropSeries(names: ['Id', 'Species']);
}
```

Next, we can build the tree:

```dart
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

void main() async {
  final originalData = await loadIrisDataset();
  final data = originalData.dropSeries(names: ['Id', 'Species']);
  final tree = KDTree(data);
}
```

And query nearest neighbours for an arbitrary point. Let's say, we want to find 5 nearest neighbours for the point `[6.5, 3.01, 4.5, 1.5]`:

```dart
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/vector.dart';

void main() async {
  final originalData = await loadIrisDataset();
  final data = originalData.dropSeries(names: ['Id', 'Species']);
  final tree = KDTree(data);
  final neighbourCount = 5;
  final point = Vector.fromList([6.5, 3.01, 4.5, 1.5]);
  final neighbours = tree.query(point, neighbourCount);
 
  print(neighbours);
}
```

The last instruction prints the following:

```
(Index: 75, Distance: 0.17349341930302867), (Index: 51, Distance: 0.21470911402365767), (Index: 65, Distance: 0.26095956499211426), (Index: 86, Distance: 0.29681616124778537), (Index: 56, Distance: 0.4172527193942372))
```

The nearest point has an index 75 in the original data. Let's check a record at the index:

```dart
import 'package:ml_dataframe/ml_dataframe.dart';

void main() async {
  final originalData = await loadIrisDataset();
 
  print(originalData.rows.elementAt(75));
}
```

It prints the following:

```
(76, 6.6, 3.0, 4.4, 1.4, Iris-versicolor)
```

Remember, we dropped `Id` and `Species` columns which are the very first and the very last elements in the output, so the
rest elements, `6.6, 3.0, 4.4, 1.4` look quite similar to our target point - `6.5, 3.01, 4.5, 1.5`, so the query result makes 
sense. 

If you want to use `KDTree` outside the ml_algo ecosystem, meaning you don't want to use [ml_linalg](https://pub.dev/packages/ml_linalg) and [ml_dataframe](https://pub.dev/packages/ml_dataframe)
packages in your application, you may import only [KDTree](https://pub.dev/documentation/ml_algo/latest/kd_tree/kd_tree-library.html) library and use [fromIterable](https://pub.dev/documentation/ml_algo/latest/kd_tree/KDTree/KDTree.fromIterable.html) constructor and [queryIterable](https://pub.dev/documentation/ml_algo/latest/kd_tree/KDTree/queryIterable.html)
method to perform the query: 

```dart
import 'package:ml_algo/kd_tree.dart';

void main() async {
  final tree = KDTree.fromIterable([
    // some data here
  ]);
  final neighbourCount = 5;
  final neighbours = tree.queryIterable([/* some point here */], neighbourCount);
 
  print(neighbours);
}
```

As usual, we can persist our tree by saving it to a JSON file:

```dart
import 'dart:io';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

void main() async {
  final originalData = await loadIrisDataset();
  final data = originalData.dropSeries(names: ['Id', 'Species']);
  final tree = KDTree(data);
 
  // ...

  await tree.saveAsJson('path/to/json/file.json');
 
  // ...

  final file = File('path/to/json/file.json');
  final encodedTree = await file.readAsString();
  final restoredTree = KDTree.fromJson(encodedTree);

  print(restoredTree);
}
```

## Models retraining

Someday our previously shining model can degrade in terms of prediction accuracy - in this case, we can retrain it. 
Retraining means simply re-running the same learning algorithm that was used to generate our current model
keeping the same hyperparameters but using a new data set with the same features:

```dart
import 'dart:io';

final fileName = 'diabetes_classifier.json';
final file = File(fileName);
final encodedModel = await file.readAsString();
final classifier = LogisticRegressor.fromJson(encodedModel);

// ... 
// here we do something and realize that our classifier performance is not so good
// ...

final newData = await fromCsv('path/to/dataset/with/new/data/to/retrain/the/classifier');
final retrainedClassifier = classifier.retrain(newData);

```

The workflow with other predictors (SoftmaxRegressor, DecisionTreeClassifier and so on) is quite similar to the described
above for LogisticRegressor, feel free to experiment with other models.

## A couple of words about linear models which use gradient optimisation methods

Sometimes you may get NaN or Infinity as a value of your score, or it may be equal to some inconceivable value 
(extremely big or extremely low). To prevent so, you need to find a proper value of the initial learning rate, and also 
you may choose between the following learning rate strategies: `constant`, `timeBased`, `stepBased` and `exponential`:

```dart
final createClassifier = (DataFrame samples) =>
    LogisticRegressor(
      ...,
      initialLearningRate: 1e-5,
      learningRateType: LearningRateType.timeBased,
      ...,
    );
```

### Contacts
If you have questions, feel free to text me on
 - [Twitter](https://twitter.com/ilgyrd) 
 - [Facebook](https://www.facebook.com/ilya.gyrdymov)
 - [Linkedin](https://www.linkedin.com/in/gyrdym/)
