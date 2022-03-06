[![Build Status](https://github.com/gyrdym/ml_algo/workflows/CI%20pipeline/badge.svg)](https://github.com/gyrdym/ml_algo/actions?query=branch%3Amaster+)
[![Coverage Status](https://coveralls.io/repos/github/gyrdym/ml_algo/badge.svg?branch=master)](https://coveralls.io/github/gyrdym/ml_algo?branch=master)
[![pub package](https://img.shields.io/pub/v/ml_algo.svg)](https://pub.dartlang.org/packages/ml_algo)
[![Gitter Chat](https://badges.gitter.im/gyrdym/gyrdym.svg)](https://gitter.im/gyrdym/)

# Machine learning algorithms for Dart developers

**Table of contents**

- [What is ml_algo for](#what-is-ml_algo-for)
- [The library's content](#the-librarys-content)
- [Examples](#examples)
    - [Logistic regression](#logistic-regression)
    - [Linear regression](#linear-regression)
- [Models retraining](#models-retraining)
- [Notes on gradient based optimisation algorithms](#a-couple-of-words-about-linear-models-which-use-gradient-optimisation-methods)



## What is ml_algo for?

The main purpose of the library is to give native Dart implementation of machine learning algorithms to those who are 
interested both in Dart language and data science. This library aims at Dart VM and Flutter, it's impossible to use 
it in the web applications.

## The library's content

- #### Model selection
    - [CrossValidator](https://github.com/gyrdym/ml_algo/blob/master/lib/src/model_selection/cross_validator/cross_validator.dart). 
    Factory that creates instances of cross validators. Cross validation allows researchers to fit different 
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
    A class that performs classification using `k nearest neighbours algorithm` - it makes prediction basing on 
    the first `k` closest observations to the given one.

- #### Regression algorithms
    - [LinearRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/linear_regressor/linear_regressor.dart). 
    A general class for finding a linear pattern in training data and predicts outcome as real numbers depending on the pattern.
    
    - [LinearRegressor.lasso](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/linear_regressor/linear_regressor.dart)
    Implementation of linear regression algorithm based on coordinate descent with lasso regularisation 
     

    - [KnnRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/knn_regressor/knn_regressor.dart)
    A class that makes prediction for each new observation basing on first `k` closest observations from 
    training data. It may catch non-linear pattern of the data.
    
For more information on the library's API, please visit [API reference](https://pub.dev/documentation/ml_algo/latest/ml_algo/ml_algo-library.html) 

## Examples

### Logistic regression

Let's classify records from well-known dataset - [Pima Indians Diabets Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
via [Logistic regressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/logistic_regressor/logistic_regressor.dart)

**Important note:**

Please pay attention to problems which classifiers and regressors exposed by the library solve. E.g. 
[Logistic regressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/logistic_regressor/logistic_regressor.dart)
solves only **binary classification** problem, and that means that you can't use this classifier with a dataset 
with more than two classes, keep that in mind - in order to find out more about regresseors and classifiers, please refer to
the [api documentation](https://pub.dev/documentation/ml_algo/latest/ml_algo/ml_algo-library.html) of the package

Import all necessary packages. First, it's needed to ensure if you have `ml_preprocessing` and `ml_dataframe` packages 
in your dependencies:

````
dependencies:
  ml_dataframe: ^0.5.1
  ml_preprocessing: ^6.0.0
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

Download the dataset from [Pima Indians Diabets Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

#### For a desktop application: 

Just provide a proper path to your downloaded file and use a function-factory `fromCsv` from `ml_dataframe` package to 
read the file:

````dart
final samples = await fromCsv('datasets/pima_indians_diabetes_database.csv');
````

#### For a flutter application:

Be sure that you have ml_dataframe package version at least 0.5.1 and ml_algo package version at least 16.0.0 
in your pubspec.yaml:

````
dependencies:
  ...
  ml_algo: ^16.0.0
  ml_dataframe: ^0.5.1
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

Data in this file is represented by 768 records and 8 features. 9th column is a label column, it contains either 0 or 1 
on each row. This column is our target - we should predict a class label for each observation. The column's name is
`class variable (0 or 1)`. Let's store it:

````dart
final targetColumnName = 'class variable (0 or 1)';
````

Now it's the time to prepare data splits. Since we have a smallish dataset (only 768 records), we can't afford to
split the data into just train and test sets and evaluate the model on them, the best approach in our case is Cross 
Validation. According to this, let's split the data in the following way using the library's [splitData](https://github.com/gyrdym/ml_algo/blob/master/lib/src/model_selection/split_data.dart) 
function:

```dart
final splits = splitData(samples, [0.7]);
final validationData = splits[0];
final testData = splits[1];
```

`splitData` accepts `DataFrame` instance as the first argument and ratio list as the second one. Now we have 70% of our
data as a validation set and 30% as a test set for evaluating generalization error.

### Set up a model selection algorithm 

Then we may create an instance of `CrossValidator` class to fit [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
of our model. We should pass validation data (our `validationData` variable), and a number of folds into CrossValidator 
constructor.
 
````dart
final validator = CrossValidator.kFold(validationData, numberOfFolds: 5);
````

Let's create a factory for the classifier with desired hyperparameters. We have to decide after the cross validation, 
if the selected hyperparametrs are good enough or not:

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
- `optimizerType` - type of optimization algorithm that will be used to learn coefficients of our model, this time we
decided to use vanilla gradient ascent algorithm
- `iterationsLimit` - number of learning iterations. Selected optimization algorithm (gradient ascent in our case) will 
be run this amount of times
- `learningRateType` - a strategy for learning rate update. In our case the learning rate will decrease after every 
iteration
- `batchSize` - size of data (in rows) that will be used per each iteration. As we have a really small dataset we may use
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

### Evaluate performance of the model

Assume, we chose really good hyperprameters. In order to validate this hypothesis let's use CrossValidator instance 
created before:

````dart
final scores = await validator.evaluate(createClassifier, MetricType.accuracy);
````

Since the CrossValidator instance returns a [Vector](https://github.com/gyrdym/ml_linalg/blob/master/lib/vector.dart) of scores as a result of our predictor evaluation, we may choose
any way to reduce all the collected scores to a single number, for instance we may use Vector's `mean` method:

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

Let's assess our hyperparameters on test set in order to evaluate the model's generalization error:

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
To do so we may store the model to a file as JSON:

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

Please note that all the hyperparameters that we used to generate the model are persisted as the model's readonly 
fields, and we can access it anytime:

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

Data in this file is represented by 505 records and 13 features. 14th column is a target. Since we use autoheader, the
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

`splitData` accepts `DataFrame` instance as the first argument and ratio list as the second one. Now we have 80% of our
data as a train set and 20% as a test set.

Let's train the model:

```dart
final model = LinearRegressor(trainData, targetName);
```

By default, `LinearRegressor` uses closed-form solution to train the model. It's possible to use a different solution type,
e.g. one can use gradient-based algorithm: 

```dart
final model = LinearRegressor(
  samples
  targetName,
  optimizerType: LinearOptimizerType.gradient,
  iterationsLimit: 90,
  learningRateType: LearningRateType.timeBased,
);
```

As you may noticed, we have to provide a bunch of hyperparameters in case of gradient-based regression.

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
  final samples = await fromCsv('datasets/housing.csv', headerExists: false, columnDelimiter: ' ');
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
  final samples = DataFrame.fromRawCsv(rawCsvContent, fieldDelimiter: ' ');
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

## Models retraining

Someday our previously shining model can degrade in terms of prediction accuracy - in this case we can retrain it. 
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
