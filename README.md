[![Build Status](https://travis-ci.com/gyrdym/ml_algo.svg?branch=master)](https://travis-ci.com/gyrdym/ml_algo)
[![Coverage Status](https://coveralls.io/repos/github/gyrdym/ml_algo/badge.svg?branch=master)](https://coveralls.io/github/gyrdym/ml_algo?branch=master)
[![pub package](https://img.shields.io/pub/v/ml_algo.svg)](https://pub.dartlang.org/packages/ml_algo)
[![Gitter Chat](https://badges.gitter.im/gyrdym/gyrdym.svg)](https://gitter.im/gyrdym/)

# Machine learning algorithms with dart

## What is the ml_algo for?

The main purpose of the library is to give native Dart implementation of machine learning algorithms to those who are 
interested both in Dart language and data science. This library targeted to the dart vm, so to get smoothest experience with 
the lib, please do not use it in a browser.

## The library's content

- #### Model selection
    - [CrossValidator](https://github.com/gyrdym/ml_algo/blob/master/lib/src/model_selection/cross_validator/cross_validator.dart). 
    Factory that creates instances of cross validators. Cross validation allows researchers to fit different 
    [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) of machine learning algorithms 
    assessing prediction quality on different parts of a dataset. 

- #### Classification algorithms
    - [LogisticRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/logistic_regressor/logistic_regressor.dart). 
    A class that performs linear binary classification of data. To use this kind of classifier your data have to be 
    [linearly separable](https://en.wikipedia.org/wiki/Linear_separability).

    - [SoftmaxRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/softmax_regressor/softmax_regressor.dart). 
    A class that performs linear multiclass classification of data. To use this kind of classifier your data have to be 
    [linearly separable](https://en.wikipedia.org/wiki/Linear_separability).
        
    - [DecisionTreeClassifier](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/decision_tree_classifier/decision_tree_classifier.dart)
    A class that performs classification using decision trees. May work with data with non-linear patterns.
    
    - [KnnClassifier](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/knn_classifier/knn_classifier.dart)
    A class that performs classification using `k nearest neighbours algorithm` - it makes prediction basing on 
    the first `k` closest observations to the given one.

- #### Regression algorithms
    - [LinearRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/linear_regressor/linear_regressor.dart). 
    A class that finds a linear pattern in training data and predicts outcome as real numbers depending on the pattern. 

    - [KnnRegressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/regressor/knn_regressor/knn_regressor.dart)
    A class that makes prediction for each new observation basing on first `k` closest observations from 
    training data. It may catch non-linear pattern of the data.
    
For more information on the library's API, please visit [API reference](https://pub.dev/documentation/ml_algo/latest/ml_algo/ml_algo-library.html) 

## Examples

### Logistic regression

Let's classify records from well-known dataset - [Pima Indians Diabets Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
via [Logistic regressor](https://github.com/gyrdym/ml_algo/blob/master/lib/src/classifier/logistic_regressor/logistic_regressor.dart)

Import all necessary packages. First, it's needed to ensure, if you have `ml_preprocessing` and `ml_dataframe` package 
in your dependencies:

````
dependencies:
  ml_dataframe: ^0.2.0
  ml_preprocessing: ^5.2.0
````

We need these repos to parse raw data in order to use it farther. For more details, please
visit [ml_preprocessing](https://github.com/gyrdym/ml_preprocessing) repository page.

````dart  
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_preprocessing/ml_preprocessing.dart';
````

Download dataset from [Pima Indians Diabets Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database) and 
read it (of course, you should provide a proper path to your downloaded file): 

````dart
final samples = await fromCsv('datasets/pima_indians_diabetes_database.csv', headerExists: true);
````

*For flutter developers: please, read the official flutter.dev article [Read and write files](https://flutter.dev/docs/cookbook/persistence/reading-writing-files) 
before manipulating with file system in order to build a correct path to your dataset*

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
    learningRateType: LearningRateType.decreasingAdaptive,
    batchSize: trainSamples.rows.length,
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
full-batch gradient ascent, that's why we used `trainSamples.rows.length` here - the total amount of data.
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

Assume, we chose good hyperprameters which can lead to a high-performant model. In order to validate our hypothesis let's 
use CrossValidator instance created before:

````dart
final scores = await validator.evaluate(createClassifier, MetricType.accuracy);
````

Since the CrossValidator's instance returns a [Vector](https://github.com/gyrdym/ml_linalg/blob/master/lib/vector.dart) of scores as a result of our predictor evaluation, we may choose
any way to reduce all the collected scores to a single number, for instance we may use Vector's `mean` method:

```dart
final accuracy = scores.mean();
```  

Let's print the score:
````dart
print('accuracy on k fold validation: ${accuracy.toStringAsFixed(2)}');
````

We will see something like this:

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

Seems, our model has a good generalization ability, and that means we may use it in the future.
To do so we may store the model to a file as JSON:

```dart
await classifier.saveAsJson('diabetes_classifier.json');
```

After that we can simply read the model from the file and make predictions:

```dart
import 'dart:io';

final file = File(fileName);
final encodedData = await file.readAsString();
final classifier = LogisticRegressor.fromJson(encodedData);
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

All the code above all together:

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
      learningRateType: LearningRateType.decreasingAdaptive,
      batchSize: trainSamples.rows.length,
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

The workflow with other predictors (SoftmaxRegressor, DecisionTreeClassifier and so on) is quite similar to the described
above for LogisticRegressor, feel free to experiment with other models. 

### Contacts
If you have questions, feel free to write me on 
 - [Facebook](https://www.facebook.com/ilya.gyrdymov)
 - [Linkedin](https://www.linkedin.com/in/gyrdym/)
