import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

/// An interface for any classifier (linear, non-linear, parametric,
/// non-parametric, etc.)
abstract class Classifier extends Predictor {
  /// Returns predicted distribution of probabilities for each observation in
  /// the passed [testFeatures]
  DataFrame predictProbabilities(DataFrame testFeatures);

  /// A value using to encode positive class.
  ///
  /// Example:
  ///
  /// Given a positive class label equals 100
  ///
  /// Given a negative class label equals -100
  ///
  /// Given a dataset
  ///
  /// feature_1 | feature_2 | feature_3 | target class 1 | target class 2 | target class 3
  ///
  ///    123    |    233    |    444    |      100       |      -100      |     -100
  ///
  ///    333    |    100    |    101    |      100       |      -100      |     -100
  ///
  ///    321    |    911    |    321    |     -100       |       100      |     -100
  ///
  ///    221    |    987    |    222    |     -100       |      -100      |      100
  ///
  ///    908    |    404    |    503    |     -100       |       100      |     -100
  ///
  /// If a prediction algorithm meets 100 in a target column, it will
  /// interpret the value as a positive outcome for the corresponding class
  num get positiveLabel;

  /// A value using to encode negative class.
  ///
  /// Example:
  ///
  /// Given a positive class label equals 100
  ///
  /// Given a negative class label equals -100
  ///
  /// Given a dataset
  ///
  /// feature_1 | feature_2 | feature_3 | target class 1 | target class 2 | target class 3
  ///
  ///    123    |    233    |    444    |      100       |      -100      |     -100
  ///
  ///    333    |    100    |    101    |      100       |      -100      |     -100
  ///
  ///    321    |    911    |    321    |     -100       |       100      |     -100
  ///
  ///    221    |    987    |    222    |     -100       |      -100      |      100
  ///
  ///    908    |    404    |    503    |     -100       |       100      |     -100
  ///
  /// If a prediction algorithm meets -100 in a target column, it will
  /// interpret the value as a negative outcome for the corresponding class
  num get negativeLabel;
}
