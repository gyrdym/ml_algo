import 'package:ml_algo/src/classifier/linear_classifier.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/helpers/get_probabilities.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

mixin LinearClassifierMixin implements LinearClassifier {
  @override
  DataFrame predictProbabilities(DataFrame features) {
    final processedFeatures = addInterceptIf(
      fitIntercept,
      features.toMatrix(),
      interceptScale,
    );

    final probabilities = getProbabilities(
      processedFeatures,
      coefficientsByClasses,
      linkFunction,
    );

    return DataFrame.fromMatrix(
      probabilities,
      header: classNames,
    );
  }
}
