import 'package:ml_algo/src/classifier/classifier.dart';
import 'package:ml_algo/src/helpers/add_intercept_if.dart';
import 'package:ml_algo/src/helpers/get_probabilities.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_linalg/matrix.dart';

mixin LinearClassifierMixin implements Classifier {
  LinkFunction get linkFunction;
  bool get fitIntercept;
  double get interceptScale;

  @override
  Matrix predictProbabilities(Matrix features) {
    final processedFeatures = addInterceptIf(fitIntercept, features,
        interceptScale);
    return getProbabilities(processedFeatures,
        coefficientsByClasses, linkFunction);
  }
}
