import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/estimators/estimator_interface.dart';
import 'package:dart_ml/src/predictors/predictor_interface.dart';

abstract class CrossValidatorInterface {
  VectorInterface validate(PredictorInterface predictor, List<VectorInterface> features, VectorInterface labels,
                           {EstimatorInterface estimator});
}