import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/predictors/predictor_interface.dart';
import 'package:dart_ml/src/validators/kfold_cross_validator.dart';
import 'package:dart_ml/src/estimators/estimator_interface.dart';

class LooCrossValidator {
  final KFoldCrossValidator _validator = new KFoldCrossValidator();
  VectorInterface validate(PredictorInterface predictor, List<VectorInterface> features, VectorInterface labels,
                           {EstimatorInterface estimator}) {
    return _validator.validate(predictor, features, labels, numberOfFolds: features.length, estimator: estimator);
  }
}
