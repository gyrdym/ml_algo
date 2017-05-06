import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/predictors/predictor_interface.dart';

abstract class CrossValidatorInterface {
  List<double> validate(PredictorInterface predictor, List<VectorInterface> features, List<double> labels);
}