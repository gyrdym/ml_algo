import 'package:ml_algo/src/regressor/gradient_type.dart';

int getBatchSize(GradientType gradientType, int fallbackValue) {
  switch (gradientType) {
    case GradientType.miniBatch:
      return fallbackValue;

    case GradientType.batch:
      return double.maxFinite.floor();

    case GradientType.stochastic:
    default:
      return 1;
  }
}