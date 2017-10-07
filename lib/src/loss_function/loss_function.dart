part of 'package:dart_ml/src/interface.dart';

abstract class LossFunction {
  double loss(double predictedLabel, double originalLabel);
}
