part of 'package:dart_ml/src/interface.dart';

abstract class LearningRateGenerator {
  void init(double initialValue);
  double getNextValue();
  void stop();
}
