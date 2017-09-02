part of 'package:dart_ml/src/interface.dart';

abstract class Randomizer {
  Iterable<int> getIntegerInterval(int lowerBound, int upperBound);
  Iterable<double> getDoubleInterval(double lowerBound, double upperBound);

  int getIntegerFromInterval(int start, int end);
  double getDoubleFromInterval(double start, double end);
}