part of 'package:dart_ml/src/interface.dart';

abstract class Randomizer {
  ///returns random interval constrained by [lowerBound] (include) and [upperBound] (exclude)
  Iterable<int> getIntegerInterval(int lowerBound, int upperBound);

  ///returns random interval constrained by [lowerBound] (include) and [upperBound] (exclude)
  Iterable<double> getDoubleInterval(double lowerBound, double upperBound);

  ///returns random integer from interval that starts with [start] (include) and ends with [end] (exclude)
  int getIntegerFromInterval(int start, int end);

  ///returns random double from interval that starts with [start] (include) and ends with [end] (exclude)
  double getDoubleFromInterval(double start, double end);
}