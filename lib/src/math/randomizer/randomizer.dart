part of 'package:dart_ml/src/interface.dart';

abstract class Randomizer {
  int getIntegerFromInterval(int start, int end);
  double getDoubleFromInterval(double start, double end);
}