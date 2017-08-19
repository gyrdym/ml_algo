part of 'package:dart_ml/src/dart_ml.dart';

abstract class Splitter {
  Iterable<Iterable<int>> split(int numberOfSamples);
}