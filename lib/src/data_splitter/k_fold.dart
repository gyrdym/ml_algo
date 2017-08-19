part of 'package:dart_ml/src/dart_ml.dart';

abstract class KFoldSplitter implements Splitter {
  void configure({int numberOfFolds});
}