part of 'package:dart_ml/src/interface.dart';

abstract class KFoldSplitter implements Splitter {
  void configure({int numberOfFolds});
}