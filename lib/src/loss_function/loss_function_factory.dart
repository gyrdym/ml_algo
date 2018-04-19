import 'package:dart_ml/src/loss_function/cross_entropy.dart';
import 'package:dart_ml/src/loss_function/logistic_loss.dart';
import 'package:dart_ml/src/loss_function/loss_function.dart';
import 'package:dart_ml/src/loss_function/squared_loss.dart';

class LossFunctionFactory {
  static LossFunction Squared() => const SquaredLoss();
  static LossFunction CrossEntropy() => const CrossEntropyLoss();
  static LossFunction Logistic() => const LogisticLoss();
}