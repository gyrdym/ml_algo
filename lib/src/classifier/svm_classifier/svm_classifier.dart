import 'package:ml_algo/src/classifier/linear_classifier.dart';
import 'package:ml_algo/src/classifier/svm_classifier/svm_classifier_impl.dart';
import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/predictor/retrainable.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

abstract class SVMClassifier
    implements
        Assessable,
        Serializable,
        Retrainable<SVMClassifier>,
        LinearClassifier {
  factory SVMClassifier(
    DataFrame trainData,
    String targetName, {
    DType dtype,
    num learningRate,
    int iterationLimit,
    bool fitIntercept,
    num interceptScale,
    num negativeLabel,
    num positiveLabel,
  }) = SVMClassifierImpl;
}
