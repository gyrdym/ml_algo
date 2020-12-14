import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_algo/src/predictor/retrainable.dart';

abstract class SerializablePredictor implements
    Assessable,
    Serializable,
    Retrainable,
    Predictor {}
