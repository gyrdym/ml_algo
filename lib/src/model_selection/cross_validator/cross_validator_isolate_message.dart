import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/metric/metric_type_json_converter.dart';
import 'package:ml_algo/src/model_selection/cross_validator/_helpers/decode_predictor.dart';
import 'package:ml_algo/src/model_selection/cross_validator/_helpers/decode_predictor_type.dart';
import 'package:ml_algo/src/model_selection/cross_validator/_helpers/encode_predictor_type.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator_isolate_message_json_keys.dart';
import 'package:ml_algo/src/model_selection/cross_validator/predictor_type.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

class CrossValidatorIsolateMessage {
  final Predictor predictorPrototype;
  final DataFrame trainData;
  final DataFrame testData;
  final PredictorType predictorType;
  final MetricType metricType;

  CrossValidatorIsolateMessage(
    this.predictorPrototype,
    this.trainData,
    this.testData,
    this.predictorType,
    this.metricType,
  );

  static CrossValidatorIsolateMessage fromJson(Map<String, dynamic> json) {
    final predictorType = decodePredictorType(
        json[predictorTypeJsonKey] as String);

    return CrossValidatorIsolateMessage(
      decodePredictor(
        predictorType,
        json[predictorPrototypeJsonKey] as Map<String, dynamic>,
      ),
      DataFrame.fromJson(json[trainDataJsonKey] as Map<String, dynamic>),
      DataFrame.fromJson(json[testDataJsonKey] as Map<String, dynamic>),
      predictorType,
      MetricTypeJsonConverter()
          .fromJson(json[metricTypeJsonKey] as String),
    );
  }

  Map<String, dynamic> toJson() => {
    predictorPrototypeJsonKey: predictorPrototype.toJson(),
    trainDataJsonKey: trainData.toJson(),
    testDataJsonKey: testData.toJson(),
    predictorTypeJsonKey: encodePredictorType(predictorType),
    metricTypeJsonKey: MetricTypeJsonConverter().toJson(metricType),
  };
}
