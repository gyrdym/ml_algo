import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/metric/metric_type.dart';
import 'package:ml_algo/src/model_selection/cross_validator/_helpers/from_serializable_predictor_Json.dart';
import 'package:ml_algo/src/model_selection/cross_validator/cross_validator_isolate_message_json_keys.dart';
import 'package:ml_algo/src/model_selection/serializable_predictor.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

part 'cross_validator_isolate_message.g.dart';

@JsonSerializable()
class CrossValidatorIsolateMessage {
  @JsonKey(
    name: predictorPrototypeJsonKey,
    fromJson: fromSerializablePredictorJson,
  )
  final SerializablePredictor predictorPrototype;

  @JsonKey(name: trainDataJsonKey)
  final DataFrame trainData;

  @JsonKey(name: testDataJsonKey)
  final DataFrame testData;

  @JsonKey(name: metricTypeJsonKey)
  final MetricType metricType;

  CrossValidatorIsolateMessage(
    this.predictorPrototype,
    this.trainData,
    this.testData,
    this.metricType,
  );

  factory CrossValidatorIsolateMessage.fromJson(Map<String, dynamic> json) =>
      _$CrossValidatorIsolateMessageFromJson(json);

  Map<String, dynamic> toJson() =>
      _$CrossValidatorIsolateMessageToJson(this);
}
